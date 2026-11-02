"""Local-only dataset, model registry and immutable run storage (Python 3.8+)."""
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import uuid
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
from PIL import Image, ImageOps

LABELS = ['강한','고급스러운','고전적인','귀여운','깔끔한','단순한','독특한','부드러운','섬세한','소박한','신비한','우아한','원초적인','유려한','익살스러운','자연적인','장식적인','조화로운','풍성한','현대적인','화려한','활동적인']
ID = re.compile(r'^[A-Za-z0-9_-]{1,80}$')
SPLITS = {'train','dev','eval'}
TRAIN_DEFAULTS = dict(epochs=10, batch=8, lr=0.0001)
LEGACY_WEIGHTS = ('dinov3_end_to_end_strict.pt', 'dinov3_pseudopretrain_strict.pt', 'image_residual.pt')

def supports_training(model):
    # Older manifests marked legacy models inference-only; the new engine can
    # continue their existing A/B heads without rewriting those source manifests.
    return model.get('engine')=='legacy_dino_rbf_v1' or model.get('training_supported',True)

def inspect_model_folder(folder):
    """Read only a selected local directory; do not follow links or search mounts."""
    p=local_only(folder)
    if not p.is_dir(): raise ValueError('가중치가 들어 있는 로컬 폴더를 선택하세요.')
    package_file=local_only(p/'package.json')
    spec=read(package_file) if package_file.is_file() else {}
    if spec and spec.get('engine') not in ('dino_head_v1','legacy_dino_rbf_v1','bert5_fusion_v1'):
        raise ValueError('지원하지 않는 모델 형식입니다.')
    if spec.get('engine')=='bert5_fusion_v1':
        required=['fusion_config.json','backbone/config.json','backbone/preprocessor_config.json','backbone/model.safetensors']
        required += ['weights/'+tag+'.pt' for tag in ('klue','xlmr','kcbert','mbert','kobigbird')]
        required += ['text/'+tag+'/config.json' for tag in ('klue','xlmr','kcbert','mbert','kobigbird')]
        mapping={};missing=[]
        for rel in required:
            path=local_only(p/rel)
            if not path.is_file():missing.append(rel)
            else:mapping[rel]=str(path)
        for tag in ('klue','xlmr','kcbert','mbert','kobigbird'):
            tokenfiles=[]
            for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','sentencepiece.bpe.model','added_tokens.json'):
                path=local_only(p/'text'/tag/name)
                if path.is_file():mapping['text/'+tag+'/'+name]=str(path);tokenfiles.append(name)
            if 'tokenizer.json' not in tokenfiles and 'vocab.txt' not in tokenfiles:missing.append('text/'+tag+'/tokenizer.json 또는 vocab.txt')
        if spec.get('labels')!=LABELS:raise ValueError('22개 감성어 순서가 다릅니다.')
        return dict(source=str(p),engine='bert5_fusion_v1',missing=missing,files=mapping,hashes={rel:sha(Path(src)) for rel,src in mapping.items()},
                    bytes=sum(Path(src).stat().st_size for src in mapping.values()),name=spec.get('name',p.name),has_head=True,
                    trained_pixel_hashes=spec.get('trained_pixel_hashes',[]),provenance=spec.get('provenance','이미지+설명문 5종 융합 모델; 학습 이력 미확인'))
    legacy=spec.get('engine')=='legacy_dino_rbf_v1' or any(local_only(p/sub/name).is_file() for sub in ('','weights','runtime_weights') for name in LEGACY_WEIGHTS)
    engine='legacy_dino_rbf_v1' if legacy else 'dino_head_v1'
    mapping={}; missing=[]
    for filename in LEGACY_WEIGHTS if legacy else ():
        matches=[local_only(p/sub/filename) for sub in ('','weights','runtime_weights') if local_only(p/sub/filename).is_file()]
        if len(matches)>1: raise ValueError(filename+'이 여러 위치에 있습니다. 사용할 파일 하나만 있는 폴더를 선택하세요.')
        if matches: mapping['weights/'+filename]=str(matches[0])
        else: missing.append(filename)
    candidates=[local_only(p/n) for n in ('backbone','dinov3l') if local_only(p/n/'config.json').is_file()]
    if not legacy and local_only(p/'config.json').is_file(): candidates.append(p)
    if len(candidates)>1: raise ValueError('backbone과 dinov3l 중 사용할 원본 백본 폴더 하나만 넣어 주세요.')
    if not candidates:
        missing.extend(['backbone 또는 dinov3l 폴더의 config.json', '원본 preprocessor_config.json', '원본 백본 model.safetensors (또는 전체 분할 파일)'])
    else:
        backbone=candidates[0]; config=read(local_only(backbone/'config.json'))
        if config.get('model_type')!='dinov3_vit': raise ValueError('DINOv3 ViT 백본만 지원합니다.')
        for filename in ('config.json','preprocessor_config.json'):
            path=local_only(backbone/filename)
            if not path.is_file(): missing.append('backbone/'+filename)
            else: mapping['backbone/'+filename]=str(path)
        index=local_only(backbone/'model.safetensors.index.json')
        if index.is_file():
            mapping['backbone/'+index.name]=str(index)
            weights=read(index).get('weight_map',{})
            if not weights: raise ValueError('백본 분할 목록이 비어 있습니다.')
            weight_names=sorted(set(weights.values()))
        else: weight_names=['model.safetensors']
        for name in weight_names:
            if not isinstance(name,str) or PurePosixPath(name).name!=name or '\\' in name or ':' in name or not name.endswith('.safetensors'):
                raise ValueError('백본 가중치 목록에 허용되지 않은 경로가 있습니다.')
            path=local_only(backbone/name)
            if not path.is_file(): missing.append('backbone/'+name)
            else: mapping['backbone/'+name]=str(path)
    if spec.get('labels',LABELS)!=LABELS: raise ValueError('22개 감성어 순서가 다릅니다.')
    head=local_only(p/'head.safetensors')
    if not legacy and (spec.get('has_head') or head.is_file()):
        if not head.is_file(): missing.append('head.safetensors')
        else: mapping['head.safetensors']=str(head)
    if not legacy and head.is_file() and not spec:
        raise ValueError('분류 헤드의 라벨 순서를 확인할 package.json이 필요합니다. MODEL_FORMAT.md를 확인하세요.')
    # Source identity is frozen at confirmation, and checked again after copying.
    hashes={rel:sha(Path(src)) for rel,src in mapping.items()}
    return dict(source=str(p),engine=engine,missing=missing,files=mapping,hashes=hashes,
        bytes=sum(Path(src).stat().st_size for src in mapping.values()),
        name=spec.get('name',p.name),has_head=legacy or bool(head.is_file()),
        trained_pixel_hashes=spec.get('trained_pixel_hashes',[]),
        provenance=spec.get('provenance','기존 DINO A/B+RBF: 원본 배포 평가 446장 중 223장 학습 재사용 기록. 선택 파일의 동일성·성능은 별도 검증 필요.' if legacy else '기존 학습 이력 미확인'))

def now(): return datetime.now(timezone.utc).isoformat()
def uid(prefix): return prefix + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    t.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(str(t), str(p))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def validate_id(x):
    if not isinstance(x,str) or not ID.fullmatch(x): raise ValueError('올바르지 않은 ID입니다.')
    return x
def local_only(p):
    """Refuse network storage, old mount trees and symlink/reparse components."""
    p = Path(os.path.abspath(str(p)))
    if str(p).startswith(('//','\\\\')) or str(p).startswith(('/home/mount/','/workspace/mount/')):
        raise ValueError('NAS·마운트 경로는 사용할 수 없습니다. 로컬 디스크를 선택하세요.')
    for a in [p] + list(p.parents):
        if a.is_symlink(): raise ValueError('바로가기 경로는 사용할 수 없습니다: '+str(a))
        if a.exists() and getattr(a.lstat(),'st_file_attributes',0) & 0x400:
            raise ValueError('Windows 연결 폴더(junction)는 사용할 수 없습니다.')
    if os.name == 'nt':
        import ctypes
        if ctypes.windll.kernel32.GetDriveTypeW(str(p.anchor)) != 3:
            raise ValueError('서버·네트워크·이동식 드라이브 대신 로컬 고정 디스크를 사용하세요.')
    elif Path('/proc/self/mountinfo').exists():
        matches=[]
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            left,right=line.split(' - ',1); mount=left.split()[4].replace('\\040',' ')
            if str(p)==mount or str(p).startswith(mount.rstrip('/')+'/'):
                matches.append((len(mount),right.split()[0]))
        if matches and (max(matches)[1] in {'nfs','nfs4','cifs','smb3','9p'} or max(matches)[1].startswith('fuse.')):
            raise ValueError('네트워크 또는 가상 연결 저장소는 사용할 수 없습니다.')
    return p

def pixel_hash(blob):
    with Image.open(io.BytesIO(blob)) as im:
        if im.width*im.height > 40_000_000: raise ValueError('이미지는 4천만 화소 이하로 등록하세요.')
        im = ImageOps.exif_transpose(im).convert('RGB')
        return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest(), im.size

def metric(scores, rows):
    hits=0; predictions=[]; tp=Counter(); gt_count=Counter(); pred_count=Counter()
    if len(scores)!=len(rows): raise ValueError('예측 개수와 표본 개수가 다릅니다.')
    for score,r in zip(scores,rows):
        score=[float(x) for x in score]
        if len(score)!=22 or not all(math.isfinite(x) for x in score): raise ValueError('유효하지 않은 모델 점수입니다.')
        top=sorted(range(22),key=lambda j:(-score[j],j))[:5]
        names=[LABELS[j] for j in top]; correct=set(names)&set(r['labels']); hits+=len(correct)
        tp.update(correct); gt_count.update(r['labels']); pred_count.update(names)
        predictions.append(dict(id=r['id'],dataset_id=r.get('dataset_id'),split=r['split'],GT=r['labels'],top5=names,scores=score))
    # Independent label-set recount, not rounding a reported percentage.
    assert hits==sum(len(set(p['top5'])&set(p['GT'])) for p in predictions)
    return dict(sample_count=len(rows),correct_slots=hits,possible_slots=len(rows)*5,
        f1_at_5=hits/(len(rows)*5) if rows else None,
        labels=[dict(label=s,true_positive=tp[s],gt_count=gt_count[s],predicted_count=pred_count[s]) for s in LABELS]),predictions

class Store:
    def __init__(self, root):
        self.root=local_only(root)
        for name in ('datasets','models','runs','imports','cache','dataset_trash','model_trash'):
            local_only(self.root/name).mkdir(parents=True,exist_ok=True)

    def item(self,kind,key):
        if kind not in ('datasets','models','runs'): raise ValueError('잘못된 자료 종류입니다.')
        return local_only(self.root/kind/validate_id(key))

    def dataset_deleted(self,key):
        marker=local_only(self.root/'dataset_trash'/(validate_id(key)+'.json'))
        return marker.is_file() and read(marker).get('status')=='deleted'

    def model_deleted(self,key):
        marker=local_only(self.root/'model_trash'/(validate_id(key)+'.json'))
        return marker.is_file() and read(marker).get('status') in ('deleted','purging','purged')

    def check_model_available(self,key):
        def check(current,chain):
            if current in chain: raise ValueError('모델 참조가 순환합니다.')
            marker=local_only(self.root/'model_trash'/(validate_id(current)+'.json'))
            if marker.is_file() and read(marker).get('status') in ('purging','purged'):
                raise ValueError('영구 삭제된 가중치입니다. 복원할 수 없습니다: '+current)
            if self.model_deleted(current): raise ValueError('삭제 보관함의 모델입니다. 먼저 복원하세요: '+current)
            model=read(self.item('models',current)/'manifest.json')
            for field in ('base_model','backbone_model'):
                if model.get(field):check(model[field],chain|{current})
        check(key,set())

    def list(self,kind,include_deleted=False):
        filename={'datasets':'manifest.json','models':'manifest.json','runs':'status.json'}[kind]
        out=[]
        for p in sorted((self.root/kind).iterdir(),reverse=True):
            if not p.is_dir() or p.is_symlink(): continue
            try:
                if kind=='datasets' and not include_deleted and self.dataset_deleted(p.name): continue
                if kind=='models' and not include_deleted and self.model_deleted(p.name): continue
                if kind=='models':
                    marker=local_only(self.root/'model_trash'/(validate_id(p.name)+'.json'))
                    if marker.is_file() and read(marker).get('status')=='purged': continue
                item=read(local_only(p/filename))
                if kind=='datasets': item={k:v for k,v in item.items() if k!='rows'}
                out.append(item)
            except (OSError,ValueError): continue
        return out

    def import_zip(self,blob,name):
        if not str(name).strip() or len(name)>100: raise ValueError('데이터 이름을 1~100자로 입력하세요.')
        if len(blob)>256*1024**2: raise ValueError('ZIP은 256MB 이하로 나누어 등록하세요.')
        archive=zipfile.ZipFile(io.BytesIO(blob)); infos=archive.infolist()
        if len(infos)>20002 or sum(x.file_size for x in infos)>2*1024**3: raise ValueError('압축 해제 크기 또는 파일 수 제한을 초과했습니다.')
        names=set()
        for x in infos:
            path=PurePosixPath(x.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in x.filename or ':' in x.filename or not path.parts:
                raise ValueError('압축파일에 허용되지 않은 경로가 있습니다.')
            if x.filename.casefold() in names: raise ValueError('대소문자만 다른 중복 파일명이 있습니다.')
            names.add(x.filename.casefold())
            mode=x.external_attr>>16
            if stat.S_ISLNK(mode): raise ValueError('압축파일 안의 바로가기는 허용하지 않습니다.')
            if x.file_size>32*1024**2: raise ValueError('개별 파일은 32MB 이하로 등록하세요.')
        if 'labels.csv' not in archive.namelist(): raise ValueError('ZIP 최상위에 labels.csv가 필요합니다.')
        rows=list(csv.DictReader(io.StringIO(archive.read('labels.csv').decode('utf-8-sig'))))
        if not rows or len(rows)>10000: raise ValueError('데이터는 1~10,000개 행이어야 합니다.')
        output=[]; ids=set(); pixels={}; images={}
        for i,r in enumerate(rows,2):
            sid=(r.get('id') or '').strip(); image=(r.get('image') or '').strip()
            labels=[x.strip() for x in (r.get('labels') or '').split(';') if x.strip()]
            split=(r.get('split') or '').strip()
            if not sid or len(sid)>160 or sid in ids: raise ValueError(f'{i}행: ID가 없거나 중복입니다.')
            if len(labels)!=5 or len(set(labels))!=5 or not set(labels)<=set(LABELS): raise ValueError(f'{i}행: 등록된 감성어 중 서로 다른 정답 5개가 필요합니다.')
            if split not in SPLITS: raise ValueError(f'{i}행: split은 train/dev/eval 중 하나여야 합니다.')
            if image not in archive.namelist() or PurePosixPath(image).suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}:
                raise ValueError(f'{i}행: 이미지 파일을 찾을 수 없습니다.')
            raw=archive.read(image); digest,size=pixel_hash(raw)
            if digest in pixels: raise ValueError(f'{i}행: {pixels[digest]}와 픽셀이 같은 중복 이미지입니다. GT와 분할을 검토하세요.')
            ids.add(sid); pixels[digest]=sid
            dest='images/'+hashlib.sha256(raw).hexdigest()+PurePosixPath(image).suffix.lower()
            images[dest]=raw
            output.append(dict(id=sid,image=dest,labels=labels,split=split,pixel_sha256=digest,
                file_sha256=hashlib.sha256(raw).hexdigest(),width=size[0],height=size[1],description=(r.get('description') or '')[:10000]))
        key=uid('dataset'); p=self.item('datasets',key); p.mkdir()
        (p/'images').mkdir()
        for n,raw in images.items(): (p/n).write_bytes(raw)
        (p/'original.zip').write_bytes(blob)
        counts=dict(Counter(r['split'] for r in output))
        record=dict(id=key,name=name,created=now(),counts=counts,count=len(output),labels=LABELS,
                    archive_sha256=hashlib.sha256(blob).hexdigest(),rows=output)
        write(p/'manifest.json',record)
        return {k:v for k,v in record.items() if k!='rows'}

    def rows(self,dataset_ids):
        if not isinstance(dataset_ids,list) or not dataset_ids or len(dataset_ids)>100 or len(set(dataset_ids))!=len(dataset_ids): raise ValueError('데이터를 중복 없이 선택하세요.')
        rows=[]; ids={}; pixels={}
        for key in dataset_ids:
            if self.dataset_deleted(key): raise ValueError('삭제 보관함에 있는 데이터입니다. 먼저 복원하세요.')
            p=self.item('datasets',key); manifest=read(p/'manifest.json')
            if manifest['labels']!=LABELS: raise ValueError('데이터 라벨 순서가 다릅니다.')
            for row in manifest['rows']:
                r=dict(row,dataset_id=key)
                if r['id'] in ids or r['pixel_sha256'] in pixels: raise ValueError('선택한 데이터 사이에 동일 ID 또는 이미지가 있습니다. 중복·평가 누출을 먼저 검토하세요.')
                ids[r['id']]=key; pixels[r['pixel_sha256']]=key
                r['image_path']=str(local_only(p/r['image']))
                if not (p/r['image']).is_file(): raise ValueError('등록 이미지가 없어졌습니다.')
                rows.append(r)
        return rows

    def register_model(self,folder,validator=None):
        p=self.item('models',folder)
        if (p/'manifest.json').exists():
            raise ValueError('이미 등록된 모델입니다. 기존 버전은 변경하지 않습니다. 다른 폴더 이름으로 등록하세요.')
        spec=read(local_only(p/'package.json'))
        if spec.get('engine') not in ('dino_head_v1','legacy_dino_rbf_v1','bert5_fusion_v1') or spec.get('labels')!=LABELS:
            raise ValueError('지원하지 않는 모델 형식이거나 22개 라벨 순서가 다릅니다.')
        legacy=spec['engine']=='legacy_dino_rbf_v1'
        fusion=spec['engine']=='bert5_fusion_v1'
        if (legacy or fusion) and validator is None: raise ValueError('앙상블 모델은 실제 로딩 검사 후에만 등록할 수 있습니다.')
        backbone=local_only(p/'backbone')
        for child in p.rglob('*'):
            if child.is_symlink(): raise ValueError('모델 폴더에는 바로가기를 넣을 수 없습니다.')
            local_only(child)
        config=read(backbone/'config.json')
        if config.get('model_type')!='dinov3_vit': raise ValueError('DINOv3 ViT 로컬 모델 폴더가 필요합니다.')
        if not list(backbone.glob('*.safetensors')): raise ValueError('backbone에 safetensors 가중치가 필요합니다.')
        if not (backbone/'preprocessor_config.json').is_file(): raise ValueError('이미지 전처리 설정이 필요합니다.')
        if legacy:
            for filename in LEGACY_WEIGHTS:
                if not (p/'weights'/filename).is_file(): raise ValueError('필수 가중치가 없습니다: '+filename)
        elif fusion:
            checked=inspect_model_folder(p)
            if checked['missing']:raise ValueError('5종 모델 파일 누락: '+', '.join(checked['missing']))
            spec['input_mode']='image_description'
        elif spec.get('has_head') and not (p/'head.safetensors').is_file(): raise ValueError('분류 헤드 가중치가 없습니다.')
        trained=spec.get('trained_pixel_hashes',[])
        if not isinstance(trained,list) or not all(isinstance(x,str) and re.fullmatch('[a-f0-9]{64}',x) for x in trained): raise ValueError('학습 이미지 해시 형식이 잘못됐습니다.')
        spec.update(id=folder,name=str(spec.get('name',folder))[:100],created=now(),
            has_head=bool(spec.get('has_head')),provenance=spec.get('provenance','학습 이력 미확인'),
            training_supported=True,
            package_sha256=sha(p/'package.json'),weight_hashes={x.relative_to(p).as_posix():sha(x) for x in p.rglob('*') if x.is_file() and (x.suffix in ('.safetensors','.pt') or (x.suffix=='.json' and (backbone in x.parents or x.name=='package.json')))})
        if fusion:spec['weight_hashes']=inspect_model_folder(p)['hashes']
        if validator: spec['load_check']=validator(spec)
        write(p/'manifest.json',spec)
        return spec

    def training_plan(self,request):
        rows=self.rows(request.get('dataset_ids')); counts=Counter(r['split'] for r in rows)
        if not counts['train']: raise ValueError('학습용(train) 이미지가 최소 한 장 필요합니다. 개발·평가 이미지를 학습용으로 자동 변경하지 않습니다.')
        self.check_model_available(request.get('model_id'))
        model=read(self.item('models',request.get('model_id'))/'manifest.json')
        if not supports_training(model): raise ValueError('추가 학습을 지원하지 않는 모델입니다.')
        if model.get('engine')=='bert5_fusion_v1' and any(not r.get('description','').strip() for r in rows):
            raise ValueError('5종 BERT 모델은 모든 학습·개발·평가 이미지에 설명문이 필요합니다. labels.csv의 description을 입력하세요. 이미지 전용 모델과 평가 조건이 다릅니다.')
        epochs=int(request.get('epochs',TRAIN_DEFAULTS['epochs'])); batch=int(request.get('batch',TRAIN_DEFAULTS['batch'])); lr=float(request.get('lr',TRAIN_DEFAULTS['lr']))
        if not 1<=epochs<=500 or batch not in (1,2,4,8,16,32) or not math.isfinite(lr) or not 1e-7<=lr<=0.01: raise ValueError('학습 설정 범위를 확인하세요.')
        device=request.get('device','cpu')
        if device not in ('cpu','cuda:0'): raise ValueError('지원하지 않는 장치입니다.')
        seen=set(model.get('trained_pixel_hashes',[]))
        return dict(id=uid('run'),name=str(request.get('name','문양 추가 학습'))[:100],model_id=model['id'],
            dataset_ids=request['dataset_ids'],epochs=epochs,batch=batch,lr=lr,device=device,seed=42,
            engine=model.get('engine','dino_head_v1'),mode='frozen_backbone_head_training',created=now(),counts={s:counts[s] for s in ('train','dev','eval')},rows=rows,
            selection='best_dev_f1' if counts['dev'] else 'last_epoch_no_dev',
            evaluation_available=bool(counts['eval']),
            base_eval_seen=sum(r['pixel_sha256'] in seen for r in rows if r['split']=='eval'),
            base_provenance=model.get('provenance','학습 이력 미확인'),
            benchmark_note='사용자가 지정한 고정 분할. 반복 사용 벤치마크이며 유사 이미지·원천 중복은 별도 검토 필요.')
