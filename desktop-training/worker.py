"""Image-only engines: frozen DINO feature extractors, trainable existing heads."""
import argparse, io, json, math, os, sys, time, traceback, hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from collections import deque
from core import Store, LABELS, LEGACY_WEIGHTS, read, write, uid, now, local_only, sha, metric
from bert5_engine import Bert5Engine

def load_checkpoint(path):
    """Read tensor dictionaries and inert path metadata; never enable pickle execution.

    Linux PosixPath metadata cannot be constructed on Windows. Map only the two
    known pathlib globals to pure, non-filesystem path classes on every platform.
    Unknown globals remain rejected by the weights-only unpickler.
    """
    import torch
    with torch.serialization.safe_globals([(PurePosixPath,'pathlib.PosixPath'),
                                           (PureWindowsPath,'pathlib.WindowsPath')]):
        return torch.load(str(local_only(path)),map_location='cpu',weights_only=True)

def verify_files(package,spec):
    for rel,digest in spec.get('weight_hashes',{}).items():
        path=local_only(package/rel)
        if package not in path.parents or sha(path)!=digest: raise ValueError('등록 후 모델 파일이 변경되었습니다: '+rel)

def head_state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update((key+str(value.dtype)+str(tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()

class Cancelled(Exception): pass

class Progress:
    def __init__(self,folder): self.folder=folder; self.started=time.monotonic(); self.samples=deque(maxlen=60); self.steps=0
    def check(self):
        if (self.folder/'cancel.request').exists(): raise Cancelled()
    def update(self,state,message,progress,**extras):
        self.check(); prior=read(self.folder/'status.json')
        prior.update(status=state,message=message,progress=round(progress,2),elapsed_seconds=round(time.monotonic()-self.started,1),updated=now(),**extras)
        write(self.folder/'status.json',prior)
    def timing(self,duration,left):
        self.steps+=1
        if self.steps>5: self.samples.append(duration)
        return round(sum(self.samples)/len(self.samples)*left) if len(self.samples)>=20 else None

class DinoEngine:
    def __init__(self,store,model_id,device,spec=None):
        import torch
        from transformers import AutoModel, AutoImageProcessor
        from safetensors.torch import load_file
        self.torch=torch; self.device=torch.device(device); self.store=store
        if device.startswith('cuda') and not torch.cuda.is_available(): raise ValueError('NVIDIA GPU를 사용할 수 없습니다. CPU를 선택하거나 CUDA용 환경을 설치하세요.')
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        self.package=store.item('models',model_id); self.spec=spec if spec is not None else read(self.package/'manifest.json')
        if self.spec.get('engine')!='dino_head_v1' or self.spec['labels']!=LABELS: raise ValueError('모델 형식이 다릅니다.')
        current=self.package; visited=set()
        while True:
            if current.name in visited: raise ValueError('모델 참조가 순환합니다.')
            visited.add(current.name); manifest=self.spec if current==self.package else read(current/'manifest.json')
            for rel,digest in manifest.get('weight_hashes',{}).items():
                p=local_only(current/rel)
                if current not in p.parents or sha(p)!=digest: raise ValueError('등록 후 모델 파일이 변경되었습니다: '+rel)
            if 'backbone_model' not in manifest: break
            current=store.item('models',manifest['backbone_model'])
        self.backbone_id=current.name; backbone=local_only(current/'backbone')
        self.processor=AutoImageProcessor.from_pretrained(str(backbone),local_files_only=True,trust_remote_code=False)
        self.backbone=AutoModel.from_pretrained(str(backbone),local_files_only=True,trust_remote_code=False,use_safetensors=True).float().to(self.device).eval()
        if getattr(self.backbone.config,'model_type',None)!='dinov3_vit': raise ValueError('DINOv3 ViT만 지원합니다.')
        for p in self.backbone.parameters(): p.requires_grad_(False)
        dim=self.backbone.config.hidden_size*2
        self.head=torch.nn.Sequential(torch.nn.LayerNorm(dim),torch.nn.Linear(dim,1024),torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.Linear(1024,22)).float().to(self.device)
        if self.spec.get('has_head'): self.head.load_state_dict(load_file(str(local_only(self.package/'head.safetensors'))),strict=True)
        self.head.eval()

    def features(self,images):
        import torch
        batch=self.processor(images=images,return_tensors='pt')['pixel_values'].to(self.device)
        with torch.inference_mode():
            hid=self.backbone(pixel_values=batch).last_hidden_state
            prefix=1+getattr(self.backbone.config,'num_register_tokens',0)
            return torch.cat([hid[:,0],hid[:,prefix:].mean(1)],dim=1).float().cpu()

    def scores(self,features,batch=64):
        import torch
        self.head.eval(); outputs=[]
        with torch.inference_mode():
            for x in features.split(batch): outputs.extend(self.head(x.to(self.device)).sigmoid().cpu().tolist())
        return outputs

class LegacyImageEngine:
    """Original image inference equations, without GT/ID lookup or caption inputs.

    CPU uses FP32; capable CUDA uses original BF16 autocast. Original numerical
    equivalence still requires the complete real assets and 446-image evaluation.
    Models are loaded sequentially to avoid keeping three backbones in memory.
    """
    def __init__(self,store,model_id,device,spec=None):
        import torch
        from transformers import AutoConfig, AutoImageProcessor
        from torchvision import transforms
        self.torch=torch; self.package=store.item('models',model_id)
        self.spec=spec if spec is not None else read(self.package/'manifest.json')
        if self.spec.get('engine')!='legacy_dino_rbf_v1' or self.spec.get('labels')!=LABELS:
            raise ValueError('A/B+RBF 모델 형식 또는 라벨 순서가 다릅니다.')
        verify_files(self.package,self.spec)
        self.base_id=self.spec.get('base_model',model_id)
        self.base_package=store.item('models',self.base_id)
        if self.base_id!=model_id:
            base_spec=read(self.base_package/'manifest.json')
            if base_spec.get('base_model') or base_spec.get('engine')!='legacy_dino_rbf_v1': raise ValueError('원본 A/B 모델 참조가 올바르지 않습니다.')
            verify_files(self.base_package,base_spec)
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('NVIDIA GPU를 사용할 수 없습니다. CPU를 선택하세요.')
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        self.bf16=self.device.type=='cuda' and torch.cuda.is_bf16_supported()
        self.dtype=torch.bfloat16 if self.bf16 else torch.float32
        self.backbone=local_only(self.base_package/'backbone')
        config=AutoConfig.from_pretrained(str(self.backbone),local_files_only=True,trust_remote_code=False)
        if config.model_type!='dinov3_vit': raise ValueError('DINOv3 ViT 백본이 필요합니다.')
        self.dim=config.hidden_size*2
        self.processor=AutoImageProcessor.from_pretrained(str(self.backbone),local_files_only=True,trust_remote_code=False)
        self.transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        r=load_checkpoint(self.base_package/'weights/image_residual.pt')
        if r.get('architecture')!='dinov3_rbf_neural_residual' or r.get('vocab')!=LABELS:
            raise ValueError('RBF 구조 또는 감성어 순서가 다릅니다.')
        self.centers=r['centers'].float(); self.weight=r['state_dict']['weight'].float(); self.gamma=float(r['gamma'])
        if self.centers.ndim!=2 or self.centers.shape[1]!=self.dim or self.weight.shape!=(22,len(self.centers)):
            raise ValueError('백본 특징 크기와 RBF 가중치 크기가 다릅니다.')
        if not math.isfinite(self.gamma) or self.gamma<=0 or not torch.isfinite(self.centers).all() or not torch.isfinite(self.weight).all():
            raise ValueError('RBF에 유효하지 않은 값이 있습니다.')
        self.history=dict(train_records=r.get('train_records'),selected_validation_records=r.get('selected_validation_records'))

    def _context(self):
        from contextlib import nullcontext
        return self.torch.autocast('cuda',dtype=self.torch.bfloat16) if self.bf16 else nullcontext()

    def _backbone(self):
        from transformers import AutoModel
        return AutoModel.from_pretrained(str(self.backbone),local_files_only=True,trust_remote_code=False,
            use_safetensors=True,torch_dtype=self.dtype).to(self.device).eval()

    def _feature(self,model,pixels):
        hidden=model(pixel_values=pixels).last_hidden_state
        prefix=1+int(getattr(model.config,'num_register_tokens',0))
        return self.torch.cat([hidden[:,0],hidden[:,prefix:].mean(1)],dim=1)

    def features(self,images): return self._extract(images,False)

    def training_features(self,images): return self._extract(images,True)

    def _extract(self,images,training):
        torch=self.torch; nn=torch.nn
        from safetensors.torch import load_file
        pixels=torch.stack([self.transform(im.convert('RGB')) for im in images]).to(self.device)
        combined=None; cached=[]; heads={}
        override=load_file(str(self.package/'head.safetensors')) if self.spec.get('base_model') else None
        for name,filename,ratio in zip(('a','b'),LEGACY_WEIGHTS[:2],(.46,.54)):
            backbone=self._backbone()
            head=nn.Sequential(nn.LayerNorm(self.dim),nn.Linear(self.dim,1024),nn.GELU(),nn.Dropout(.2),nn.Linear(1024,22)).float().to(self.device)
            model=nn.Module(); model.add_module('backbone',backbone); model.add_module('head',head)
            checkpoint=load_checkpoint(self.base_package/'weights'/filename)
            if checkpoint.get('vocab',LABELS)!=LABELS: raise ValueError('DINO 분류기의 감성어 순서가 다릅니다.')
            keys=set(model.state_dict()); compatible={}
            for key,value in checkpoint['state_dict'].items():
                candidate=key
                for before,after in [('backbone.layer.','backbone.model.layer.'),('backbone.model.layer.','backbone.layer.')]:
                    remapped=key.replace(before,after,1)
                    if key.startswith(before) and remapped in keys: candidate=remapped
                if candidate in compatible: raise ValueError('중복 가중치 키가 있습니다.')
                compatible[candidate]=value
            model.load_state_dict(compatible,strict=True); model.eval()
            if override is not None:
                head.load_state_dict({k[len(name)+1:]:v for k,v in override.items() if k.startswith(name+'.')},strict=True)
            del checkpoint,compatible
            with torch.inference_mode():
                views=[]; feature_views=[]
                for view in (pixels,pixels.flip(-1),pixels.flip(-2),pixels.flip((-1,-2))):
                    with self._context():
                        f=self._feature(backbone,view).float()
                        views.append(head(f))
                    if training: feature_views.append(f.cpu())
                scores=torch.stack(views).mean(0).sigmoid().float().cpu()
                combined=ratio*scores if combined is None else combined+ratio*scores
                if training: cached.append(torch.stack(feature_views,dim=1).reshape(len(images),-1))
            if training: heads[name]=head
            del model,backbone,head,views
            if self.device.type=='cuda': torch.cuda.empty_cache()
        backbone=self._backbone()
        batch=self.processor(images=images,return_tensors='pt')['pixel_values'].to(self.device)
        with torch.inference_mode():
            with self._context(): features=self._feature(backbone,batch).float().cpu()
            features=torch.nn.functional.normalize(features,dim=1)
            kernel=torch.exp(-self.gamma*(1-features@self.centers.T).clamp(min=0))
            residual=torch.nn.functional.linear(kernel,self.weight)
            combined=combined+residual
        del backbone
        if self.device.type=='cuda': torch.cuda.empty_cache()
        if not torch.isfinite(combined).all(): raise ValueError('모델 예측에 유효하지 않은 값이 있습니다.')
        if training:
            self.head=nn.ModuleDict(heads).float().to(self.device)
            return torch.cat(cached+[residual],dim=1).clone()
        return combined

    def training_logits(self,features):
        features=features.to(self.device); width=4*self.dim
        with self._context():
            # Match the original inference path: one head call per view. Flattening
            # all four views into one GEMM changes BF16 rounding on CUDA.
            outputs=[]
            for i,name in enumerate(('a','b')):
                views=features[:,i*width:(i+1)*width].reshape(len(features),4,self.dim)
                outputs.append(self.torch.stack([self.head[name](views[:,v].contiguous()) for v in range(4)]).mean(0))
            return outputs

    def training_loss(self,features,target):
        logits=self.training_logits(features)
        return sum(r*self.torch.nn.functional.binary_cross_entropy_with_logits(x.float(),target) for r,x in zip((.46,.54),logits))

    def scores(self,features,batch=64):
        if features.shape[1]==22: return features.tolist()
        self.head.eval(); outputs=[]
        with self.torch.no_grad():
            for x in features.split(batch):
                logits=self.training_logits(x)
                values=.46*logits[0].sigmoid().float()+.54*logits[1].sigmoid().float()+x[:,-22:].to(self.device)
                outputs.extend(values.cpu().tolist())
        return outputs

def make_engine(store,model_id,device,spec=None):
    if spec is None:store.check_model_available(model_id)
    manifest=spec if spec is not None else read(store.item('models',model_id)/'manifest.json')
    cls=Bert5Engine if manifest.get('engine')=='bert5_fusion_v1' else LegacyImageEngine if manifest.get('engine')=='legacy_dino_rbf_v1' else DinoEngine
    return cls(store,model_id,device,spec)

def probe_model(root,model_id,spec):
    from PIL import Image
    store=Store(root); engine=make_engine(store,model_id,'cpu',spec)
    images=[Image.new('RGB',(256,256),(120,120,120))]
    features=engine.features(images,['회색 바탕의 문양']) if isinstance(engine,Bert5Engine) else engine.features(images)
    if spec.get('has_head'):
        scores=engine.scores(features)
        if len(scores)!=1 or len(scores[0])!=22 or not all(math.isfinite(x) for x in scores[0]):
            raise ValueError('실제 로딩 후 22개 점수를 생성하지 못했습니다.')
    verify_files(store.item('models',model_id),spec)
    report=dict(status='passed',created=now(),device='cpu',synthetic_image_only=not isinstance(engine,Bert5Engine),description_input=isinstance(engine,Bert5Engine),
                research_performance_verified=False,engine=spec['engine'])
    if hasattr(engine,'history'): report['checkpoint_training_history']=engine.history
    write(store.item('models',model_id)/'registration_check.json',report)
    return report

def execute(root,run_id):
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import save_file, load_file
    store=Store(root); folder=store.item('runs',run_id); cfg=read(folder/'config.json'); progress=Progress(folder)
    torch.manual_seed(cfg.get('seed',42))
    progress.update('loading','로컬 모델을 확인하고 불러오는 중',0)
    spec=read(store.item('models',cfg['model_id'])/'manifest.json')
    if cfg['device'].startswith('cuda') and torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(torch.device(cfg['device']))
    engine=make_engine(store,cfg['model_id'],cfg['device'])
    def runtime():
        cuda=engine.device.type=='cuda'
        return dict(device=str(engine.device),torch_version=torch.__version__,cuda_version=torch.version.cuda,
                    gpu_name=torch.cuda.get_device_name(engine.device) if cuda else None,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device) if cuda else 0,
                    elapsed_seconds=round(time.monotonic()-progress.started,3))
    if cfg.get('kind')=='infer':
        progress.update('extracting','이미지 특징을 추출하는 중',40)
        with Image.open(folder/'input.image') as im:
            image=im.convert('RGB') if isinstance(engine,(LegacyImageEngine,Bert5Engine)) else ImageOps.exif_transpose(im).convert('RGB')
        features=engine.features([image],[cfg.get('description','')]) if isinstance(engine,Bert5Engine) else engine.features([image]); scores=engine.scores(features)[0]
        if not all(math.isfinite(x) for x in scores): raise ValueError('예측에 유효하지 않은 수치가 있습니다.')
        top=sorted(range(22),key=lambda j:(-scores[j],j))[:5]
        write(folder/'result.json',dict(kind='infer',id=run_id,model_id=cfg['model_id'],top5=[dict(label=LABELS[j],score=scores[j]) for j in top],scores=scores,GT_input=False,description_input=isinstance(engine,Bert5Engine),runtime=runtime()))
        progress.update('complete','추론 완료',100,eta_seconds=0); return
    rows=cfg['rows']; pieces=[]; batch=cfg['batch']
    for start in range(0,len(rows),batch):
        progress.check(); images=[]
        for r in rows[start:start+batch]:
            path=local_only(r['image_path'])
            if store.root/'datasets' not in path.parents or sha(path)!=r['file_sha256']: raise ValueError('등록 이후 이미지가 변경되었습니다.')
            with Image.open(path) as im: images.append(im.convert('RGB') if isinstance(engine,(LegacyImageEngine,Bert5Engine)) else ImageOps.exif_transpose(im).convert('RGB'))
        pieces.append(engine.features(images,[r.get('description','') for r in rows[start:start+batch]]) if isinstance(engine,Bert5Engine) else engine.training_features(images) if isinstance(engine,LegacyImageEngine) else engine.features(images))
        progress.update('extracting','이미지·설명문 특징 추출 · GT는 인코더에 입력하지 않습니다.' if isinstance(engine,Bert5Engine) else '이미지 특징 추출 · GT와 설명문은 인코더에 입력하지 않습니다.',40*min(start+batch,len(rows))/len(rows),processed=min(start+batch,len(rows)),total=len(rows),eta_seconds=None)
    features=torch.cat(pieces); del pieces
    indexes={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ('train','dev','eval')}
    train=features[indexes['train']]; dev=features[indexes['dev']]
    def target(indices): return torch.tensor([[float(label in rows[i]['labels']) for label in LABELS] for i in indices],dtype=torch.float32)
    y=target(indexes['train']); dev_rows=[rows[i] for i in indexes['dev']]
    baseline=engine.scores(features) if engine.spec.get('has_head') else None
    initial_state={k:v.detach().cpu().clone() for k,v in engine.head.state_dict().items()}
    write(folder/'head_initialization.json',dict(parent_model=cfg['model_id'],state_sha256=head_state_hash(initial_state),created=now()))
    optimizer=torch.optim.AdamW(engine.head.parameters(),lr=cfg['lr'],weight_decay=.01)
    rng=torch.Generator().manual_seed(cfg['seed']); best=-1.; selected=0; history=[]
    steps_per=math.ceil(len(train)/batch); total_steps=steps_per*cfg['epochs']; step=0
    for epoch in range(1,cfg['epochs']+1):
        engine.head.train(); loss_sum=0
        for ix in torch.randperm(len(train),generator=rng).split(batch):
            progress.check(); begin=time.monotonic()
            if isinstance(engine,(LegacyImageEngine,Bert5Engine)): loss=engine.training_loss(train[ix],y[ix].to(engine.device))
            else:
                logits=engine.head(train[ix].to(engine.device))
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[ix].to(engine.device))
            if not torch.isfinite(loss): raise ValueError('학습 손실이 유효하지 않습니다.')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            if engine.device.type=='cuda': torch.cuda.synchronize()
            loss_sum+=loss.item()*len(ix); step+=1
            eta=progress.timing(time.monotonic()-begin,total_steps-step)
            progress.update('training',f'분류 헤드 학습 · {epoch}/{cfg["epochs"]} 에폭',40+50*step/total_steps,epoch=epoch,epochs=cfg['epochs'],step=step,total_steps=total_steps,eta_seconds=eta,eta_scope='남은 학습 배치만 추정. 개발 평가·최종 평가·저장은 별도.')
        dev_metric,_=metric(engine.scores(dev) if len(dev) else [],dev_rows)
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(train),dev_f1_at_5=dev_metric['f1_at_5']))
        if not len(dev) or dev_metric['f1_at_5']>best:
            best=dev_metric['f1_at_5'] if len(dev) else -1.; selected=epoch
            save_file({k:v.detach().cpu().contiguous() for k,v in engine.head.state_dict().items()},str(folder/'best_head.safetensors'))
        write(folder/'history.json',history)
    progress.update('evaluating','저장 가중치를 고정한 뒤 결과를 확인하는 중',92,eta_seconds=None)
    engine.head.load_state_dict(load_file(str(folder/'best_head.safetensors')))
    selection='최고 dev F1@5, 동점이면 빠른 에폭' if len(dev) else '개발 자료 없음: 미리 지정한 마지막 에폭 저장. 성능 개선 판단 불가.'
    write(folder/'frozen_before_evaluation.json',dict(selected_epoch=selected,head_sha256=sha(folder/'best_head.safetensors'),selection=selection,created=now()))
    scores=engine.scores(features); metrics={}; all_predictions=[]
    for split,ix in indexes.items():
        subset=[rows[i] for i in ix]; result,preds=metric([scores[i] for i in ix],subset)
        if baseline is not None: result['baseline']=metric([baseline[i] for i in ix],subset)[0]
        metrics[split]=result; all_predictions.extend(preds)
    with (folder/'predictions.jsonl').open('w',encoding='utf-8') as f:
        for p in all_predictions: f.write(json.dumps(p,ensure_ascii=False)+'\n')
    # Recount from the on-disk artifact, separately from the metric function.
    saved=[json.loads(line) for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    for split,ix in indexes.items():
        subset=[p for p in saved if p['split']==split]
        slots=sum(sum(label in p['GT'] for label in p['top5']) for p in subset)
        if len(subset)!=len(ix) or slots!=metrics[split]['correct_slots']:
            raise ValueError('저장된 예측의 독립 지표 재계산이 일치하지 않습니다.')
        for p in subset:
            expected=[LABELS[j] for j in sorted(range(22),key=lambda j:(-p['scores'][j],j))[:5]]
            if expected!=p['top5']: raise ValueError('저장 점수와 Top-5가 일치하지 않습니다.')
    # Reload the exact saved head, then independently recompute Top-5 metrics.
    engine.head.load_state_dict(load_file(str(folder/'best_head.safetensors')))
    fresh=engine.scores(features)
    for split,ix in indexes.items():
        fresh_result,fresh_predictions=metric([fresh[i] for i in ix],[rows[i] for i in ix])
        expected=[p['top5'] for p in all_predictions if p['split']==split]
        if [p['top5'] for p in fresh_predictions]!=expected: raise ValueError('가중치 재로딩 후 Top-5 재현 검증이 실패했습니다.')
        if fresh_result['correct_slots']!=metrics[split]['correct_slots']: raise ValueError('지표 재계산이 일치하지 않습니다.')
    progress.update('saving','평가 결과와 새 가중치를 보관하는 중',98)
    key=uid('model'); destination=store.item('models',key); destination.mkdir()
    save_file({k:v.detach().cpu().contiguous() for k,v in engine.head.state_dict().items()},str(destination/'head.safetensors'))
    trained=sorted(set(engine.spec.get('trained_pixel_hashes',[]))|{r['pixel_sha256'] for r in rows if r['split']=='train'})
    legacy=isinstance(engine,(LegacyImageEngine,Bert5Engine));fusion=isinstance(engine,Bert5Engine)
    spec=dict(id=key,name=cfg['name'],engine=engine.spec['engine'],labels=LABELS,has_head=True,training_supported=True,created=now(),
        parent_model=cfg['model_id'],run_id=run_id,trained_pixel_hashes=trained,
        provenance='새 train만 추가 학습; 초기 모델 이력: '+cfg['base_provenance'],
        weight_hashes={'head.safetensors':sha(destination/'head.safetensors')},selected_epoch=selected,
        evaluation_f1=metrics['eval']['f1_at_5'],base_eval_seen=cfg['base_eval_seen'],selection=selection)
    spec.update({'base_model':engine.base_id} if legacy else {'backbone_model':engine.backbone_id})
    if fusion:spec['input_mode']='image_description'
    write(destination/'manifest.json',spec)
    write(folder/'result.json',dict(id=run_id,kind='train',model_id=key,selected_epoch=selected,metrics=metrics,runtime=runtime(),
        checks=dict(fresh_load_top5_equal=True,independent_metric=True,GT_input_to_encoder=False,description_input=fusion,
            head_weights_changed=any(not torch.equal(initial_state[k],v.cpu()) for k,v in engine.head.state_dict().items())),
        base_eval_seen=cfg['base_eval_seen'],base_provenance=cfg['base_provenance'],benchmark_note=cfg['benchmark_note'],
        selection=selection,evaluation_available=bool(indexes['eval']),
        training_mode='기존 5종 융합부 추가 학습 · DINO/텍스트 인코더/앙상블 비율 고정' if fusion else '기존 A/B 분류 헤드 추가 학습 · DINO 특징 추출부/RBF/앙상블 비율 고정' if legacy else '동결 DINOv3 + FP32 분류 헤드 학습'))
    progress.update('complete','학습·평가 완료. 적용 모델은 자동으로 바뀌지 않습니다.',100,eta_seconds=0,model_id=key)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
    group=p.add_mutually_exclusive_group(required=True); group.add_argument('--run'); group.add_argument('--probe-model')
    args=p.parse_args()
    if args.probe_model:
        probe_model(args.root,args.probe_model,json.load(sys.stdin)); return
    folder=Store(args.root).item('runs',args.run)
    try: execute(args.root,args.run)
    except Cancelled:
        s=read(folder/'status.json'); s.update(status='cancelled',message='중단했습니다. 입력·로그·저장된 가중치는 보존했습니다.'); write(folder/'status.json',s)
    except Exception as e:
        traceback.print_exc(); s=read(folder/'status.json'); s.update(status='failed',message=str(e)[:1000]); write(folder/'status.json',s); sys.exit(1)

if __name__=='__main__': main()
