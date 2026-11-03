"""Shared local application service. No HTTP, browser, or Qt dependency."""
import importlib.util
import csv
import io
import json
import hashlib
import os
import shutil
from pathlib import Path, PurePosixPath
import subprocess
import sys
import threading
import zipfile
import stat
from datetime import datetime, timezone, timedelta

from core import Store, LABELS, local_only, read, write, uid, now, pixel_hash, inspect_model_folder, sha

ROOT = Path(__file__).absolute().parent
RUNNING = {'queued', 'loading', 'extracting', 'training', 'evaluating', 'saving'}


class WorkspaceLock:
    """Uses the same lock as the previous browser application."""
    def __init__(self, root):
        root = local_only(root)
        root.mkdir(parents=True, exist_ok=True)
        self.file = local_only(root / 'application.lock').open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                if not self.file.read(1):
                    self.file.write(b'0'); self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError('같은 저장 폴더의 프로그램이 이미 실행 중입니다. 기존 창을 닫아 주세요.')

    def close(self):
        if not self.file.closed:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            self.file.close()


class App:
    def __init__(self, root=ROOT / 'workspace'):
        self.store = Store(root)
        self.process = None
        self.lock = threading.Lock()
        for status in self.store.list('runs'):
            if status.get('status') in RUNNING:
                status.update(status='interrupted', message='이전 실행이 중단되었습니다. 기존 기록은 보존했습니다.')
                write(self.store.item('runs', status['id']) / 'status.json', status)
        self.initialize_bundled_default()

    def initialize_bundled_default(self):
        """Select a verified bundled model once; never replace a user's selection."""
        if local_only(self.store.root/'active.json').exists(): return
        settings=local_only(self.store.root.parent/'default_model.json')
        if not settings.is_file(): return
        default=read(settings); folder=self.store.item('models',default['model_id'])
        if self.store.model_deleted(default['model_id']): return
        manifest=local_only(folder/'manifest.json')
        if not manifest.is_file() or sha(manifest)!=default.get('manifest_sha256'):
            raise ValueError('기본 가중치 파일이 누락되거나 변경됐습니다. 전체 ZIP을 새 로컬 폴더에 다시 풀어 주세요.')
        spec=read(manifest)
        if spec.get('load_check',{}).get('status')!='passed' or not spec.get('has_head'):
            raise ValueError('기본 모델의 로딩 검증 기록이 없습니다.')
        for rel in spec.get('weight_hashes',{}):
            path=local_only(folder/rel)
            if folder not in path.parents or not path.is_file():
                raise ValueError('기본 모델 파일이 부족합니다: '+rel)
        # Full file hashes are verified again by the worker on every actual use.
        self.activate(default['model_id'])

    def snapshot(self):
        libs = {x: importlib.util.find_spec(x) is not None for x in ('torch', 'torchvision', 'transformers', 'safetensors', 'numpy')}
        active = self.store.root / 'active.json'
        models = [{k: v for k, v in m.items() if k not in ('trained_pixel_hashes', 'weight_hashes')}
                  for m in self.store.list('models')]
        deleted=[d for d in self.store.list('datasets',include_deleted=True) if self.store.dataset_deleted(d['id'])]
        deleted_models=[]
        for m in self.store.list('models',include_deleted=True):
            if self.store.model_deleted(m['id']):
                item={k:v for k,v in m.items() if k not in ('trained_pixel_hashes','weight_hashes')}
                record=read(local_only(self.store.root/'model_trash'/(m['id']+'.json')))
                item['trash']={k:record.get(k) for k in ('status','purge_after','error')}
                deleted_models.append(item)
        return dict(datasets=self.store.list('datasets'), deleted_datasets=deleted, models=models, deleted_models=deleted_models, runs=self.store.list('runs'),
                    active=read(active) if active.exists() else None, labels=LABELS,
                    dependencies=libs, storage=str(self.store.root), retention=self.model_retention(), version='0.15')

    def model_retention(self):
        path=local_only(self.store.root/'model_retention.json')
        result=read(path) if path.is_file() else dict(enabled=False,days=7)
        if type(result.get('enabled')) is not bool or type(result.get('days')) is not int or not 1<=result['days']<=365:
            raise ValueError('가중치 보관 기간 설정이 잘못됐습니다.')
        return result

    def _require_model_idle(self):
        if self.is_running() or any(r.get('status') in RUNNING for r in self.store.list('runs')):
            raise ValueError('학습·추론이 실행 중입니다. 작업 완료 또는 안전 중단 후 처리하세요.')

    def set_model_retention(self,enabled,days):
        if type(enabled) is not bool or type(days) is not int or not 1<=days<=365:
            raise ValueError('보관 기간은 1~365일로 설정하세요.')
        with self.lock:
            self._require_model_idle()
            current=datetime.now(timezone.utc)
            policy=dict(enabled=enabled,days=days,updated=current.isoformat())
            # Existing archived weights receive a fresh grace period, never a backdated deadline.
            for model in self.store.list('models',include_deleted=True):
                marker=local_only(self.store.root/'model_trash'/(model['id']+'.json'))
                if not marker.is_file():continue
                record=read(marker)
                if record.get('status')!='deleted':continue
                record['purge_after']=(current+timedelta(days=days)).isoformat() if enabled else None
                write(marker,record)
            write(local_only(self.store.root/'model_retention.json'),policy)
            return policy

    def _purge_model_locked(self,model_id):
        self._require_model_idle()
        folder=self.store.item('models',model_id)
        marker=local_only(self.store.root/'model_trash'/(model_id+'.json'))
        record=read(marker) if marker.is_file() else {}
        if record.get('status') not in ('deleted','purging'):
            raise ValueError('삭제 보관함의 가중치만 영구 삭제할 수 있습니다.')
        active=local_only(self.store.root/'active.json')
        if active.is_file() and read(active).get('model_id')==model_id:
            raise ValueError('현재 적용 중인 모델은 영구 삭제할 수 없습니다.')
        # Archived children also require their original backbone for restoration.
        dependent=[m['id'] for m in self.store.list('models',include_deleted=True)
                   if m['id']!=model_id and model_id in (m.get('base_model'),m.get('backbone_model'))]
        if dependent:raise ValueError('다른 모델이 참조합니다. 삭제 보관함의 참조 모델도 먼저 영구 삭제해야 합니다: '+', '.join(dependent[:5]))
        model=read(local_only(folder/'manifest.json'))
        if model.get('id')!=model_id:raise ValueError('모델 ID가 일치하지 않습니다.')
        files=[];directories=[]
        def inventory(directory):
            for entry in directory.iterdir():
                path=local_only(entry)  # Refuse links/junctions before descending, including NAS aliases.
                info=path.lstat()
                if stat.S_ISDIR(info.st_mode):
                    inventory(path);directories.append(path)
                elif stat.S_ISREG(info.st_mode):
                    if path!=folder/'manifest.json':files.append((path,info.st_dev,info.st_ino))
                else:raise ValueError('일반 파일이 아닌 항목이 있어 삭제하지 않았습니다: '+str(path))
        inventory(folder)
        # The worker keeps one duplicate checkpoint in its originating run; preserve all metrics/logs.
        if model.get('run_id'):
            run=self.store.item('runs',model['run_id']);result=local_only(run/'result.json')
            checkpoint=local_only(run/'best_head.safetensors')
            if checkpoint.exists():
                if not result.is_file() or read(result).get('model_id')!=model_id:
                    raise ValueError('학습 임시 가중치의 소유 모델을 확인할 수 없습니다.')
                info=checkpoint.lstat()
                if not stat.S_ISREG(info.st_mode):raise ValueError('학습 임시 가중치가 일반 파일이 아닙니다.')
                files.append((checkpoint,info.st_dev,info.st_ino))
        record.update(status='purging',manifest=model)
        record.setdefault('removed_files',[]);record.setdefault('removed_bytes',0)
        write(marker,record)  # Tombstone is durable before any file is removed.
        try:
            for path,device,inode in files:
                path=local_only(path);info=path.lstat()
                if (info.st_dev,info.st_ino)!=(device,inode) or not stat.S_ISREG(info.st_mode):
                    raise ValueError('삭제 대상이 변경되어 중단했습니다: '+str(path))
                path.unlink()
                record['removed_files'].append(str(path.relative_to(self.store.root)))
                record['removed_bytes']+=info.st_size
                write(marker,record)
            for directory in directories:
                local_only(directory).rmdir()
            record.update(status='purged',purged_at=now(),error=None)
            record.setdefault('events',[]).append(dict(action='purged',time=record['purged_at']))
            write(marker,record)
        except Exception as exc:
            record['error']=str(exc);write(marker,record)
            raise ValueError('일부 파일 삭제 후 중단되었습니다. 복원할 수 없으며 영구 삭제를 다시 시도하세요: '+str(exc)) from exc
        return dict(id=model_id,name=model.get('name',model_id),removed_bytes=record['removed_bytes'])

    def purge_model(self,model_id):
        with self.lock:return self._purge_model_locked(model_id)

    def cleanup_expired_models(self):
        with self.lock:
            policy=self.model_retention()
            report=dict(deleted=[],blocked=[],deferred=False)
            if not policy['enabled']:return report
            if self.is_running() or any(r.get('status') in RUNNING for r in self.store.list('runs')):
                report['deferred']=True;return report
            pending={}
            current=datetime.now(timezone.utc)
            for model in self.store.list('models',include_deleted=True):
                marker=local_only(self.store.root/'model_trash'/(model['id']+'.json'))
                if not marker.is_file():continue
                record=read(marker)
                if record.get('status') not in ('deleted','purging') or not record.get('purge_after'):continue
                try:
                    if datetime.fromisoformat(record['purge_after'])<=current:pending[model['id']]=None
                except (TypeError,ValueError):report['blocked'].append(dict(id=model['id'],reason='삭제 예정일 오류'))
            # Remove children before parents; don't cascade-delete an unexpired child.
            while pending:
                progress=False
                for key in list(pending):
                    try:
                        report['deleted'].append(self._purge_model_locked(key));del pending[key];progress=True
                    except (OSError,ValueError) as exc:pending[key]=str(exc)
                if not progress:break
            report['blocked'] += [dict(id=k,reason=v) for k,v in pending.items()]
            return report

    def set_model_deleted(self,model_id,deleted):
        with self.lock:
            if self.is_running() or any(r.get('status') in RUNNING for r in self.store.list('runs')):
                raise ValueError('학습·추론이 실행 중입니다. 작업 완료 또는 안전 중단 후 삭제·복원하세요.')
            folder=self.store.item('models',model_id);model=read(local_only(folder/'manifest.json'))
            if model['id']!=model_id: raise ValueError('모델 ID가 일치하지 않습니다.')
            marker=local_only(self.store.root/'model_trash'/(model_id+'.json'))
            if marker.is_file() and read(marker).get('status') in ('purging','purged'):
                raise ValueError('영구 삭제를 시작한 가중치는 복원할 수 없습니다.')
            if self.store.model_deleted(model_id)==deleted:
                raise ValueError('이미 삭제된 모델입니다.' if deleted else '이미 사용 중인 모델입니다.')
            if deleted:
                active=local_only(self.store.root/'active.json')
                if active.is_file() and read(active).get('model_id')==model_id:
                    raise ValueError('현재 적용 중인 모델은 삭제할 수 없습니다. 먼저 다른 모델을 적용하세요.')
                dependent=[m.get('name',m['id']) for m in self.store.list('models') if m['id']!=model_id and model_id in (m.get('base_model'),m.get('backbone_model'))]
                if dependent: raise ValueError('다른 모델이 실행에 참조하는 원본 가중치입니다. 참조 모델부터 삭제 보관함으로 보내세요: '+', '.join(dependent[:5]))
            else:
                for field in ('base_model','backbone_model'):
                    if model.get(field):self.store.check_model_available(model[field])
            marker=local_only(self.store.root/'model_trash'/(model_id+'.json'))
            record=read(marker) if marker.is_file() else dict(id=model_id,events=[])
            record['status']='deleted' if deleted else 'restored'
            record['events'].append(dict(action=record['status'],time=now()))
            policy=self.model_retention()
            record['purge_after']=(datetime.now(timezone.utc)+timedelta(days=policy['days'])).isoformat() if deleted and policy['enabled'] else None
            write(marker,record)
            return dict(id=model_id,name=model.get('name',model_id),status=record['status'])

    def set_dataset_deleted(self,dataset_id,deleted):
        """Recoverable list deletion: original paths remain for historical runs."""
        with self.lock:
            if self.is_running() or any(r.get('status') in RUNNING for r in self.store.list('runs')):
                raise ValueError('학습·추론이 실행 중입니다. 작업 완료 또는 안전 중단 후 삭제·복원하세요.')
            folder=self.store.item('datasets',dataset_id)
            manifest=read(local_only(folder/'manifest.json'))
            if manifest['id']!=dataset_id: raise ValueError('등록 데이터 ID가 일치하지 않습니다.')
            if self.store.dataset_deleted(dataset_id)==deleted:
                raise ValueError('이미 삭제된 데이터입니다.' if deleted else '이미 사용 중인 데이터입니다.')
            marker=local_only(self.store.root/'dataset_trash'/(dataset_id+'.json'))
            record=read(marker) if marker.is_file() else dict(id=dataset_id,events=[])
            record['status']='deleted' if deleted else 'restored'
            record['events'].append(dict(action=record['status'],time=now()))
            write(marker,record)
            return dict(id=dataset_id,name=manifest['name'],count=manifest['count'],status=record['status'])

    def inspect_model(self, path):
        return inspect_model_folder(path)

    def validate_model(self, model_id, spec, store=None):
        store = store or self.store
        executable=Path(sys.executable)
        if executable.name.lower()=='pythonw.exe': executable=executable.with_name('python.exe')
        env=os.environ.copy()
        env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUTF8='1',
                   HF_HOME=str(store.root/'cache/huggingface'),TORCH_HOME=str(store.root/'cache/torch'),
                   TMPDIR=str(store.root/'cache'),TEMP=str(store.root/'cache'),TMP=str(store.root/'cache'))
        p=store.item('models',model_id)
        with (p/'registration_check.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([str(executable),str(ROOT/'worker.py'),'--root',str(store.root),'--probe-model',model_id],
                input=json.dumps(spec),text=True,encoding='utf-8',stdout=log,stderr=subprocess.STDOUT,
                cwd=str(ROOT),env=env,timeout=1800,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if result.returncode:
            raise ValueError('가중치 로딩 검사가 실패했습니다. 등록하지 않았습니다. '+str(p/'registration_check.log'))
        report=read(p/'registration_check.json')
        if report.get('status')!='passed': raise ValueError('모델 로딩 검증을 통과하지 못했습니다.')
        return report

    def export_model(self, model_id, destination, overwrite=False):
        from model_archive import export_model
        with self.lock:
            if self.is_running(): raise ValueError('학습·추론이 끝난 후 가중치를 저장하세요.')
            return export_model(self.store, model_id, destination, overwrite)

    def import_model_archive(self, path):
        from model_archive import import_model
        with self.lock:
            if self.is_running(): raise ValueError('학습·추론이 끝난 후 가중치를 불러오세요.')
            if not all(self.snapshot()['dependencies'].values()):
                raise ValueError('가중치 로딩 검사를 위해 먼저 실행 환경을 설치하세요.')
            return import_model(self.store, path, self.validate_model)

    def import_model(self, checked):
        """Import an allowlist, preserve source and never expose half-copied models."""
        with self.lock:
            if self.is_running(): raise ValueError('실행 중인 학습·추론을 마친 후 등록하세요.')
            fresh=self.inspect_model(checked['source'])
            if fresh['missing']: raise ValueError('필수 파일이 부족합니다: '+', '.join(fresh['missing']))
            if fresh!=checked: raise ValueError('파일 확인 후 폴더가 변경되었습니다. 다시 확인해 주세요.')
            if not all(self.snapshot()['dependencies'].values()):
                raise ValueError('실제 모델 확인을 위해 install-engine-windows.bat를 먼저 실행하세요.')
            if shutil.disk_usage(self.store.root).free < fresh['bytes']+512*1024**2:
                raise ValueError('모델 복사와 확인에 필요한 로컬 저장 공간이 부족합니다.')
            key=uid('imported'); dest=self.store.item('models',key); dest.mkdir()
            write(dest/'import_status.json',dict(status='copying',source=fresh['source'],created=now()))
            try:
                for rel,src in fresh['files'].items():
                    target=local_only(dest/rel); target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(local_only(src),target)
                    if sha(target)!=fresh['hashes'][rel]: raise ValueError('복사 중 원본 가중치가 변경되었습니다: '+rel)
                spec=dict(engine=fresh['engine'],name=fresh['name'],labels=LABELS,has_head=fresh['has_head'],
                          provenance=fresh['provenance'],trained_pixel_hashes=fresh['trained_pixel_hashes'],
                          import_source_hashes=fresh['hashes'])
                write(dest/'package.json',spec)
                result=self.store.register_model(key,lambda manifest:self.validate_model(key,manifest))
                write(dest/'import_status.json',dict(status='complete',created=now()))
                return result
            except Exception as e:
                write(dest/'import_status.json',dict(status='failed',message=str(e),created=now()))
                raise

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def import_data(self, path, name, labels=None, split='train'):
        """Normalize local folder/CSV/single image to the existing validated import."""
        path = local_only(path)
        if path.is_file() and path.suffix.lower()=='.zip':
            if path.stat().st_size > 256 * 1024**2:
                raise ValueError('ZIP은 256MB 이하로 나누어 등록하세요.')
            return self.store.import_zip(path.read_bytes(), name)
        image_mode=path.is_file() and path.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}
        if image_mode:
            if not labels or len(labels)!=5 or len(set(labels))!=5 or not set(labels)<=set(LABELS):
                raise ValueError('이미지를 등록하려면 실제 정답 감성어를 서로 다르게 5개 선택하세요. 정답 없이 사용하려면 추론 화면을 이용하세요.')
            if split not in ('train','dev','eval'): raise ValueError('학습·검증·평가(테스트) 중 용도를 선택하세요.')
            rows=[dict(id=path.stem,image='images/'+path.name,labels=';'.join(labels),split=split,description='')]
            text=io.StringIO(newline=''); writer=csv.DictWriter(text,fieldnames=['id','image','labels','split','description'])
            writer.writeheader(); writer.writerows(rows); csv_bytes=text.getvalue().encode('utf-8-sig')
            inputs={rows[0]['image']:path}
        else:
            csv_path=local_only(path/'labels.csv') if path.is_dir() else path
            if csv_path.suffix.lower()!='.csv' or not csv_path.is_file():
                raise ValueError('labels.csv가 있는 폴더, 정답 CSV, ZIP 또는 이미지 파일을 선택하세요.')
            if csv_path.stat().st_size>8*1024**2: raise ValueError('정답 CSV는 8MB 이하로 나누어 등록하세요.')
            csv_bytes=csv_path.read_bytes()
            reader=csv.DictReader(io.StringIO(csv_bytes.decode('utf-8-sig')))
            if not {'id','image','labels','split'}<=set(reader.fieldnames or []):
                raise ValueError('정답 CSV에 id, image, labels, split 열이 필요합니다.')
            rows=[]
            for row in reader:
                rows.append(row)
                if len(rows)>10000: raise ValueError('데이터는 10,000개 행 이하로 나누어 등록하세요.')
            inputs={}
            for row in rows:
                rel=(row.get('image') or '').strip(); relative=PurePosixPath(rel)
                if not rel or relative.is_absolute() or '..' in relative.parts or '\\' in rel or ':' in rel:
                    raise ValueError('CSV 이미지 경로는 같은 폴더 아래의 상대 경로여야 합니다.')
                image_path=local_only(csv_path.parent/rel)
                if csv_path.parent not in image_path.parents or image_path.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:
                    raise ValueError('CSV 이미지 경로 또는 이미지 확장자를 확인하세요: '+rel)
                inputs[rel]=image_path
        # Read only the files named in the CSV, never recursively scan a directory.
        total=len(csv_bytes)
        for rel,image_path in inputs.items():
            if not image_path.is_file(): raise ValueError('이미지 파일을 찾을 수 없습니다: '+rel)
            size=image_path.stat().st_size
            if size>32*1024**2: raise ValueError('개별 이미지는 32MB 이하로 등록하세요.')
            total+=size
            if total>2*1024**3: raise ValueError('전체 데이터는 2GB 이하로 나누어 등록하세요.')
        blob=io.BytesIO()
        with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('labels.csv',csv_bytes)
            archive.writestr('import_source.json',json.dumps(dict(source=str(path),kind='image' if image_mode else 'csv_or_folder'),ensure_ascii=False))
            for rel,image_path in inputs.items():
                # Recheck local path just before reading; don't follow symlinks/junctions.
                archive.write(local_only(image_path),rel)
                if blob.tell()>256*1024**2: raise ValueError('등록 데이터 압축 크기가 256MB를 넘습니다. 나누어 등록하세요.')
        return self.store.import_zip(blob.getvalue(),name)

    @staticmethod
    def _inference_bytes(path):
        path = local_only(path)
        with path.open('rb') as source: raw = source.read(16 * 1024**2 + 1)
        if len(raw) > 16 * 1024**2: raise ValueError('추론 이미지는 16MB 이하로 선택하세요: ' + path.name)
        return raw

    @staticmethod
    def inference_paths(paths=None, folder=None):
        """Bounded, local selection; folder mode never follows subfolders/links."""
        if folder is not None:
            directory = local_only(folder)
            if not directory.is_dir(): raise ValueError('이미지 폴더를 선택하세요.')
            paths = []
            for entry in directory.iterdir():
                if entry.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'} or entry.is_symlink(): continue
                entry = local_only(entry)
                if entry.is_file(): paths.append(str(entry))
                if len(paths) > 1000: raise ValueError('한 번에 최대 1,000장을 선택하세요.')
            paths.sort(key=lambda x: (Path(x).name.casefold(), Path(x).name))
        if not isinstance(paths, (list, tuple)) or not 1 <= len(paths) <= 1000:
            raise ValueError('이미지 1~1,000장을 선택하세요. 폴더는 바로 아래 이미지만 읽습니다.')
        selected = []; seen = set(); total = 0
        for value in paths:
            path = local_only(value)
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp'} or not path.is_file():
                raise ValueError('지원하는 이미지 파일을 선택하세요: ' + path.name)
            key = os.path.normcase(str(path))
            if key in seen: continue
            seen.add(key); size = path.stat().st_size; total += size
            if size > 16 * 1024**2: raise ValueError('개별 추론 이미지는 16MB 이하로 선택하세요: ' + path.name)
            if total > 512 * 1024**2: raise ValueError('선택한 이미지 합계는 512MB 이하로 나누어 실행하세요.')
            selected.append(str(path))
        return selected

    def inference_comparison(self, run_id, index=None):
        """Read-only AFTER inference: match input pixels, never feed GT to worker."""
        folder=self.store.item('runs',run_id)
        if read(folder/'status.json').get('status') not in ('complete', 'cancelled', 'failed', 'interrupted'):
            raise ValueError('추론 완료 후 정답과 비교할 수 있습니다.')
        result=read(folder/'result.json')
        if result.get('kind') == 'infer_batch':
            if type(index) is not int or not 0 <= index < len(result['items']):
                raise ValueError('이미지 순서를 확인하세요.')
            item = result['items'][index]
            if item.get('status') != 'complete': raise ValueError('이 이미지의 추론은 완료되지 않았습니다.')
            input_path = folder / 'inputs' / f'{index:04d}.image'
            result = dict(item, model_id=result['model_id'], GT_input=False)
        elif result.get('kind') == 'infer':
            input_path = folder/'input.image'
        else: raise ValueError('이미지 추론 결과가 아닙니다.')
        digest,_=pixel_hash(input_path.read_bytes())
        matches=[]
        for dataset in self.store.list('datasets'):
            manifest=read(self.store.item('datasets',dataset['id'])/'manifest.json')
            for row in manifest['rows']:
                if row['pixel_sha256']==digest:
                    matches.append(dict(dataset_name=dataset['name'],labels=row['labels'],split=row['split']))
        comparison=dict(result=result,status='unregistered',matches=matches,labels=[],hits=None)
        if not matches: return comparison
        sets={tuple(sorted(row['labels'])) for row in matches}
        if len(sets)>1:
            comparison['status']='conflict'; return comparison
        labels=matches[0]['labels']
        comparison.update(status='matched',labels=labels,hits=sum(p['label'] in labels for p in result['top5']))
        return comparison

    def launch(self, kind, payload):
        with self.lock:
            if self.is_running():
                raise ValueError('다른 작업이 실행 중입니다. 완료 또는 중단 후 실행하세요.')
            if not all(self.snapshot()['dependencies'].values()):
                raise ValueError('학습 실행 환경이 없습니다. install-engine-windows.bat를 먼저 실행하세요.')
            if kind == 'train':
                plan = self.store.training_plan(payload)
            elif kind == 'infer':
                self.store.check_model_available(payload.get('model_id'))
                model = read(self.store.item('models', payload.get('model_id')) / 'manifest.json')
                if not model.get('has_head'):
                    raise ValueError('학습된 분류 헤드가 없는 모델입니다.')
                if 'images' in payload:
                    inputs = payload['images']
                    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 1000 or not all(isinstance(i, dict) for i in inputs):
                        raise ValueError('이미지 1~1,000장을 선택하세요.')
                    paths = self.inference_paths([i.get('image_path', '') for i in inputs])
                    if len(paths) != len(inputs): raise ValueError('같은 파일이 중복 선택됐습니다.')
                    staged_inputs = []; actual_bytes = 0
                    for index, (item, path) in enumerate(zip(inputs, paths)):
                        description = item.get('description', '')
                        if model.get('engine') == 'bert5_fusion_v1' and (not isinstance(description, str) or not description.strip() or len(description) > 20000):
                            raise ValueError(f'{index+1}번째 이미지 {Path(path).name}: 설명문(1~20,000자)을 입력하세요.')
                        raw = self._inference_bytes(path); pixel_hash(raw); actual_bytes += len(raw)
                        if actual_bytes > 512 * 1024**2: raise ValueError('선택한 이미지 합계는 512MB 이하로 나누어 실행하세요.')
                        record = dict(index=index, file_name=Path(path).name, file_sha256=hashlib.sha256(raw).hexdigest())
                        if model.get('engine') == 'bert5_fusion_v1': record['description'] = description.strip()
                        staged_inputs.append(record)
                    plan = dict(id=uid('infer'), kind='infer', model_id=model['id'], device=payload.get('device', 'cpu'), created=now(), inputs=staged_inputs)
                    if plan['device'] not in ('cpu', 'cuda:0'): raise ValueError('지원하지 않는 장치입니다.')
                else:
                    plan, raw = self._single_inference_plan(model, payload)
            else:
                raise ValueError('지원하지 않는 실행 종류입니다.')
            p = self.store.item('runs', plan['id']); p.mkdir()
            write(p / 'config.json', plan)
            write(p / 'status.json', dict(id=plan['id'], kind=kind, name=plan.get('name', '이미지 추론'),
                  status='queued', created=now(), progress=0, message='작업 준비 중'))
            try:
                if kind == 'infer' and 'inputs' in plan:
                    (p/'inputs').mkdir()
                    for record, path in zip(plan['inputs'], paths):
                        raw = self._inference_bytes(path)
                        if hashlib.sha256(raw).hexdigest() != record['file_sha256']:
                            raise ValueError('선택 이후 이미지가 변경됐습니다: ' + record['file_name'])
                        (p/'inputs'/f'{record["index"]:04d}.image').write_bytes(raw)
                elif kind == 'infer':
                    (p / 'input.image').write_bytes(raw)
            except Exception as e:
                status = read(p/'status.json'); status.update(status='failed', message=str(e))
                write(p/'status.json', status); raise
            env = os.environ.copy()
            env.update(HF_HOME=str(self.store.root / 'cache/huggingface'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       TORCH_HOME=str(self.store.root / 'cache/torch'), TMPDIR=str(self.store.root / 'cache'),
                       TEMP=str(self.store.root / 'cache'), TMP=str(self.store.root / 'cache'), PYTHONUTF8='1')
            executable = Path(sys.executable)
            if executable.name.lower() == 'pythonw.exe':
                executable = executable.with_name('python.exe')
            try:
                with (p / 'worker.log').open('w', encoding='utf-8') as log:
                    self.process = subprocess.Popen([str(executable), str(ROOT / 'worker.py'), '--root', str(self.store.root), '--run', plan['id']],
                        cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            except Exception as e:
                status = read(p / 'status.json'); status.update(status='failed', message=str(e))
                write(p / 'status.json', status); raise
            process = self.process
            def watch():
                rc = process.wait(); status = read(p / 'status.json')
                if status['status'] not in ('complete', 'failed', 'cancelled'):
                    status.update(status='failed', message='작업 프로세스가 종료됐습니다. worker.log를 확인하세요.', returncode=rc)
                    write(p / 'status.json', status)
            threading.Thread(target=watch, daemon=True).start()
            return {'id': plan['id']}

    def _single_inference_plan(self, model, payload):
        description=payload.get('description','')
        if model.get('engine')=='bert5_fusion_v1' and (not isinstance(description,str) or not description.strip() or len(description)>20000):
            raise ValueError('5종 BERT 추론에는 설명문(1~20,000자)이 필요합니다. 이미지 전용 모델과 입력 조건이 다릅니다.')
        path = local_only(payload['image_path'])
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError('추론 이미지는 16MB 이하로 선택하세요.')
        raw = self._inference_bytes(path); pixel_hash(raw)
        plan = dict(id=uid('infer'), kind='infer', model_id=model['id'],
                    device=payload.get('device', 'cpu'), created=now())
        if model.get('engine')=='bert5_fusion_v1':plan['description']=description.strip()
        if plan['device'] not in ('cpu', 'cuda:0'):
            raise ValueError('지원하지 않는 장치입니다.')
        return plan, raw

    def cancel(self, key):
        p = self.store.item('runs', key)
        if read(p / 'status.json')['status'] not in RUNNING:
            raise ValueError('이미 종료된 작업입니다.')
        (p / 'cancel.request').touch()

    def activate(self, key):
        with self.lock:
            self.store.check_model_available(key)
            model = read(self.store.item('models', key) / 'manifest.json')
            if not model.get('has_head'):
                raise ValueError('학습된 분류 헤드가 없습니다.')
            record = dict(model_id=model['id'], changed=now())
            write(self.store.root / 'active.json', record)
            with (self.store.root / 'activation_history.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
            return record
