"""Transactional relocation of registered data/models; never touch source imports."""
import hashlib
import os
import shutil
import stat
from pathlib import Path
from core import Store, local_only, read, write, sha, uid, now


def inventory(folder):
    result=[]; directories=[]
    for parent, children, files in os.walk(local_only(folder),followlinks=False):
        base=local_only(parent); directories.append(base)
        for name in children: local_only(base/name)
        for name in files:
            path=local_only(base/name); info=path.stat()
            if not stat.S_ISREG(info.st_mode): raise ValueError('일반 파일만 저장 위치를 변경할 수 있습니다: '+str(path))
            result.append(dict(relative=path.relative_to(folder).as_posix(),size=info.st_size,
                               mtime_ns=info.st_mtime_ns,inode=info.st_ino))
    return sorted(result,key=lambda x:x['relative']),directories


def nearest_existing(path):
    while not path.exists(): path=path.parent
    return local_only(path)


def copy_verified(source,target,on_created=None):
    digest=hashlib.sha256()
    with local_only(source).open('rb') as src,local_only(target).open('xb') as dst:
        if on_created:on_created(target,target.stat().st_ino)
        for block in iter(lambda:src.read(8*1024**2),b''):
            dst.write(block);digest.update(block)
        dst.flush();os.fsync(dst.fileno())
    if sha(target)!=digest.hexdigest(): raise ValueError('저장 위치 변경 중 복사 검증에 실패했습니다.')
    return digest.hexdigest()


def relocate(store,datasets,models):
    destinations=store.validate_locations(dict(datasets=local_only(datasets),models=local_only(models)))
    changes=[]
    for kind,target in destinations.items():
        source=store.kind_root(kind)
        if source==target: continue
        # Never merge another installation or move into/around the old storage.
        for current in store.locations.values():
            if target==current or target in current.parents or current in target.parents:
                raise ValueError('기존 저장 폴더와 겹치지 않는 빈 폴더를 선택하세요.')
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise ValueError('새 저장 위치는 비어 있는 폴더여야 합니다: '+str(target))
        files,dirs=inventory(source)
        changes.append(dict(kind=kind,source=source,target=target,files=files,dirs=dirs,
                            target_existed=target.exists(),created=[]))
    if not changes: return dict(changed=False,paths={k:str(v) for k,v in destinations.items()},warnings=[])
    required={}
    for item in changes:
        disk=nearest_existing(item['target']); device=disk.stat().st_dev
        size=sum(f['size'] for f in item['files'])
        record=required.setdefault(device,dict(path=disk,bytes=0)); record['bytes']+=size
    for record in required.values():
        if shutil.disk_usage(record['path']).free<record['bytes']+64*1024**2:
            raise ValueError('복사·검증을 위한 새 저장 위치의 여유 공간이 부족합니다.')
    journal=store.root/'storage_migrations'/(uid('move')+'.json')
    log=dict(status='copying',created=now(),changes=[dict(kind=i['kind'],source=str(i['source']),
             target=str(i['target']),files=i['files']) for i in changes])
    write(journal,log)
    committed=False
    try:
        for item in changes:
            target=item['target']
            if target.exists() and any(target.iterdir()):raise ValueError('선택한 새 폴더에 파일이 생겼습니다. 덮어쓰지 않습니다.')
            target.mkdir(parents=True,exist_ok=True)
            for directory in item['dirs']:
                local_only(target/directory.relative_to(item['source'])).mkdir(parents=True,exist_ok=True)
            for record in item['files']:
                dest=local_only(target/record['relative'])
                if dest.exists(): raise ValueError('복사 중 대상 파일이 생겼습니다. 덮어쓰지 않습니다.')
                record['sha256']=copy_verified(item['source']/record['relative'],dest,
                    lambda path,inode,i=item:i['created'].append((path,inode)))
            current,_=inventory(item['source'])
            expected=[{k:v for k,v in f.items() if k!='sha256'} for f in item['files']]
            if current!=expected: raise ValueError('이동 중 원본 자료가 변경됐습니다. 기존 위치를 유지합니다.')
            copied,copied_dirs=inventory(target)
            if [(f['relative'],f['size']) for f in copied]!=[(f['relative'],f['size']) for f in item['files']]:
                raise ValueError('새 저장 폴더에 예상하지 못한 파일이 있습니다.')
            if {p.relative_to(target) for p in copied_dirs}!={p.relative_to(item['source']) for p in item['dirs']}:
                raise ValueError('새 저장 폴더에 예상하지 못한 폴더가 있습니다.')
            item['copied_inventory']=copied
        # Verify all source trees once more before the single atomic configuration switch.
        for item in changes:
            current,_=inventory(item['source'])
            if current!=[{k:v for k,v in f.items() if k!='sha256'} for f in item['files']]:
                raise ValueError('이동 중 원본 자료가 변경됐습니다. 기존 위치를 유지합니다.')
            if inventory(item['target'])[0]!=item['copied_inventory']:
                raise ValueError('검증 후 새 저장 폴더가 변경됐습니다. 기존 위치를 유지합니다.')
        write(store.root/'storage.json',dict(version=1,**{k:str(v) for k,v in destinations.items()}))
        committed=True
        log.update(status='committed',committed=now());write(journal,log)
    except Exception as exc:
        if not committed:
            # Roll back only our exact copies; unknown files/directories are left alone.
            for item in changes:
                for path,inode in reversed(item['created']):
                    if path.is_file() and not path.is_symlink() and path.stat().st_ino==inode:path.unlink()
                for source_dir in sorted(item['dirs'],key=lambda p:len(p.parts),reverse=True):
                    path=local_only(item['target']/source_dir.relative_to(item['source']))
                    if path==item['target'] and item['target_existed']: continue
                    try:path.rmdir()
                    except OSError:pass
            log.update(status='failed_before_commit',error=str(exc));write(journal,log)
        raise
    warnings=[]
    # New paths are live and complete. Retire old copies only when still identical.
    for item in changes:
        try:
            current,_=inventory(item['source'])
            if current!=[{k:v for k,v in f.items() if k!='sha256'} for f in item['files']]:
                raise ValueError('이전 폴더가 변경되어 자동 정리를 하지 않았습니다.')
            for record in item['files']: local_only(item['source']/record['relative']).unlink()
            for directory in sorted(item['dirs'],key=lambda p:len(p.parts),reverse=True):directory.rmdir()
        except (OSError,ValueError) as exc:
            warnings.append(str(item['source'])+': '+str(exc))
    log.update(status='complete' if not warnings else 'cleanup_pending',completed=now(),warnings=warnings)
    write(journal,log)
    return dict(changed=True,paths={k:str(v) for k,v in destinations.items()},warnings=warnings,
                moved_bytes=sum(f['size'] for i in changes for f in i['files']),journal=str(journal))
