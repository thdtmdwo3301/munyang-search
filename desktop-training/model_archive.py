"""Portable model backups: selected weights plus their runtime dependencies, not data."""
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from core import Store, LABELS, local_only, read, write, sha, uid, now, validate_id

FORMAT = 'pattern-model-backup-v1'
ENGINES = {'dino_head_v1', 'legacy_dino_rbf_v1', 'bert5_fusion_v1'}
REFERENCES = ('base_model', 'backbone_model')
FIELDS = {'id', 'name', 'engine', 'labels', 'has_head', 'training_supported', 'created',
          'parent_model', 'run_id', 'trained_pixel_hashes', 'provenance', 'weight_hashes',
          'selected_epoch', 'evaluation_f1', 'base_eval_seen', 'selection', 'input_mode',
          'base_model', 'backbone_model', 'source_run_id', 'source_parent_model',
          'imported_from_model', 'imported_at'}
LIMIT = 100 * 1024**3


def relative_path(value):
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or any(ord(c) < 32 for c in value)):
        raise ValueError('가중치 내부 경로가 올바르지 않습니다.')
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        raise ValueError('가중치 내부 경로가 올바르지 않습니다.')
    # File names must also be safe on Windows.
    for part in p.parts:
        if (part.rstrip(' .') != part or any(c in part for c in '<>"|?*')
                or part.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                    *('COM'+str(i) for i in range(1, 10)), *('LPT'+str(i) for i in range(1, 10))}):
            raise ValueError('Windows에서 사용할 수 없는 파일 이름입니다.')
    return value


def model_order(models, selected):
    if not isinstance(models, dict) or not 1 <= len(models) <= 100:
        raise ValueError('모델 개수 또는 목록 형식이 올바르지 않습니다.')
    order, visiting = [], set()
    def visit(key):
        validate_id(key)
        if key in visiting: raise ValueError('모델 참조가 순환합니다.')
        if key in order: return
        if key not in models: raise ValueError('참조하는 원본 가중치가 없습니다: '+key)
        visiting.add(key)
        spec = models[key]
        if spec.get('id') != key or spec.get('engine') not in ENGINES or spec.get('labels') != LABELS:
            raise ValueError('모델 종류 또는 감성어 순서가 올바르지 않습니다.')
        for field in REFERENCES:
            if spec.get(field): visit(spec[field])
        visiting.remove(key); order.append(key)
    visit(selected)
    if set(order) != set(models): raise ValueError('선택 모델과 관계없는 파일이 포함됐습니다.')
    return order


def export_model(store, model_id, destination, overwrite=False):
    store.check_model_available(model_id)
    destination = local_only(destination)
    if destination.suffix.lower() != '.zip': raise ValueError('저장 파일 이름은 .zip으로 끝나야 합니다.')
    if store.root == destination or store.root in destination.parents:
        raise ValueError('프로그램 내부 보관함 대신 별도의 로컬 저장 폴더를 선택하세요.')
    if not destination.parent.is_dir(): raise ValueError('저장할 폴더가 없습니다.')
    if destination.exists() and (not overwrite or not destination.is_file()):
        raise ValueError('같은 이름의 파일이 있습니다. 다른 이름을 선택하세요.')
    models, sources, entries = {}, {}, {}
    def collect(key):
        if key in models: return
        folder = store.item('models', key)
        original = read(local_only(folder/'manifest.json'))
        spec = {k: v for k, v in original.items() if k in FIELDS}
        models[key] = spec
        for field in REFERENCES:
            if spec.get(field): collect(spec[field])
        hashes = spec.get('weight_hashes', {})
        if not hashes: raise ValueError('가중치 무결성 기록이 없어 저장할 수 없습니다: '+key)
        for rel, digest in hashes.items():
            relative_path(rel)
            if not re.fullmatch('[a-f0-9]{64}', digest): raise ValueError('가중치 해시 형식 오류')
            path = local_only(folder/rel)
            if not path.is_file(): raise ValueError('가중치 파일이 없습니다: '+rel)
            arc = key+'/'+rel
            sources[arc] = path
            entries[arc] = dict(sha256=digest, bytes=path.stat().st_size)
        # Keep available model license texts alongside the actual weights.
        for rel in ('backbone/LICENSE.md', 'LICENSE.md', 'LICENSE', 'NOTICE'):
            path = local_only(folder/rel)
            if path.is_file() and key+'/'+rel not in sources:
                sources[key+'/'+rel] = path
                entries[key+'/'+rel] = dict(sha256=sha(path), bytes=path.stat().st_size)
    collect(model_id)
    model_order(models, model_id)
    size = sum(x['bytes'] for x in entries.values())
    if size > LIMIT or len(entries) > 20000: raise ValueError('가중치 백업 크기 제한을 초과했습니다.')
    if shutil.disk_usage(destination.parent).free < size + 64*1024**2:
        raise ValueError('선택한 저장 위치의 여유 공간이 부족합니다.')
    bundle = dict(format=FORMAT, selected=model_id, models=models, files=entries, created=now())
    fd, tmpname = tempfile.mkstemp(prefix='.pattern-save-', suffix='.partial', dir=destination.parent)
    temporary = Path(tmpname)
    try:
        with os.fdopen(fd, 'w+b') as stream:
            with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr('bundle.json', json.dumps(bundle, ensure_ascii=False, allow_nan=False))
                for arc, source in sources.items():
                    digest, count = hashlib.sha256(), 0
                    with local_only(source).open('rb') as src, archive.open(arc, 'w', force_zip64=True) as dst:
                        for block in iter(lambda: src.read(8*1024**2), b''):
                            dst.write(block); digest.update(block); count += len(block)
                    if digest.hexdigest() != entries[arc]['sha256'] or count != entries[arc]['bytes']:
                        raise ValueError('원본 가중치가 변경되었거나 손상되었습니다: '+arc)
            stream.flush(); os.fsync(stream.fileno())
        local_only(destination)
        if destination.exists() and not overwrite: raise ValueError('저장할 파일이 이미 생겼습니다. 다른 이름을 선택하세요.')
        os.replace(temporary, destination)
        return dict(path=str(destination), bytes=destination.stat().st_size, model_id=model_id,
                    model_count=len(models), name=models[model_id].get('name', model_id))
    finally:
        if temporary.exists(): temporary.unlink()


def inspect_archive(path):
    path = local_only(path)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) > 20001 or len({n.casefold() for n in names}) != len(names):
            raise ValueError('중복 파일 또는 너무 많은 파일이 포함됐습니다.')
        for info in infos:
            relative_path(info.filename)
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError('일반 파일만 가져올 수 있습니다.')
            if info.flag_bits & 1: raise ValueError('암호화된 ZIP은 사용할 수 없습니다.')
        if 'bundle.json' not in names or archive.getinfo('bundle.json').file_size > 16*1024**2:
            raise ValueError('이 프로그램의 다른 이름으로 저장으로 만든 ZIP을 선택하세요.')
        bundle = json.loads(archive.read('bundle.json'))
        if bundle.get('format') != FORMAT: raise ValueError('지원하지 않는 가중치 ZIP 형식입니다.')
        order = model_order(bundle['models'], bundle['selected'])
        entries = bundle['files']
        if set(entries) | {'bundle.json'} != set(names): raise ValueError('가중치 목록과 ZIP 파일이 다릅니다.')
        allowed = {key+'/'+relative_path(rel) for key, spec in bundle['models'].items()
                   for rel in spec.get('weight_hashes', {})}
        allowed |= {key+'/'+rel for key in order for rel in ('backbone/LICENSE.md', 'LICENSE.md', 'LICENSE', 'NOTICE')}
        if not set(entries) <= allowed: raise ValueError('모델과 관계없는 파일이 포함됐습니다.')
        size = 0
        for arc, record in entries.items():
            relative_path(arc)
            if arc.split('/')[0] not in order or len(arc.split('/')) < 2:
                raise ValueError('모델 밖의 파일은 가져올 수 없습니다.')
            if arc.split('/', 1)[1] in ('manifest.json', 'registration_check.json', 'import_status.json'):
                raise ValueError('내부 관리 파일은 가져올 수 없습니다.')
            if not re.fullmatch('[a-f0-9]{64}', record.get('sha256', '')):
                raise ValueError('가중치 해시 형식 오류')
            if type(record.get('bytes')) is not int or record['bytes'] < 0 or record['bytes'] != archive.getinfo(arc).file_size:
                raise ValueError('가중치 크기가 일치하지 않습니다.')
            size += record['bytes']
        if size > LIMIT: raise ValueError('가중치 백업 크기 제한을 초과했습니다.')
        for key, spec in bundle['models'].items():
            if not spec.get('weight_hashes'): raise ValueError('모델 파일 해시가 없습니다.')
            for rel, digest in spec['weight_hashes'].items():
                relative_path(rel)
                if entries.get(key+'/'+rel, {}).get('sha256') != digest:
                    raise ValueError('필수 모델 파일의 해시가 일치하지 않습니다.')
        return bundle, size, order


def import_model(store, path, validator):
    path = local_only(path)
    bundle, size, order = inspect_archive(path)
    if shutil.disk_usage(store.root).free < size + 512*1024**2:
        raise ValueError('가중치를 불러올 로컬 저장 공간이 부족합니다.')
    mapping = {old: uid('loaded') for old in order}
    committed = []
    # Stage and load-check the whole dependency graph before adding anything to the shelf.
    with tempfile.TemporaryDirectory(prefix='model-import-', dir=store.root/'imports') as temporary:
        stage = Store(Path(temporary)/'workspace')
        with zipfile.ZipFile(path) as archive:
            if json.loads(archive.read('bundle.json')) != bundle: raise ValueError('선택한 ZIP이 변경됐습니다.')
            for arc, entry in bundle['files'].items():
                old, rel = arc.split('/', 1)
                target = local_only(stage.item('models', mapping[old])/rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                digest, count = hashlib.sha256(), 0
                with archive.open(arc) as src, target.open('xb') as dst:
                    for block in iter(lambda: src.read(8*1024**2), b''):
                        count += len(block)
                        if count > entry['bytes']: raise ValueError('ZIP 파일 크기 제한 초과')
                        digest.update(block); dst.write(block)
                if digest.hexdigest() != entry['sha256'] or count != entry['bytes']:
                    raise ValueError('저장된 가중치가 손상됐습니다: '+arc)
        for old in order:
            spec = {k: v for k, v in bundle['models'][old].items() if k in FIELDS}
            spec['id'] = mapping[old]
            for field in REFERENCES:
                if spec.get(field): spec[field] = mapping[spec[field]]
            # An imported model must never own or delete a source workspace's training run.
            if spec.get('run_id'): spec['source_run_id'] = spec.pop('run_id')
            if spec.get('parent_model'):
                parent = spec.pop('parent_model')
                spec['source_parent_model'] = parent
            spec['imported_from_model'] = old
            spec['imported_at'] = now()
            write(stage.item('models', spec['id'])/'manifest.json', spec)
            report = validator(spec['id'], spec, store=stage)
            if report.get('status') != 'passed': raise ValueError('가중치 로딩 검증 실패')
            spec['load_check'] = report
            write(stage.item('models', spec['id'])/'manifest.json', spec)
        try:
            for old in order:
                key = mapping[old]; destination = store.item('models', key)
                if destination.exists(): raise ValueError('같은 모델 ID가 이미 있습니다.')
                stage.item('models', key).rename(destination); committed.append(destination)
        except Exception:
            for folder in reversed(committed): shutil.rmtree(folder)
            raise
    selected = mapping[bundle['selected']]
    return dict(id=selected, name=bundle['models'][bundle['selected']].get('name', selected),
                model_count=len(order), automatic_activation=False)
