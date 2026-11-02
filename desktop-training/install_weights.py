"""Install pinned, split model assets. Standard library only; never splits weights."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

from core_storage_check import check

ROOT = Path(__file__).absolute().parent
BLOCK = 8 * 1024**2
MODELS = {'bundled_image_v1', 'bundled_bert5_v1'}


def safe(path):
    path = Path(os.path.abspath(path))
    check(path)
    for item in [path] + list(path.parents):
        if item.is_symlink() or (item.exists() and getattr(item.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Linked paths are not supported: ' + str(path))
    return path


def sha(path):
    h = hashlib.sha256()
    with safe(path).open('rb') as source:
        for block in iter(lambda: source.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()


def matches(path, size, expected):
    path = safe(path)
    return path.is_file() and path.stat().st_size == size and sha(path) == expected


def validate_manifest(value):
    if value.get('format') != 'pattern-weight-parts-v1' or set(value['models']) != MODELS:
        raise ValueError('Unsupported weight manifest')
    if value.get('archive') != 'weights.zip' or not value.get('parts'):
        raise ValueError('Invalid archive manifest')
    for index, part in enumerate(value['parts'], 1):
        if part['file'] != f'weights.zip.part{index:03d}' or type(part['bytes']) is not int or part['bytes'] <= 0:
            raise ValueError('Invalid part order or size')
        if not re.fullmatch('[0-9a-f]{64}', part['sha256']):
            raise ValueError('Invalid part digest')
    if value['bytes'] != sum(p['bytes'] for p in value['parts']) or not re.fullmatch('[0-9a-f]{64}', value['sha256']):
        raise ValueError('Invalid archive size or digest')
    files = value['files']
    if len(files) != len(set(files)) or set(files) != set(value['file_hashes']) or set(files) != set(value['file_sizes']):
        raise ValueError('Invalid file inventory')
    if 'default_model.json' not in files:
        raise ValueError('Default model metadata is missing')
    for name in files:
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name or str(rel) != name:
            raise ValueError('Unsafe archive path')
        if name != 'default_model.json' and not any(name.startswith('workspace/models/' + mid + '/') for mid in MODELS):
            raise ValueError('File outside bundled model allowlist')
        if not re.fullmatch('[0-9a-f]{64}', value['file_hashes'][name]):
            raise ValueError('Invalid file digest')
        if type(value['file_sizes'][name]) is not int or value['file_sizes'][name] < 0:
            raise ValueError('Invalid file size')
    for mid in MODELS:
        if f'workspace/models/{mid}/manifest.json' not in files:
            raise ValueError('Model metadata is missing')
    default = value['default_model']
    mid = default.get('model_id')
    if mid not in MODELS or default.get('manifest_sha256') != value['file_hashes'][f'workspace/models/{mid}/manifest.json']:
        raise ValueError('Invalid default model binding')
    encoded = (json.dumps(default, indent=2) + '\n').encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != value['file_hashes']['default_model.json'] or len(encoded) != value['file_sizes']['default_model.json']:
        raise ValueError('Default metadata digest mismatch')
    return value


def valid_url(url, allow_test_http=False):
    parsed = urllib.parse.urlsplit(url)
    test_http = allow_test_http and parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost'}
    if (parsed.scheme != 'https' and not test_http) or not parsed.hostname or parsed.username or parsed.password or parsed.fragment or any(c in url for c in '\r\n'):
        raise ValueError('Use a direct HTTPS download URL without embedded credentials')
    return url


class SecureRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_test_http=False):
        self.allow_test_http = allow_test_http

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        valid_url(newurl, self.allow_test_http)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@contextmanager
def workspace_lock(root):
    workspace = safe(root / 'workspace')
    workspace.mkdir(exist_ok=True)
    stream = safe(workspace / 'application.lock').open('a+b')
    locked = False
    try:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            if not stream.read(1):
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    except OSError as exc:
        if not locked:
            raise RuntimeError('Close the desktop application and other installers first.') from exc
        raise
    finally:
        if locked and os.name == 'nt':
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        stream.close()


class WeightInstaller:
    def __init__(self, root, manifest, sources, parts_dir=None, allow_test_http=False):
        self.root = safe(root)
        self.manifest = validate_manifest(manifest)
        self.sources = sources
        self.local_parts = safe(parts_dir if parts_dir is not None else self.root / 'weight_parts')
        self.cache = safe(self.root / 'workspace/cache/weight_install' / manifest['sha256'])
        self.allow_test_http = allow_test_http
        self.opener = urllib.request.build_opener(SecureRedirect(allow_test_http))
        self.downloaded = []

    def url(self, name):
        direct = self.sources.get('urls', {}).get(name)
        base = self.sources.get('base_url', '').strip()
        url = direct or (base.rstrip('/') + '/' + name if base else '')
        if not url:
            raise ValueError('Download URLs are not configured. Ask the distributor to fill weight_sources.json, or place all six parts in the weight_parts folder.')
        return valid_url(url, self.allow_test_http)

    def installed(self):
        missing, preserved = [], []
        for mid in sorted(MODELS):
            if safe(self.root / 'workspace/model_trash' / (mid + '.json')).exists():
                preserved.append(mid + ' (user-deleted; not restored)')
                continue
            folder = safe(self.root / 'workspace/models' / mid)
            if not folder.exists():
                missing.append(mid)
                continue
            for name in self.manifest['files']:
                if name.startswith('workspace/models/' + mid + '/'):
                    if not matches(self.root / name, self.manifest['file_sizes'][name], self.manifest['file_hashes'][name]):
                        raise ValueError('Existing model is incomplete or changed; nothing was overwritten. Use a new empty program folder: ' + mid)
            preserved.append(mid)
        return missing, preserved

    def download(self, part):
        dest = safe(self.cache / part['file'])
        url = self.url(part['file'])
        partial = safe(dest.with_name(dest.name + '.partial'))
        print('Downloading ' + part['file'] + ' ...', flush=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'PatternTraining-Installer/0.13', 'Accept-Encoding': 'identity'})
        h, count = hashlib.sha256(), 0
        next_progress = 10
        with self.opener.open(request, timeout=90) as response, partial.open('wb') as output:
            valid_url(response.geturl(), self.allow_test_http)
            if response.status != 200:
                raise ValueError('Unexpected download response')
            while True:
                block = response.read(BLOCK)
                if not block:
                    break
                count += len(block)
                if count > part['bytes']:
                    raise ValueError('Download exceeded the expected size: ' + part['file'])
                output.write(block)
                h.update(block)
                progress = count * 100 // part['bytes']
                if progress >= next_progress:
                    print(f"  {part['file']}: {min(progress, 100)}%", flush=True)
                    next_progress = (progress // 10 + 1) * 10
        if count != part['bytes'] or h.hexdigest() != part['sha256']:
            raise ValueError('Downloaded file failed SHA-256 verification: ' + part['file'])
        os.replace(partial, dest)
        self.downloaded.append(part['file'])
        return dest

    def parts(self):
        paths = []
        for index, part in enumerate(self.manifest['parts'], 1):
            print(f"Checking part {index}/{len(self.manifest['parts'])} ...", flush=True)
            local = safe(self.local_parts / part['file'])
            cached = safe(self.cache / part['file'])
            if local.exists():
                if not matches(local, part['bytes'], part['sha256']):
                    raise ValueError('Local part is damaged; replace it and retry: ' + part['file'])
                paths.append(local)
            elif matches(cached, part['bytes'], part['sha256']):
                paths.append(cached)
            else:
                paths.append(self.download(part))
        return paths

    def restore(self, paths):
        archive = safe(self.cache / 'weights.zip')
        partial = safe(self.cache / 'weights.zip.partial')
        h, count = hashlib.sha256(), 0
        print('Reassembling and verifying model archive ...', flush=True)
        with partial.open('wb') as output:
            for path in paths:
                with safe(path).open('rb') as source:
                    for block in iter(lambda: source.read(BLOCK), b''):
                        output.write(block); h.update(block); count += len(block)
        if count != self.manifest['bytes'] or h.hexdigest() != self.manifest['sha256']:
            raise ValueError('Reassembled archive failed SHA-256 verification')
        os.replace(partial, archive)
        return archive

    def unpack(self, archive, stage):
        print('Verifying and installing model files ...', flush=True)
        with zipfile.ZipFile(archive) as z:
            if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(self.manifest['files']):
                raise ValueError('Archive file inventory does not match')
            for info in z.infolist():
                name = info.filename
                if info.file_size != self.manifest['file_sizes'][name] or stat.S_ISLNK(info.external_attr >> 16) or info.is_dir():
                    raise ValueError('Invalid archive member: ' + name)
                output = safe(stage / name)
                output.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with z.open(info) as source, output.open('xb') as dest:
                    for block in iter(lambda: source.read(BLOCK), b''):
                        dest.write(block); h.update(block)
                if h.hexdigest() != self.manifest['file_hashes'][name]:
                    raise ValueError('Extracted file failed SHA-256 verification: ' + name)
        default = json.loads((stage / 'default_model.json').read_text(encoding='utf-8'))
        mid = default.get('model_id')
        if mid not in MODELS or default.get('manifest_sha256') != self.manifest['file_hashes'][f'workspace/models/{mid}/manifest.json']:
            raise ValueError('Invalid default model binding')
        return default

    def run(self):
        with workspace_lock(self.root):
            print('Checking existing models; existing weights will not be overwritten ...', flush=True)
            missing, preserved = self.installed()
            if not missing:
                self.ensure_default()
                print('Existing models and active selection preserved. No download needed.', flush=True)
                return {'status': 'already_installed', 'installed': [], 'preserved': preserved, 'downloaded': []}
            self.cache.mkdir(parents=True, exist_ok=True)
            # Validate all URLs/available parts before downloading any large asset.
            to_download = 0
            for part in self.manifest['parts']:
                local = safe(self.local_parts / part['file'])
                cached = safe(self.cache / part['file'])
                if local.exists():
                    if not local.is_file() or local.stat().st_size != part['bytes']:
                        raise ValueError('Local part size mismatch: ' + part['file'])
                elif not cached.exists():
                    self.url(part['file']); to_download += part['bytes']
                else:
                    # A cached part is checked by SHA-256 below; reserve retry space.
                    to_download += part['bytes']
            required = to_download + self.manifest['bytes'] + sum(self.manifest['file_sizes'].values()) + 512 * 1024**2
            if shutil.disk_usage(self.cache).free < required:
                raise RuntimeError(f'Not enough disk space. Free at least {required / 1024**3:.1f} GiB and retry.')
            archive = self.restore(self.parts())
            with tempfile.TemporaryDirectory(prefix='extract-', dir=self.cache) as temporary:
                stage = safe(temporary)
                default = self.unpack(archive, stage)
                for mid in missing:
                    dest = safe(self.root / 'workspace/models' / mid)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if dest.exists():
                        raise FileExistsError('Model appeared during installation; refusing to overwrite it')
                    os.rename(stage / 'workspace/models' / mid, dest)
                self.ensure_default()
            result = dict(status='installed', installed=missing, preserved=preserved, downloaded=self.downloaded, archive_sha256=self.manifest['sha256'])
            # Remove only this installer's disposable cache, never caller-supplied parts.
            for name in ['weights.zip', 'weights.zip.partial'] + [p['file'] for p in self.manifest['parts']] + [p['file'] + '.partial' for p in self.manifest['parts']]:
                path = safe(self.cache / name)
                if path.parent != self.local_parts and path.is_file():
                    path.unlink()
            safe(self.cache / 'installed.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
            print('Weights ready. Both models are available; existing trained models and active selection were preserved.', flush=True)
            return result

    def ensure_default(self):
        default = self.manifest['default_model']
        dest = safe(self.root / 'default_model.json')
        deleted = safe(self.root / 'workspace/model_trash' / (default['model_id'] + '.json'))
        if not dest.exists() and not deleted.exists():
            with dest.open('xb') as output:
                output.write((json.dumps(default, indent=2) + '\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts-dir', type=Path, help='Optional existing local folder containing all weight parts')
    args = parser.parse_args()
    try:
        manifest = json.loads(safe(ROOT / 'weights_parts.json').read_text(encoding='utf-8'))
        sources = json.loads(safe(ROOT / 'weight_sources.json').read_text(encoding='utf-8'))
        WeightInstaller(ROOT, manifest, sources, args.parts_dir).run()
        return 0
    except Exception as exc:
        # Do not echo signed download URLs or credential-bearing HTTP exceptions.
        message = str(exc) if isinstance(exc, (ValueError, FileExistsError, RuntimeError)) else type(exc).__name__ + ': installation failed; check connection, disk space and retry.'
        print('Weight setup failed: ' + message, file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
