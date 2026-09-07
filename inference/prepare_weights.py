"""Verify original Git LFS weights and restore the split XLM-R checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record['size'] or digest(path) != record['sha256']:
        raise RuntimeError('Missing or mismatched weight: ' + str(path) + '. Run git lfs pull; existing files will not be overwritten.')


def prepare(manifest, directory):
    for model in manifest['models']:
        target = directory / model['name']
        if target.exists():
            verify(target, model)
            print(model['name'] + ': SHA-256 verified', flush=True)
            continue
        if len(model['parts']) == 1:
            verify(target, model)
        for part in model['parts']:
            verify(directory / part['name'], part)
        temporary = target.with_suffix(target.suffix + '.restore')
        if temporary.exists():
            raise RuntimeError('Previous partial restore exists; move it before retrying: ' + str(temporary))
        try:
            with temporary.open('xb') as output:
                for part in model['parts']:
                    with (directory / part['name']).open('rb') as source:
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                            output.write(block)
            verify(temporary, model)
            if target.exists():
                raise RuntimeError('Destination appeared during restore: ' + str(target))
            temporary.rename(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        print(model['name'] + ': restored and SHA-256 verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-dir', type=Path, default=HERE / 'weights')
    args = parser.parse_args()
    manifest = json.loads((HERE / 'weights-manifest.json').read_text())
    prepare(manifest, args.weights_dir)
