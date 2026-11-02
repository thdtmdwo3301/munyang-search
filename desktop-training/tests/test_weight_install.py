import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from install_weights import ROOT, MODELS, WeightInstaller, valid_url, validate_manifest, workspace_lock


def digest(data):
    return hashlib.sha256(data).hexdigest()


class WeightInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.client = self.folder / 'client'
        self.client.mkdir()
        self.parts = self.folder / 'parts'
        self.parts.mkdir()
        self.files = {}
        for mid in sorted(MODELS):
            self.files[f'workspace/models/{mid}/manifest.json'] = json.dumps({'id': mid}).encode()
            self.files[f'workspace/models/{mid}/weights.bin'] = bytes(range(256)) * 20
        default = {'model_id': 'bundled_image_v1', 'manifest_sha256': digest(self.files['workspace/models/bundled_image_v1/manifest.json'])}
        self.files['default_model.json'] = (json.dumps(default, indent=2) + '\n').encode()
        self.manifest = dict(format='pattern-weight-parts-v1', archive='weights.zip', models=sorted(MODELS), files=sorted(self.files), file_hashes={k:digest(v) for k,v in self.files.items()}, file_sizes={k:len(v) for k,v in self.files.items()}, default_model=default)
        self.pack(self.files)

    def pack(self, files):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            for name, data in files.items():
                z.writestr(name, data)
        data = archive.getvalue()
        self.manifest.update(bytes=len(data), sha256=digest(data), parts=[])
        for index, offset in enumerate(range(0, len(data), 4096), 1):
            name = f'weights.zip.part{index:03d}'
            chunk = data[offset:offset+4096]
            (self.parts / name).write_bytes(chunk)
            self.manifest['parts'].append(dict(file=name, bytes=len(chunk), sha256=digest(chunk)))

    def installer(self, **kwargs):
        return WeightInstaller(self.client, self.manifest, {}, self.parts, **kwargs)

    def test_local_roundtrip_preserves_user_models_and_selection(self):
        active = self.client / 'workspace/active.json'
        active.parent.mkdir()
        active.write_text('{"model_id":"my-trained-model"}')
        user = self.client / 'workspace/models/my-trained-model/user.bin'
        user.parent.mkdir(parents=True)
        user.write_bytes(b'keep me')
        result = self.installer().run()
        self.assertEqual(result['status'], 'installed')
        for name, value in self.files.items():
            self.assertEqual((self.client/name).read_bytes(), value)
        self.assertEqual(active.read_text(), '{"model_id":"my-trained-model"}')
        self.assertEqual(user.read_bytes(), b'keep me')
        self.assertTrue(all((self.parts/p['file']).exists() for p in self.manifest['parts']))
        with patch.object(WeightInstaller, 'download', side_effect=AssertionError('unexpected download')):
            self.assertEqual(self.installer().run()['status'], 'already_installed')

    def test_default_recovery_after_interruption_and_preservation(self):
        self.installer().run()
        default = self.client/'default_model.json'
        default.unlink()
        self.installer().run()
        self.assertEqual(default.read_bytes(), self.files['default_model.json'])
        default.write_text('user-controlled settings')
        self.installer().run()
        self.assertEqual(default.read_text(), 'user-controlled settings')

    def test_modified_original_never_overwritten(self):
        self.installer().run()
        path = self.client/'workspace/models/bundled_image_v1/weights.bin'
        path.write_bytes(b'user changed this')
        with self.assertRaisesRegex(ValueError, 'nothing was overwritten'):
            self.installer().run()
        self.assertEqual(path.read_bytes(), b'user changed this')

    def test_missing_sources_fails_before_install(self):
        runner = WeightInstaller(self.client, self.manifest, {})
        with self.assertRaisesRegex(ValueError, 'URLs are not configured'):
            runner.run()
        self.assertFalse((self.client/'workspace/models').exists())

    def test_corrupt_local_part_rejected_and_preserved(self):
        path = self.parts/self.manifest['parts'][0]['file']
        data = path.read_bytes()
        path.write_bytes(b'x' + data[1:])
        with self.assertRaisesRegex(ValueError, 'damaged'):
            self.installer().run()
        self.assertFalse((self.client/'workspace/models').exists())
        self.assertEqual(path.read_bytes(), b'x' + data[1:])

    def test_file_hash_failure_prevents_all_model_commits(self):
        bad_files = dict(self.files)
        key = 'workspace/models/bundled_image_v1/weights.bin'
        bad_files[key] = b'x' * len(bad_files[key])
        self.pack(bad_files)
        with self.assertRaisesRegex(ValueError, 'Extracted file failed'):
            self.installer().run()
        self.assertFalse((self.client/'workspace/models').exists())

    def test_extra_zip_entry_rejected(self):
        self.pack({**self.files, '../../escape.txt': b'bad'})
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.installer().run()
        self.assertFalse((self.folder/'escape.txt').exists())

    def test_deleted_original_stays_deleted(self):
        trash = self.client/'workspace/model_trash/bundled_image_v1.json'
        trash.parent.mkdir(parents=True)
        trash.write_text('{"status":"purged"}')
        result = self.installer().run()
        self.assertEqual(result['installed'], ['bundled_bert5_v1'])
        self.assertFalse((self.client/'workspace/models/bundled_image_v1').exists())
        self.assertFalse((self.client/'default_model.json').exists())

    def test_running_application_blocks_install(self):
        with workspace_lock(self.client):
            with self.assertRaisesRegex(RuntimeError, 'Close the desktop'):
                self.installer().run()

    def test_symlink_and_unsafe_manifest_rejected(self):
        link = self.client/'workspace'
        try:
            link.symlink_to(self.folder, target_is_directory=True)
        except OSError:
            self.skipTest('Symlink permission unavailable')
        with self.assertRaises((RuntimeError, ValueError)):
            self.installer().run()
        self.manifest['parts'][0]['file'] = '../escape'
        with self.assertRaisesRegex(ValueError, 'order'):
            validate_manifest(self.manifest)

    def test_url_scheme_and_embedded_credentials_rejected(self):
        for url in ['http://example.org/a', 'file:///tmp/a', 'https://user:pass@example.org/a', 'https://example.org/a#fragment']:
            with self.assertRaises(ValueError):
                valid_url(url)
        self.assertEqual(valid_url('https://example.org/a'), 'https://example.org/a')

    def test_network_download_retry_reuses_verified_parts(self):
        hits = []
        directory = self.parts
        fail_name = self.manifest['parts'][1]['file']
        failures = [True]

        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(directory), **kwargs)

            def log_message(self, *args):
                pass

            def do_GET(self):
                hits.append(self.path)
                if self.path == '/' + fail_name and failures[0]:
                    failures[0] = False
                    self.send_response(200); self.end_headers(); self.wfile.write(b'corrupt')
                else:
                    super().do_GET()

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        sources = {'base_url': f'http://127.0.0.1:{server.server_port}'}
        runner = WeightInstaller(self.client, self.manifest, sources, allow_test_http=True)
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            runner.run()
        result = WeightInstaller(self.client, self.manifest, sources, allow_test_http=True).run()
        self.assertEqual(result['status'], 'installed')
        self.assertEqual(hits.count('/'+self.manifest['parts'][0]['file']), 1)
        self.assertEqual(hits.count('/'+fail_name), 2)
        self.assertFalse((runner.cache/'weights.zip').exists())
        self.assertTrue((runner.cache/'installed.json').exists())

    def test_partial_commit_retry(self):
        import os
        original = os.rename
        calls = []

        def interrupted(source, dest):
            calls.append(dest)
            if len(calls) == 2:
                raise OSError('simulated interruption')
            return original(source, dest)

        with patch('install_weights.os.rename', side_effect=interrupted):
            with self.assertRaises(OSError):
                self.installer().run()
        self.assertTrue((self.client/'workspace/models/bundled_bert5_v1').exists())
        result = self.installer().run()
        self.assertEqual(result['installed'], ['bundled_image_v1'])
        self.assertEqual((self.client/'default_model.json').read_bytes(), self.files['default_model.json'])

    def test_install_script_hooks_fail_fast(self):
        windows = (ROOT/'install-windows.bat').read_text()
        self.assertIn('%PYTHON_CMD% install_weights.py\nif errorlevel 1 goto fail', windows)
        self.assertLess(windows.index('%PYTHON_CMD% install_weights.py'), windows.index('-m pip install'))
        self.assertIn('if not exist "weights_parts.json" goto weights_ready', windows)
        linux = (ROOT/'install-linux.sh').read_text()
        self.assertIn('if [[ -f weights_parts.json ]]; then "$python_cmd" install_weights.py; fi', linux)


if __name__ == '__main__':
    unittest.main()
