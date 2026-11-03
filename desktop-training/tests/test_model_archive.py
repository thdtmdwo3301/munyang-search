import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from app_service import App
from core import LABELS, read, write, sha


class ModelArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent.parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.app = App(self.root/'source')
        self.target = App(self.root/'target')
        self.archive = self.root/'추가학습 저장.zip'
        self.add_model('base', {'backbone/model.safetensors': b'backbone', 'backbone/config.json': b'{}'})
        self.add_model('child', {'head.safetensors': b'new-head'}, backbone_model='base', selected_epoch=3,
                       trained_pixel_hashes=['f'*64], evaluation_f1=.4, provenance='test provenance')
        self.app.activate('child')

    def add_model(self, key, files, **extra):
        folder = self.app.store.item('models', key)
        for rel, blob in files.items():
            path = folder/rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blob)
        write(folder/'manifest.json', dict(id=key, name='테스트 '+key, engine='dino_head_v1', labels=LABELS,
            has_head=True, weight_hashes={r: sha(folder/r) for r in files}, **extra))

    def load(self):
        with patch.object(self.target, 'snapshot', return_value={'dependencies': {'torch': True}}), \
             patch.object(self.target, 'validate_model', return_value={'status': 'passed'}) as probe:
            result = self.target.import_model_archive(self.archive)
        return result, probe

    def rewrite(self, change):
        with zipfile.ZipFile(self.archive) as z: entries = {n: z.read(n) for n in z.namelist()}
        change(entries)
        with zipfile.ZipFile(self.archive, 'w') as z:
            for name, data in entries.items(): z.writestr(name, data)

    def test_roundtrip_dependencies_and_selection_preserved(self):
        before = read(self.app.store.root/'active.json')
        result = self.app.export_model('child', self.archive)
        self.assertEqual(result['model_count'], 2)
        with zipfile.ZipFile(self.archive) as z:
            self.assertNotIn('active.json', z.namelist())
            self.assertFalse(any('datasets/' in n or 'runs/' in n for n in z.namelist()))
        loaded, probe = self.load()
        self.assertEqual(probe.call_count, 2)
        child = read(self.target.store.item('models', loaded['id'])/'manifest.json')
        self.assertNotEqual(child['backbone_model'], 'base')
        self.assertEqual(child['selected_epoch'], 3)
        self.assertEqual(child['trained_pixel_hashes'], ['f'*64])
        self.assertEqual(child['provenance'], 'test provenance')
        self.assertEqual((self.target.store.item('models', child['id'])/'head.safetensors').read_bytes(), b'new-head')
        self.assertEqual(read(self.app.store.root/'active.json'), before)
        self.assertFalse((self.target.store.root/'active.json').exists())
        second, _ = self.load()
        self.assertNotEqual(loaded['id'], second['id'])

    def test_overwrite_requires_permission_and_original_is_preserved_on_failure(self):
        self.app.export_model('child', self.archive); original = self.archive.read_bytes()
        with self.assertRaisesRegex(ValueError, '같은 이름'): self.app.export_model('child', self.archive)
        (self.app.store.item('models', 'child')/'head.safetensors').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, '손상'): self.app.export_model('child', self.archive, True)
        self.assertEqual(self.archive.read_bytes(), original)
        self.assertFalse(list(self.root.glob('*.partial')))

    def test_corruption_rejected_before_registration(self):
        self.app.export_model('child', self.archive)
        self.rewrite(lambda entries: entries.update({'child/head.safetensors': b'bad-head'}))
        with self.assertRaises(ValueError): self.load()
        self.assertEqual(self.target.store.list('models'), [])

    def test_path_escape_and_unexpected_file_rejected(self):
        self.app.export_model('child', self.archive)
        self.rewrite(lambda entries: entries.update({'../escape': b'bad'}))
        with self.assertRaises(ValueError): self.load()
        self.assertFalse((self.root/'escape').exists())

    def test_failed_load_does_not_register_partial_graph(self):
        self.app.export_model('child', self.archive)
        with patch.object(self.target, 'snapshot', return_value={'dependencies': {'torch': True}}), \
             patch.object(self.target, 'validate_model', side_effect=[{'status': 'passed'}, ValueError('bad head')]):
            with self.assertRaisesRegex(ValueError, 'bad head'): self.target.import_model_archive(self.archive)
        self.assertEqual(self.target.store.list('models'), [])
        self.assertEqual(list((self.target.store.root/'imports').iterdir()), [])

    def test_missing_dependency_rejected(self):
        spec = read(self.app.store.item('models', 'child')/'manifest.json')
        spec['backbone_model'] = 'missing'; write(self.app.store.item('models', 'child')/'manifest.json', spec)
        with self.assertRaises(FileNotFoundError): self.app.export_model('child', self.archive)

    def test_internal_or_network_destination_rejected(self):
        for path in (self.app.store.root/'backup.zip', Path('/home/mount/NAS_MOUNT/backup.zip')):
            with self.assertRaises(ValueError): self.app.export_model('child', path)

    def test_busy_and_deleted_models_rejected(self):
        with patch.object(self.app, 'is_running', return_value=True):
            with self.assertRaises(ValueError): self.app.export_model('child', self.archive)
            with self.assertRaises(ValueError): self.app.import_model_archive(self.archive)
        write(self.app.store.root/'model_trash/child.json', {'status': 'deleted'})
        with self.assertRaises(ValueError): self.app.export_model('child', self.archive)

    def test_insufficient_disk_no_partial_file(self):
        usage = type('Usage', (), {'free': 0})()
        with patch('model_archive.shutil.disk_usage', return_value=usage):
            with self.assertRaisesRegex(ValueError, '공간'): self.app.export_model('child', self.archive)
        self.assertFalse(self.archive.exists())

    def test_cycle_and_wrong_label_order_rejected(self):
        self.app.export_model('child', self.archive)
        def mutate(entries):
            bundle = json.loads(entries['bundle.json'])
            bundle['models']['base']['backbone_model'] = 'child'
            entries['bundle.json'] = json.dumps(bundle).encode()
        self.rewrite(mutate)
        with self.assertRaisesRegex(ValueError, '순환'): self.load()

    def test_import_does_not_claim_or_delete_original_run(self):
        child = self.app.store.item('models', 'child')
        spec = read(child/'manifest.json'); spec['run_id'] = 'original_run'
        spec['parent_model'] = 'base'; write(child/'manifest.json', spec)
        original_run = self.app.store.item('runs', 'original_run'); original_run.mkdir()
        write(original_run/'result.json', dict(model_id='child'))
        (original_run/'best_head.safetensors').write_bytes(b'original checkpoint')
        self.app.export_model('child', self.archive)
        with patch.object(self.app, 'snapshot', return_value={'dependencies': {'torch': True}}), \
             patch.object(self.app, 'validate_model', return_value={'status': 'passed'}):
            imported = self.app.import_model_archive(self.archive)
        loaded = read(self.app.store.item('models', imported['id'])/'manifest.json')
        self.assertNotIn('run_id', loaded)
        self.assertEqual(loaded['source_run_id'], 'original_run')
        self.app.set_model_deleted(imported['id'], True)
        self.app.purge_model(imported['id'])
        self.assertEqual((original_run/'best_head.safetensors').read_bytes(), b'original checkpoint')
        self.assertEqual(read(self.app.store.root/'active.json')['model_id'], 'child')


def assert_real_roundtrip(test, store, model_id, image, description=None):
    """Called by the existing small-engine integration tests, not a mocked model."""
    import torch
    from worker import make_engine
    app = App(store.root)
    source = make_engine(store, model_id, 'cpu')
    features = source.features([image], [description]) if description else source.features([image])
    expected = torch.tensor(source.scores(features))
    with tempfile.TemporaryDirectory(dir=store.root.parent) as tmp:
        archive = Path(tmp)/'model.zip'
        app.export_model(model_id, archive)
        target = App(Path(tmp)/'fresh')
        result = target.import_model_archive(archive)
        fresh = make_engine(target.store, result['id'], 'cpu')
        features = fresh.features([image], [description]) if description else fresh.features([image])
        torch.testing.assert_close(torch.tensor(fresh.scores(features)), expected, rtol=0, atol=0)
        test.assertFalse((target.store.root/'active.json').exists())
        test.assertEqual(fresh.spec['selected_epoch'], source.spec['selected_epoch'])

