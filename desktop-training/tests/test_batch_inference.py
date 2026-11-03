"""Batch selection, input isolation, partial results and sequential worker tests."""
import tempfile
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from app_service import App
from core import LABELS, read, write, sha

HAS_ENGINE = all(importlib.util.find_spec(x) for x in ('torch', 'transformers', 'safetensors'))


class BatchInferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent.parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.app = App(self.root/'workspace')
        folder = self.app.store.item('models', 'base'); folder.mkdir()
        write(folder/'manifest.json', dict(id='base', engine='dino_head_v1', has_head=True))
        self.images = []
        for i in range(3):
            path = self.root/f'표본_{i}.PNG'; Image.new('RGB', (24, 24), (i*80, 40, 30)).save(path)
            self.images.append(str(path))

    def plan(self, **kwargs):
        payload = dict(model_id='base', images=[dict(image_path=p, labels=LABELS[:5], description='NOT IMAGE INPUT') for p in self.images], **kwargs)
        # Suppress only process spawning; exercise real preparation and snapshot copying.
        with patch.object(self.app, 'snapshot', return_value={'dependencies': {'torch': True}}), \
             patch('app_service.subprocess.Popen', side_effect=OSError('no subprocess')):
            with self.assertRaisesRegex(OSError, 'no subprocess'): self.app.launch('infer', payload)
        record = self.app.store.list('runs')[0]
        folder = self.app.store.item('runs', record['id'])
        record['status'] = 'queued'; write(folder/'status.json', record)
        return folder

    def test_folder_order_dedup_and_no_recursion(self):
        sub = self.root/'sub'; sub.mkdir(); Image.new('RGB', (1, 1)).save(sub/'hidden.png')
        (self.root/'linked.png').symlink_to(sub/'hidden.png')
        selected = self.app.inference_paths(folder=self.root)
        self.assertEqual(selected, self.images)
        self.assertEqual(self.app.inference_paths([self.images[0]]*2), [self.images[0]])
        with self.assertRaisesRegex(ValueError, '바로가기'): self.app.inference_paths([str(self.root/'linked.png')])
        with self.assertRaisesRegex(ValueError, 'NAS'): self.app.inference_paths(['/home/mount/NAS_MOUNT/no-read.png'])
        with self.assertRaisesRegex(ValueError, '1,000'): self.app.inference_paths(self.images*334)
        with self.assertRaises(ValueError): self.app.inference_paths([])

    def test_batch_plan_copies_images_without_gt_or_description(self):
        folder = self.plan(); cfg = read(folder/'config.json')
        self.assertEqual(len(cfg['inputs']), 3)
        for i, item in enumerate(cfg['inputs']):
            self.assertEqual(set(item), {'index', 'file_name', 'file_sha256'})
            self.assertEqual(sha(folder/'inputs'/f'{i:04d}.image'), sha(Path(self.images[i])))
        self.assertNotIn('NOT IMAGE INPUT', (folder/'config.json').read_text())

    def test_corrupt_and_missing_description_rejected_before_launch(self):
        path = self.root/'bad.png'; path.write_bytes(b'not an image')
        with patch.object(self.app, 'snapshot', return_value={'dependencies': {'torch': True}}), patch('app_service.subprocess.Popen') as spawn:
            with self.assertRaises(Exception): self.app.launch('infer', dict(model_id='base', images=[dict(image_path=str(path))]))
            folder = self.app.store.item('models', 'base')
            write(folder/'manifest.json', dict(id='base', engine='bert5_fusion_v1', has_head=True))
            with self.assertRaisesRegex(ValueError, '2번째'):
                self.app.launch('infer', dict(model_id='base', images=[dict(image_path=self.images[0], description='첫 이미지'), dict(image_path=self.images[1])]))
            spawn.assert_not_called()
        self.assertEqual(self.app.store.list('runs'), [])

    def fake_engine(self, fail=False, cancel_folder=None):
        import torch
        class Engine:
            device = torch.device('cpu')
            def features(self, images):
                assert len(images) == 1
                return images[0].getpixel((0, 0))[0]
            def scores(self, feature):
                if fail and feature: raise RuntimeError('fixture failure')
                if cancel_folder: (cancel_folder/'cancel.request').touch()
                return [[float((j+feature)%22)/22 for j in range(22)]]
        return Engine()

    @unittest.skipUnless(HAS_ENGINE, 'ML runtime not installed')
    def test_worker_loads_once_and_image_specific_results_and_gt(self):
        from worker import execute
        folder = self.plan()
        with patch('worker.make_engine', return_value=self.fake_engine()) as load:
            execute(self.app.store.root, folder.name)
            load.assert_called_once()
        result = read(folder/'result.json')
        self.assertEqual(read(folder/'status.json')['status'], 'complete')
        self.assertFalse(result['GT_input']); self.assertFalse(result['description_input'])
        self.assertEqual([i['file_name'] for i in result['items']], [Path(p).name for p in self.images])
        self.assertNotEqual(result['items'][0]['scores'], result['items'][1]['scores'])
        self.app.import_data(self.images[0], '첫 이미지', labels=LABELS[:5], split='train')
        first = self.app.inference_comparison(folder.name, 0)
        other = self.app.inference_comparison(folder.name, 1)
        self.assertEqual(first['status'], 'matched'); self.assertEqual(other['status'], 'unregistered')
        self.assertEqual(other['result']['top5'], result['items'][1]['top5'])
        with self.assertRaises(ValueError): self.app.inference_comparison(folder.name, -1)

    @unittest.skipUnless(HAS_ENGINE, 'ML runtime not installed')
    def test_cancel_keeps_completed_results(self):
        from worker import execute, Cancelled
        folder = self.plan()
        with patch('worker.make_engine', return_value=self.fake_engine(cancel_folder=folder)):
            with self.assertRaises(Cancelled): execute(self.app.store.root, folder.name)
        result = read(folder/'result.json')
        self.assertEqual([i['status'] for i in result['items']], ['complete', 'pending', 'pending'])
        status = read(folder/'status.json'); status['status'] = 'cancelled'; write(folder/'status.json', status)
        self.assertEqual(self.app.inference_comparison(folder.name, 0)['status'], 'unregistered')

    @unittest.skipUnless(HAS_ENGINE, 'ML runtime not installed')
    def test_failure_preserves_completed_and_pending_distinction(self):
        from worker import execute
        folder = self.plan()
        with patch('worker.make_engine', return_value=self.fake_engine(fail=True)):
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'): execute(self.app.store.root, folder.name)
        result = read(folder/'result.json')
        self.assertEqual([i['status'] for i in result['items']], ['complete', 'failed', 'pending'])
        self.assertEqual(result['items'][1]['error'], 'fixture failure')


if __name__ == '__main__': unittest.main()
