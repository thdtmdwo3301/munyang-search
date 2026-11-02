import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app_service import App, WorkspaceLock
from core import write, read, sha
from tests.test_core import archive, model_source


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'workspace'
        self.app = App(self.root)

    def test_single_instance_lock(self):
        lock = WorkspaceLock(self.root)
        try:
            with self.assertRaises(RuntimeError): WorkspaceLock(self.root)
        finally: lock.close()
        second = WorkspaceLock(self.root); second.close()

    def test_import_without_http(self):
        path = Path(self.tmp.name) / 'data.zip'; path.write_bytes(archive())
        result = self.app.import_data(path, '합성 검증 자료')
        self.assertEqual(result['count'], 3)
        self.assertEqual(len(self.app.snapshot()['datasets']), 1)

    def test_dataset_delete_restore_preserves_history_and_files(self):
        dataset=self.app.store.import_zip(archive(),'잘못 등록한 자료');key=dataset['id']
        folder=self.app.store.item('datasets',key)
        hashes={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}
        run=self.app.store.item('runs','completed');run.mkdir()
        write(run/'status.json',dict(id='completed',status='complete',kind='train'))
        write(run/'config.json',dict(dataset_ids=[key],rows=self.app.store.rows([key])))
        before=sha(run/'config.json')
        model=self.app.store.item('models','saved');model.mkdir()
        write(model/'manifest.json',dict(id='saved',name='완료 가중치',has_head=True))
        model_before=sha(model/'manifest.json')
        self.app.set_dataset_deleted(key,True)
        self.assertEqual(self.app.snapshot()['datasets'],[])
        self.assertEqual(self.app.snapshot()['deleted_datasets'][0]['id'],key)
        with self.assertRaisesRegex(ValueError,'복원'):self.app.store.rows([key])
        self.assertEqual(hashes,{str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()})
        self.assertEqual(before,sha(run/'config.json'));self.assertEqual(model_before,sha(model/'manifest.json'))
        restarted=App(self.root)
        self.assertEqual(restarted.snapshot()['datasets'],[])
        restarted.set_dataset_deleted(key,False)
        self.assertEqual(len(restarted.store.rows([key])),3)
        self.assertEqual(restarted.snapshot()['deleted_datasets'],[])
        events=read(self.root/'dataset_trash'/(key+'.json'))['events']
        self.assertEqual([e['action'] for e in events],['deleted','restored'])

    def test_dataset_delete_blocks_running_and_bad_targets(self):
        dataset=self.app.store.import_zip(archive(),'보호');key=dataset['id']
        run=self.app.store.item('runs','busy');run.mkdir()
        write(run/'status.json',dict(id='busy',status='training'))
        with self.assertRaisesRegex(ValueError,'실행 중'):self.app.set_dataset_deleted(key,True)
        write(run/'status.json',dict(id='busy',status='complete'))
        with patch.object(self.app,'is_running',return_value=True):
            with self.assertRaisesRegex(ValueError,'실행 중'):self.app.set_dataset_deleted(key,True)
        for bad in ('../models/saved','/home/mount/NAS_MOUNT/forbidden'):
            with self.assertRaises(ValueError):self.app.set_dataset_deleted(bad,True)
        self.assertEqual(len(self.app.snapshot()['datasets']),1)

    def model_fixture(self,key,**extra):
        folder=self.app.store.item('models',key);folder.mkdir()
        write(folder/'manifest.json',dict(id=key,name=key,has_head=True,engine='legacy_dino_rbf_v1',**extra))
        (folder/'fixture.weight').write_bytes(b'immutable weight fixture')
        return folder

    def test_model_delete_restore_preserves_files_and_choices(self):
        active=self.model_fixture('active');candidate=self.model_fixture('candidate')
        self.app.activate('active')
        before={p.name:sha(p) for p in candidate.iterdir()}
        run=self.app.store.item('runs','finished');run.mkdir()
        write(run/'status.json',dict(id='finished',status='complete',kind='train'))
        write(run/'result.json',dict(model_id='candidate',metrics='retained fixture'))
        result_hash=sha(run/'result.json')
        self.app.set_model_deleted('candidate',True)
        self.assertEqual([m['id'] for m in self.app.snapshot()['models']],['active'])
        self.assertEqual(self.app.snapshot()['deleted_models'][0]['id'],'candidate')
        with self.assertRaisesRegex(ValueError,'복원'):self.app.activate('candidate')
        d=self.app.store.import_zip(archive(),'data')
        with self.assertRaisesRegex(ValueError,'복원'):self.app.store.training_plan(dict(dataset_ids=[d['id']],model_id='candidate'))
        with patch.object(self.app,'snapshot',return_value={'dependencies':{'torch':True}}):
            with self.assertRaisesRegex(ValueError,'복원'):self.app.launch('infer',dict(model_id='candidate',image_path='not_read.png'))
        app=App(self.root);self.assertTrue(app.store.model_deleted('candidate'))
        app.set_model_deleted('candidate',False)
        self.assertEqual(app.snapshot()['active']['model_id'],'active')
        self.assertEqual(before,{p.name:sha(p) for p in candidate.iterdir()})
        self.assertEqual(result_hash,sha(run/'result.json'))
        self.assertEqual([e['action'] for e in read(self.root/'model_trash/candidate.json')['events']],['deleted','restored'])

    def test_model_delete_active_dependency_and_running_guards(self):
        self.model_fixture('active');self.app.activate('active')
        self.model_fixture('base')
        self.model_fixture('parent',base_model='base')
        self.model_fixture('child',base_model='base',parent_model='parent')
        with self.assertRaisesRegex(ValueError,'현재 적용'):self.app.set_model_deleted('active',True)
        with self.assertRaisesRegex(ValueError,'참조'):self.app.set_model_deleted('base',True)
        self.app.set_model_deleted('parent',True)  # lineage is not a runtime dependency
        self.app.store.check_model_available('child')
        with patch.object(self.app,'is_running',return_value=True):
            with self.assertRaisesRegex(ValueError,'실행 중'):self.app.set_model_deleted('child',True)
        self.app.set_model_deleted('child',True);self.app.set_model_deleted('base',True)
        with self.assertRaisesRegex(ValueError,'복원'):self.app.set_model_deleted('child',False)
        self.assertTrue(self.app.store.model_deleted('child'))
        self.app.set_model_deleted('base',False);self.app.set_model_deleted('child',False)
        self.app.store.check_model_available('child')
        self.model_fixture('dino_child',backbone_model='base')
        with self.assertRaisesRegex(ValueError,'참조'):self.app.set_model_deleted('base',True)
        for bad in ('../datasets/other','/home/mount/NAS_MOUNT/blocked'):
            with self.assertRaises(ValueError):self.app.set_model_deleted(bad,True)

    def test_post_inference_comparison_and_conflicting_gt(self):
        import io,zipfile
        from core import LABELS
        blob=archive();dataset=self.app.store.import_zip(blob,'사람 검토 필요 · 테스트')
        folder=self.app.store.item('runs','comparison');folder.mkdir()
        with zipfile.ZipFile(io.BytesIO(blob)) as z: (folder/'input.image').write_bytes(z.read('images/0.png'))
        write(folder/'status.json',dict(status='complete'))
        prediction=LABELS[:3]+LABELS[5:7]
        result=dict(kind='infer',top5=[dict(label=x,score=1.3-i*.1) for i,x in enumerate(prediction)],GT_input=False)
        write(folder/'result.json',result);before=(folder/'result.json').read_bytes()
        c=self.app.inference_comparison('comparison')
        self.assertEqual(c['status'],'matched');self.assertEqual(c['hits'],3)
        self.assertEqual((folder/'result.json').read_bytes(),before)
        self.assertFalse((folder/'config.json').exists())
        # Same filename/ID but different pixels must not retrieve a label.
        from PIL import Image
        Image.new('RGB',(12,12),(210,210,210)).save(folder/'input.image',format='PNG')
        self.assertEqual(self.app.inference_comparison('comparison')['status'],'unregistered')
        with zipfile.ZipFile(io.BytesIO(blob)) as z: (folder/'input.image').write_bytes(z.read('images/0.png'))
        row=dict(id='different_id',image='images/0.png',labels=';'.join(LABELS[5:10]),split='eval',description='')
        self.app.store.import_zip(archive(rows=[row]),'불일치 정답')
        c=self.app.inference_comparison('comparison')
        self.assertEqual(c['status'],'conflict');self.assertIsNone(c['hits'])
        self.assertEqual((folder/'result.json').read_bytes(),before)

    def test_comparison_requires_finished_inference(self):
        folder=self.app.store.item('runs','pending');folder.mkdir()
        write(folder/'status.json',dict(status='loading'))
        with self.assertRaisesRegex(ValueError,'완료 후'): self.app.inference_comparison('pending')

    def test_missing_engine_fails_honestly(self):
        with patch.object(self.app, 'snapshot', return_value={'dependencies': {'torch': False}}):
            with self.assertRaisesRegex(ValueError, '학습 실행 환경'):
                self.app.launch('train', {})
        self.assertEqual(self.app.store.list('runs'), [])

    def test_failed_spawn_recorded(self):
        model = self.app.store.item('models', 'base'); model.mkdir()
        write(model / 'manifest.json', {'id': 'base', 'has_head': True})
        import io
        from PIL import Image
        image = Path(self.tmp.name) / 'test.png'; Image.new('RGB', (10, 10)).save(image)
        with patch.object(self.app, 'snapshot', return_value={'dependencies': {'torch': True}}), patch('app_service.subprocess.Popen', side_effect=OSError('test spawn failure')):
            with self.assertRaises(OSError): self.app.launch('infer', {'model_id': 'base', 'image_path': str(image)})
        self.assertEqual(self.app.store.list('runs')[0]['status'], 'failed')

    def test_cancel_and_stale_state(self):
        folder = self.app.store.item('runs', 'example'); folder.mkdir()
        write(folder / 'status.json', {'id': 'example', 'status': 'training'})
        self.app.cancel('example'); self.assertTrue((folder / 'cancel.request').exists())
        App(self.root)
        self.assertEqual(read(folder / 'status.json')['status'], 'interrupted')

    def test_activate_preserves_model(self):
        folder = self.app.store.item('models', 'base'); folder.mkdir()
        manifest = {'id': 'base', 'has_head': True}; write(folder / 'manifest.json', manifest)
        self.app.activate('base')
        self.assertEqual(read(folder / 'manifest.json'), manifest)
        self.assertEqual(read(self.root / 'active.json')['model_id'], 'base')

    def test_folder_import_verified_before_registration(self):
        source=model_source(self.tmp.name,True); checked=self.app.inspect_model(source)
        with patch.object(self.app,'snapshot',return_value={'dependencies':{'torch':True}}), patch.object(self.app,'validate_model',return_value={'status':'passed','synthetic_image_only':True}) as probe:
            result=self.app.import_model(checked)
        self.assertTrue(probe.called); self.assertTrue(result['training_supported'])
        self.assertEqual(result['load_check']['status'],'passed')
        self.assertTrue(source.exists()); self.assertFalse((self.root/'active.json').exists())
        from core import sha
        package=self.app.store.item('models',result['id'])
        for rel,digest in result['weight_hashes'].items(): self.assertEqual(sha(package/rel),digest)

    def test_model_import_failure_not_registered(self):
        source=model_source(self.tmp.name); checked=self.app.inspect_model(source)
        with patch.object(self.app,'snapshot',return_value={'dependencies':{'torch':True}}), patch.object(self.app,'validate_model',side_effect=ValueError('fixture shape mismatch')):
            with self.assertRaises(ValueError): self.app.import_model(checked)
        self.assertEqual(self.app.store.list('models'),[])
        self.assertTrue(source.exists())

    def test_changed_source_requires_recheck(self):
        source=model_source(self.tmp.name); checked=self.app.inspect_model(source)
        (source/'backbone/model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'변경'): self.app.import_model(checked)
        self.assertEqual(self.app.store.list('models'),[])

    def bundle_fixture(self):
        folder=self.app.store.item('models','bundled'); folder.mkdir()
        (folder/'fixture.pt').write_bytes(b'fixture, not a real weight')
        spec=dict(id='bundled',name='fixture',has_head=True,load_check={'status':'passed'},weight_hashes={'fixture.pt':sha(folder/'fixture.pt')})
        write(folder/'manifest.json',spec)
        write(self.root.parent/'default_model.json',dict(model_id='bundled',manifest_sha256=sha(folder/'manifest.json')))
        return folder

    def test_bundled_default_selected_once(self):
        self.bundle_fixture(); app=App(self.root)
        self.assertEqual(app.snapshot()['active']['model_id'],'bundled')
        history=(self.root/'activation_history.jsonl').read_text()
        App(self.root)
        self.assertEqual((self.root/'activation_history.jsonl').read_text(),history)

    def test_bundled_default_preserves_user_selection(self):
        self.bundle_fixture(); write(self.root/'active.json',dict(model_id='user_model'))
        App(self.root)
        self.assertEqual(read(self.root/'active.json')['model_id'],'user_model')

    def test_partial_bundle_is_not_activated(self):
        folder=self.bundle_fixture(); (folder/'fixture.pt').unlink()
        with self.assertRaisesRegex(ValueError,'부족'): App(self.root)
        self.assertFalse((self.root/'active.json').exists())

    def test_changed_bundle_manifest_is_not_activated(self):
        folder=self.bundle_fixture(); write(folder/'manifest.json',dict(id='bundled',has_head=True))
        with self.assertRaisesRegex(ValueError,'변경'): App(self.root)
        self.assertFalse((self.root/'active.json').exists())


if __name__ == '__main__': unittest.main()
