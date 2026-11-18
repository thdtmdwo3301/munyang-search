"""Local storage routing, verified migration and failure safety."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from app_service import App
from core import Store, read, write, sha
from tests.test_core import archive
import storage_settings


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent))
        self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        self.app=App(self.base/'workspace')
        self.data=self.app.store.import_zip(archive(),'자료')
        model=self.app.store.item('models','base');model.mkdir()
        (model/'weights.bin').write_bytes(b'verified model bytes')
        write(model/'manifest.json',dict(id='base',name='base',has_head=True,weight_hashes={'weights.bin':sha(model/'weights.bin')}))
        self.app.activate('base')
        self.targets=dict(datasets=str(self.base/'new_data'),models=str(self.base/'new_models'))

    def test_move_preserves_ids_bytes_selection_and_history(self):
        root=self.app.store.root
        old_rows=self.app.store.rows([self.data['id']])
        run=self.app.store.item('runs','history');run.mkdir();write(run/'result.json',{'keep':'unaltered'})
        before=(run/'result.json').read_bytes();active=(root/'active.json').read_bytes()
        files={k:{p.relative_to(self.app.store.kind_root(k)).as_posix():sha(p)
                  for p in self.app.store.kind_root(k).rglob('*') if p.is_file()}
               for k in ('datasets','models')}
        result=self.app.change_storage(**self.targets)
        self.assertTrue(result['changed']);self.assertEqual(result['warnings'],[])
        restarted=App(root)
        self.assertEqual(restarted.snapshot()['storage_paths'],self.targets)
        for kind,expected in files.items():
            self.assertFalse((root/kind).exists())
            actual={p.relative_to(restarted.store.kind_root(kind)).as_posix():sha(p)
                    for p in restarted.store.kind_root(kind).rglob('*') if p.is_file()}
            self.assertEqual(actual,expected)
        self.assertEqual((root/'active.json').read_bytes(),active)
        self.assertEqual((run/'result.json').read_bytes(),before)
        self.assertTrue(restarted.store.dataset_image(old_rows[0]).is_file())
        self.assertEqual(restarted.store.rows([self.data['id']])[0]['labels'],old_rows[0]['labels'])
        self.assertFalse(restarted.change_storage(**self.targets)['changed'])
        added=restarted.store.import_zip(archive(),'new import')
        self.assertTrue((Path(self.targets['datasets'])/added['id']/'manifest.json').is_file())

    def test_move_one_root_and_return_to_default(self):
        root=self.app.store.root
        self.app.change_storage(datasets=str(root/'datasets'),models=self.targets['models'])
        self.assertTrue((root/'datasets').is_dir());self.assertFalse((root/'models').exists())
        self.app.change_storage(datasets=str(root/'datasets'),models=str(root/'models'))
        self.assertTrue((root/'models/base/weights.bin').is_file())

    def test_invalid_destinations_leave_sources_intact(self):
        occupied=self.base/'occupied';occupied.mkdir();(occupied/'keep').write_text('untouched')
        for target in (str(occupied),str(self.app.store.root),str(self.app.store.root/'cache/nested'),
                       str(self.app.store.root/'run_trash/nested'),
                       str(self.app.store.kind_root('datasets')/'inside'),self.targets['models'],
                       '/home/mount/NAS_MOUNT/forbidden'):
            with self.assertRaises(ValueError):self.app.change_storage(datasets=target,models=self.targets['models'])
        self.assertEqual((occupied/'keep').read_text(),'untouched')
        self.assertFalse((self.app.store.root/'storage.json').exists())
        link=self.base/'link';link.symlink_to(occupied,target_is_directory=True)
        with self.assertRaises(ValueError):self.app.change_storage(datasets=str(link),models=self.targets['models'])

    def test_failure_before_commit_rolls_back_exact_copies(self):
        original=storage_settings.copy_verified;count=[0]
        def fail(source,target,on_created=None):
            count[0]+=1
            result=original(source,target,on_created)
            if count[0]==2:raise OSError('simulated copy failure')
            return result
        with patch('storage_settings.copy_verified',side_effect=fail):
            with self.assertRaisesRegex(OSError,'simulated'):self.app.change_storage(**self.targets)
        self.assertFalse((self.app.store.root/'storage.json').exists())
        self.assertTrue(self.app.store.item('models','base').is_dir())
        self.assertFalse(Path(self.targets['datasets']).exists())

    def test_busy_and_insufficient_space_block(self):
        run=self.app.store.item('runs','busy');run.mkdir()
        write(run/'status.json',dict(id='busy',kind='train',status='training'))
        with self.assertRaisesRegex(ValueError,'실행 중'):self.app.change_storage(**self.targets)
        write(run/'status.json',dict(id='busy',kind='train',status='complete'))
        with patch('storage_settings.shutil.disk_usage') as usage:
            usage.return_value.free=0
            with self.assertRaisesRegex(ValueError,'여유 공간'):self.app.change_storage(**self.targets)

    def test_missing_configured_storage_never_falls_back(self):
        self.app.change_storage(**self.targets)
        path=Path(self.targets['models']);path.rename(self.base/'disconnected')
        with self.assertRaisesRegex(ValueError,'찾을 수'):Store(self.app.store.root)
        self.assertFalse((self.app.store.root/'models').exists())

    def test_training_rejects_missing_or_invalid_gt_in_existing_manifest(self):
        manifest=self.app.store.item('datasets',self.data['id'])/'manifest.json'
        data=read(manifest)
        for labels in (None,[],['강한'],['강한']*5):
            data['rows'][0]['labels']=labels;write(manifest,data)
            with self.assertRaisesRegex(ValueError,'정답 감성어 5개'):
                self.app.store.training_plan(dict(dataset_ids=[self.data['id']],model_id='base'))

    def test_model_retention_and_export_guards_after_move(self):
        other=self.app.store.item('models','unused');other.mkdir()
        write(other/'manifest.json',dict(id='unused',name='unused',has_head=True))
        (other/'weights.bin').write_bytes(b'old weights')
        self.app.change_storage(**self.targets)
        self.app.set_model_deleted('unused',True)
        result=self.app.purge_model('unused')
        self.assertEqual(result['removed_bytes'],11)
        self.assertTrue(self.app.store.item('models','base').is_dir())
        with self.assertRaisesRegex(ValueError,'내부 보관함'):
            self.app.export_model('base',Path(self.targets['models'])/'bad.zip')


if __name__=='__main__':unittest.main()
