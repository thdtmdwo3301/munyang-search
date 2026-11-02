import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

from app_service import App
from core import read, write, sha


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent.parent)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'workspace';self.app=App(self.root)

    def model(self,key,**fields):
        folder=self.app.store.item('models',key);folder.mkdir()
        write(folder/'manifest.json',dict(id=key,name=key,has_head=True,**fields))
        (folder/'weights').mkdir();(folder/'weights/fixture.bin').write_bytes(b'123456789')
        return folder

    def expire(self,key):
        marker=self.root/'model_trash'/(key+'.json');record=read(marker)
        record['purge_after']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat();write(marker,record)

    def test_purge_removes_owned_files_preserves_metrics_and_dataset(self):
        folder=self.model('old',run_id='finished')
        run=self.app.store.item('runs','finished');run.mkdir()
        write(run/'status.json',dict(id='finished',status='complete'))
        write(run/'result.json',dict(model_id='old',metrics={'f1':.3}))
        (run/'best_head.safetensors').write_bytes(b'duplicate')
        original=sha(run/'result.json');manifest_hash=sha(folder/'manifest.json')
        unrelated=self.model('unrelated');self.app.activate('unrelated')
        (self.root/'datasets/keep.txt').write_text('keep')
        self.app.set_model_deleted('old',True)
        result=self.app.purge_model('old')
        self.assertEqual(result['removed_bytes'],18)
        self.assertFalse((folder/'weights').exists());self.assertFalse((run/'best_head.safetensors').exists())
        self.assertEqual(sha(run/'result.json'),original);self.assertEqual(sha(folder/'manifest.json'),manifest_hash)
        self.assertTrue((unrelated/'weights/fixture.bin').is_file());self.assertTrue((self.root/'datasets/keep.txt').is_file())
        self.assertEqual(self.app.snapshot()['deleted_models'],[])
        self.assertEqual(read(self.root/'model_trash/old.json')['status'],'purged')
        with self.assertRaisesRegex(ValueError,'복원할 수 없습니다'):self.app.set_model_deleted('old',False)
        with self.assertRaisesRegex(ValueError,'영구 삭제'):self.app.activate('old')
        self.assertEqual([m['id'] for m in App(self.root).snapshot()['models']],['unrelated'])

    def test_policy_deadlines_migration_restart_restore_and_disable(self):
        self.model('old');self.app.set_model_deleted('old',True)
        self.assertFalse(self.app.model_retention()['enabled'])
        self.assertIsNone(read(self.root/'model_trash/old.json')['purge_after'])
        self.app.set_model_retention(True,7)
        deadline=datetime.fromisoformat(read(self.root/'model_trash/old.json')['purge_after'])
        self.assertGreater(deadline,datetime.now(timezone.utc)+timedelta(days=6))
        self.assertEqual(self.app.cleanup_expired_models()['deleted'],[])
        self.app.set_model_deleted('old',False);self.expire('old')
        self.assertEqual(self.app.cleanup_expired_models()['deleted'],[])
        self.app.set_model_deleted('old',True);self.expire('old')
        app=App(self.root)
        self.assertEqual(len(app.cleanup_expired_models()['deleted']),1)
        self.assertEqual(app.cleanup_expired_models()['deleted'],[])
        self.model('off');self.app.set_model_deleted('off',True)
        self.app.set_model_retention(False,7);self.expire('off')
        self.assertEqual(self.app.cleanup_expired_models()['deleted'],[])
        self.assertTrue((self.root/'models/off/weights/fixture.bin').exists())
        for days in (0,366,-1,True,'7'):
            with self.assertRaises(ValueError):self.app.set_model_retention(True,days)

    def test_dependencies_active_running_and_auto_order(self):
        self.model('base');self.model('child',base_model='base')
        self.model('active');self.app.activate('active')
        with self.assertRaises(ValueError):self.app.purge_model('active')
        self.app.set_model_deleted('child',True);self.app.set_model_deleted('base',True)
        with self.assertRaisesRegex(ValueError,'참조'):self.app.purge_model('base')
        self.app.set_model_retention(True,7);self.expire('base')
        report=self.app.cleanup_expired_models();self.assertEqual(len(report['blocked']),1)
        self.expire('child')
        with patch.object(self.app,'is_running',return_value=True):
            with self.assertRaisesRegex(ValueError,'실행 중'):self.app.purge_model('child')
            self.assertTrue(self.app.cleanup_expired_models()['deferred'])
        report=self.app.cleanup_expired_models()
        self.assertEqual([m['id'] for m in report['deleted']],['child','base'])
        self.assertEqual(report['blocked'],[])
        self.assertTrue((self.root/'models/active/weights/fixture.bin').exists())

    def test_path_escape_and_symlink_refused_before_deletion(self):
        folder=self.model('old');self.app.set_model_deleted('old',True)
        for key in ('../datasets','/home/mount/NAS_MOUNT/blocked',''):
            with self.assertRaises(ValueError):self.app.purge_model(key)
        outside=Path(self.tmp.name)/'outside';outside.mkdir();(outside/'keep').write_text('safe')
        link=folder/'link'
        try:link.symlink_to(outside,target_is_directory=True)
        except OSError:self.skipTest('symbolic links unavailable')
        with self.assertRaisesRegex(ValueError,'바로가기'):self.app.purge_model('old')
        self.assertTrue((folder/'weights/fixture.bin').exists());self.assertEqual((outside/'keep').read_text(),'safe')
        self.assertEqual(read(self.root/'model_trash/old.json')['status'],'deleted')

    def test_failure_mid_delete_retry_not_restore(self):
        folder=self.model('old');(folder/'second.bin').write_bytes(b'other')
        self.app.set_model_deleted('old',True)
        original=Path.unlink;calls=[]
        def fail_second(path,*args,**kwargs):
            calls.append(path)
            if len(calls)==2:raise PermissionError('file in use')
            return original(path,*args,**kwargs)
        with patch.object(Path,'unlink',fail_second):
            with self.assertRaisesRegex(ValueError,'일부 파일'):self.app.purge_model('old')
        self.assertEqual(read(self.root/'model_trash/old.json')['status'],'purging')
        with self.assertRaisesRegex(ValueError,'복원할 수 없습니다'):self.app.set_model_deleted('old',False)
        self.assertEqual(self.app.purge_model('old')['removed_bytes'],14)
        self.assertEqual(list(folder.iterdir()),[folder/'manifest.json'])

    def test_run_checkpoint_wrong_owner_protected(self):
        folder=self.model('old',run_id='run');run=self.app.store.item('runs','run');run.mkdir()
        write(run/'result.json',dict(model_id='another'))
        (run/'best_head.safetensors').write_bytes(b'dont delete')
        self.app.set_model_deleted('old',True)
        with self.assertRaisesRegex(ValueError,'소유'):self.app.purge_model('old')
        self.assertTrue((folder/'weights/fixture.bin').exists());self.assertTrue((run/'best_head.safetensors').exists())
