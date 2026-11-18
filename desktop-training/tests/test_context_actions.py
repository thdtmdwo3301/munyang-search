"""Actual Qt right-click targets and archive integrity; isolated fixture stores."""
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from app_service import App
from core import read, write, sha
from tests.test_core import archive


class DatasetExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent))
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.app=App(self.root/'workspace')
        self.blob=archive(); self.data=self.app.store.import_zip(self.blob,'sample')
        self.key=self.data['id']; self.target=self.root/'export.zip'

    def test_export_is_identical_and_reimportable(self):
        result=self.app.export_dataset(self.key,self.target)
        self.assertEqual(self.target.read_bytes(),self.blob)
        self.assertEqual(result['sha256'],sha(self.target))
        other=App(self.root/'fresh'); imported=other.import_data(self.target,'roundtrip')
        self.assertEqual(imported['counts'],dict(train=1,dev=1,eval=1))
        before=self.app.store.rows([self.key]); after=other.store.rows([imported['id']])
        for a,b in zip(before,after):
            for field in ('id','labels','split','pixel_sha256','description'): self.assertEqual(a[field],b[field])

    def test_explicit_overwrite_and_corrupt_source_preserve_destination(self):
        self.target.write_bytes(b'old')
        with self.assertRaises(ValueError): self.app.export_dataset(self.key,self.target)
        self.assertEqual(self.target.read_bytes(),b'old')
        (self.app.store.item('datasets',self.key)/'original.zip').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'무결성'): self.app.export_dataset(self.key,self.target,True)
        self.assertEqual(self.target.read_bytes(),b'old'); self.assertEqual(list(self.root.glob('*.partial')),[])

    def test_deleted_busy_internal_and_missing_parent_rejected(self):
        with self.assertRaises(ValueError): self.app.export_dataset(self.key,self.app.store.root/'bad.zip')
        with self.assertRaises(ValueError): self.app.export_dataset(self.key,self.root/'missing'/'x.zip')
        with patch.object(self.app,'is_running',return_value=True):
            with self.assertRaisesRegex(ValueError,'실행 중'): self.app.export_dataset(self.key,self.target)
        self.app.set_dataset_deleted(self.key,True)
        with self.assertRaisesRegex(ValueError,'삭제'): self.app.export_dataset(self.key,self.target)


AVAILABLE=importlib.util.find_spec('PySide6') is not None
if AVAILABLE:
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtTest import QTest
    from training_ui import MainWindow, STYLE


@unittest.skipUnless(AVAILABLE,'Native Qt runtime not installed')
class ContextActionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt=QApplication.instance() or QApplication([]); cls.qt.setStyle('Fusion'); cls.qt.setStyleSheet(STYLE)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent))
        self.root=Path(self.tmp.name); self.app=App(self.root/'workspace')
        self.a=self.app.store.import_zip(archive(),'first'); self.b=self.app.store.import_zip(archive(),'second')
        self.model('base'); self.model('candidate'); self.app.activate('base')
        self.window=MainWindow(self.app); self.window.show(); self.qt.processEvents()

    def model(self,key,**extra):
        folder=self.app.store.item('models',key); folder.mkdir()
        write(folder/'manifest.json',dict(id=key,name=key,engine='dino_head_v1',has_head=True,**extra))

    def wait(self):
        deadline=time.monotonic()+10
        while self.window.task is not None and time.monotonic()<deadline:
            self.qt.processEvents(); time.sleep(.01)
        self.qt.processEvents(); self.assertIsNone(self.window.task)

    def tearDown(self):
        self.wait()
        try:
            if self.window.context_menu: self.window.context_menu.close()
        except RuntimeError: pass
        self.window.close(); self.window.deleteLater(); self.qt.processEvents(); self.tmp.cleanup()

    def action(self,text):
        return next(a for a in self.window.context_menu.actions() if a.text()==text)

    def popup_right_click(self,combo,key):
        combo.showPopup(); self.qt.processEvents()
        index=combo.model().index(combo.findData(key),0)
        point=combo.view().visualRect(index).center()
        QTest.mouseClick(combo.view().viewport(),Qt.MouseButton.RightButton,Qt.KeyboardModifier.NoModifier,point)
        self.qt.processEvents(); self.assertTrue(self.window.context_menu.isVisible())
        self.assertTrue(combo.view().isVisible(), 'Right-click must keep the dropdown visible behind the menu')

    def test_escape_closes_only_context_menu_and_list_still_selects(self):
        combo=self.window.simple_dataset
        combo.setCurrentIndex(combo.findData(self.a['id']))
        self.popup_right_click(combo,self.b['id'])
        self.assertEqual(combo.currentData(),self.a['id'])
        QTest.keyClick(self.window.context_menu,Qt.Key.Key_Escape); self.qt.processEvents()
        self.assertTrue(combo.view().isVisible())
        index=combo.model().index(combo.findData(self.b['id']),0)
        QTest.mouseClick(combo.view().viewport(),Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier,combo.view().visualRect(index).center())
        self.qt.processEvents()
        self.assertEqual(combo.currentData(),self.b['id'])
        self.assertFalse(combo.view().isVisible())

    def test_refresh_and_release_context_event_preserve_menu_target_and_dropdown(self):
        combo=self.window.simple_dataset
        combo.setCurrentIndex(combo.findData(self.a['id']))
        self.popup_right_click(combo,self.b['id']); menu=self.window.context_menu
        highlighted=combo.view().currentIndex().row()
        self.window.refresh(); self.qt.processEvents()
        self.assertTrue(menu.isVisible()); self.assertTrue(combo.view().isVisible())
        self.assertEqual(combo.view().currentIndex().row(),highlighted)
        # Windows sends a ContextMenu event after the right mouse release.
        point=combo.view().visualRect(combo.view().currentIndex()).center()
        self.qt.sendEvent(combo.view().viewport(),QContextMenuEvent(
            QContextMenuEvent.Reason.Mouse,point,combo.view().viewport().mapToGlobal(point)))
        self.qt.processEvents(); self.assertIs(self.window.context_menu,menu)
        self.assertIn('second',menu.actions()[0].text())
        self.assertEqual(combo.currentData(),self.a['id'])

    def test_native_menu_command_closes_list_only_when_action_chosen(self):
        combo=self.window.simple_dataset
        self.popup_right_click(combo,self.b['id']); menu=self.window.context_menu
        called=[]
        def save(key): called.append((key,combo.view().isVisible()))
        with patch.object(self.window,'save_dataset_as',side_effect=save):
            QTest.mouseClick(menu,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,
                             menu.actionGeometry(self.action('다른 이름으로 저장…')).center())
            self.qt.processEvents()
        self.assertEqual(called,[(self.b['id'],False)])
        self.assertFalse(combo.view().isVisible())

    def test_keyboard_context_menu_escape_twice_and_reopen(self):
        combo=self.window.base_choice; combo.showPopup(); self.qt.processEvents()
        index=combo.model().index(combo.findData('candidate'),0)
        combo.view().setCurrentIndex(index)
        point=combo.view().visualRect(index).center()
        self.qt.sendEvent(combo.view().viewport(),QContextMenuEvent(
            QContextMenuEvent.Reason.Keyboard,point,combo.view().viewport().mapToGlobal(point)))
        self.qt.processEvents()
        self.assertTrue(combo.view().isVisible()); self.assertTrue(self.window.context_menu.isVisible())
        self.assertIn('candidate',self.window.context_menu.actions()[0].text())
        QTest.keyClick(self.window.context_menu,Qt.Key.Key_Escape); self.qt.processEvents()
        self.assertTrue(combo.view().isVisible())
        QTest.keyClick(combo.view(),Qt.Key.Key_Escape); self.qt.processEvents()
        self.assertFalse(combo.view().isVisible())
        self.popup_right_click(combo,'candidate')

    def test_popup_dataset_acts_on_clicked_row_not_selection_and_exports(self):
        self.window.simple_dataset.setCurrentIndex(self.window.simple_dataset.findData(self.a['id']))
        self.popup_right_click(self.window.simple_dataset,self.b['id'])
        self.assertIn('second',self.window.context_menu.actions()[0].text())
        dest=self.root/'data.zip'
        with patch('training_ui.QFileDialog.getSaveFileName',return_value=(str(dest),'')):
            self.action('다른 이름으로 저장…').trigger(); self.wait()
        self.assertEqual(dest.read_bytes(),(self.app.store.item('datasets',self.b['id'])/'original.zip').read_bytes())
        self.assertFalse(self.app.store.dataset_deleted(self.a['id']))
        self.assertFalse(self.app.store.dataset_deleted(self.b['id']))

    def test_popup_dataset_delete_confirmation_preserves_original_and_other_item(self):
        self.window.simple_dataset.setCurrentIndex(self.window.simple_dataset.findData(self.a['id']))
        self.popup_right_click(self.window.simple_dataset,self.b['id'])
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No): self.action('삭제…').trigger()
        self.assertFalse(self.app.store.dataset_deleted(self.b['id']))
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.action('삭제…').trigger(); self.wait()
        self.assertTrue(self.app.store.dataset_deleted(self.b['id']))
        self.assertFalse(self.app.store.dataset_deleted(self.a['id']))
        self.assertTrue((self.app.store.item('datasets',self.b['id'])/'original.zip').exists())
        self.assertEqual(self.window.simple_dataset.findData(self.b['id']),-1)

    def test_weight_popup_save_target_and_delete_guards(self):
        self.popup_right_click(self.window.base_choice,'candidate')
        with patch.object(self.window,'save_model_as') as save:
            self.action('다른 이름으로 저장…').trigger(); save.assert_called_once_with('candidate')
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.action('삭제…').trigger(); self.wait()
        self.assertTrue(self.app.store.model_deleted('candidate'))
        self.window.show_item_menu('model','base',QPoint(50,50))
        self.assertFalse(self.action('삭제…').isEnabled()); self.assertTrue(self.action('다른 이름으로 저장…').isEnabled())
        self.model('new_parent'); self.model('child',base_model='new_parent'); self.window.refresh()
        self.window.show_item_menu('model','new_parent',QPoint(50,50))
        self.assertFalse(self.action('삭제…').isEnabled())

    def test_result_popup_targets_candidate_keeps_run_and_comparison_targets_row(self):
        run=self.app.store.item('runs','completed'); run.mkdir()
        write(run/'status.json',dict(id='completed',kind='train',status='complete',name='run'))
        write(run/'config.json',dict(model_id='base'))
        write(run/'result.json',dict(model_id='candidate',metrics={}))
        self.window.simple_run_id='completed'; self.window.refresh()
        self.popup_right_click(self.window.simple_run,'completed')
        self.assertIn('학습 기록: run',self.window.context_menu.actions()[0].text())
        with patch.object(self.window,'save_model_as') as save:
            self.action('다른 이름으로 저장…').trigger(); save.assert_called_once_with('candidate')
        pos=self.window.comparison.visualRect(self.window.comparison.model().index(0,0)).center()
        self.window.comparison_menu(pos)
        self.assertIn('base',self.window.context_menu.actions()[0].text())
        self.assertFalse(self.action('삭제…').isEnabled())
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.window.delete_item('model','candidate'); self.wait()
        self.assertTrue((run/'result.json').exists()); self.assertTrue((run/'config.json').exists())
        self.assertIsNone(self.window.row_models[1])

    def completed_run(self,key,model_id='candidate',status='complete'):
        folder=self.app.store.item('runs',key); folder.mkdir()
        write(folder/'status.json',dict(id=key,kind='train',status=status,name=key))
        write(folder/'config.json',dict(model_id='base'))
        write(folder/'result.json',dict(model_id=model_id,metrics={}))
        return folder

    def test_result_list_deletes_record_even_when_weights_already_deleted(self):
        folder=self.completed_run('old_run'); before=(folder/'result.json').read_bytes()
        self.app.set_model_deleted('candidate',True)
        self.window.refresh(); self.popup_right_click(self.window.simple_run,'old_run')
        self.assertFalse(self.action('다른 이름으로 저장…').isEnabled())
        self.assertTrue(self.action('삭제…').isEnabled())
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.action('삭제…').trigger(); self.wait()
        self.assertEqual(self.window.simple_run.findData('old_run'),-1)
        self.assertEqual(self.window.comparison.rowCount(),0)
        self.assertEqual((folder/'result.json').read_bytes(),before)
        self.assertEqual(App(self.app.store.root).snapshot()['runs'],[])

    def test_result_list_delete_does_not_delete_active_model_or_other_run(self):
        self.completed_run('selected'); self.completed_run('clicked')
        self.app.activate('candidate'); self.window.simple_run_id='selected'; self.window.refresh()
        self.popup_right_click(self.window.simple_run,'clicked')
        self.assertTrue(self.action('삭제…').isEnabled())
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):
            self.action('삭제…').trigger()
        self.assertNotEqual(self.window.simple_run.findData('clicked'),-1)
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.action('삭제…').trigger(); self.wait()
        self.assertEqual(self.window.simple_run.findData('clicked'),-1)
        self.assertEqual(self.window.simple_run.currentData(),'selected')
        self.assertFalse(self.app.store.model_deleted('candidate'))
        self.assertEqual(self.app.snapshot()['active']['model_id'],'candidate')

    def test_failed_or_missing_result_still_deletable_but_running_record_not(self):
        folder=self.completed_run('failed_run',status='failed'); (folder/'result.json').unlink()
        self.window.refresh(); self.popup_right_click(self.window.simple_run,'failed_run')
        self.assertFalse(self.action('다른 이름으로 저장…').isEnabled())
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.action('삭제…').trigger(); self.wait()
        self.assertEqual(self.window.simple_run.findData('failed_run'),-1)
        self.completed_run('running',status='training'); self.window.refresh()
        self.window.show_item_menu('run','running',QPoint(20,20))
        self.assertFalse(self.action('삭제…').isEnabled())

    def test_busy_all_actions_disabled_and_closed_combo_menu_available(self):
        self.window.base_choice.customContextMenuRequested.emit(QPoint(10,10)); self.qt.processEvents()
        self.assertTrue(self.window.context_menu.isVisible())
        self.window.task=object()
        try:
            self.window.show_item_menu('dataset',self.a['id'],QPoint(40,40))
            self.assertFalse(self.action('삭제…').isEnabled()); self.assertFalse(self.action('다른 이름으로 저장…').isEnabled())
        finally: self.window.task=None


if __name__=='__main__': unittest.main()
