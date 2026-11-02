import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('PySide6') is not None
if AVAILABLE:
    if os.name != 'nt' and not os.environ.get('DISPLAY'):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication, QMessageBox, QLineEdit
    from PySide6.QtCore import Qt
    from desktop import MainWindow, STYLE
    from app_service import App
    from core import write
    from tests.test_core import archive


@unittest.skipUnless(AVAILABLE, 'Native Qt runtime not installed')
class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])
        cls.qt.setStyle('Fusion'); cls.qt.setStyleSheet(STYLE)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent))
        self.service = App(Path(self.tmp.name) / 'workspace')
        self.window = MainWindow(self.service); self.window.show(); self.qt.processEvents()

    def tearDown(self):
        self.wait_task(); self.window.close(); self.window.deleteLater(); self.qt.processEvents(); self.tmp.cleanup()

    def wait_task(self):
        deadline = time.monotonic() + 10
        while self.window.task is not None and time.monotonic() < deadline:
            self.qt.processEvents(); time.sleep(.01)
        self.assertIsNone(self.window.task, 'Background UI task did not complete')

    def test_native_navigation_and_capture(self):
        self.assertEqual(self.window.stack.count(), 5)
        for i in range(5):
            self.window.nav.setCurrentRow(i); self.qt.processEvents()
            self.assertEqual(self.window.stack.currentIndex(), i)
            scroll = self.window.stack.currentWidget()
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            if os.environ.get('PATTERN_UI_CAPTURE') == '1':
                target = Path(__file__).absolute().parent.parent / 'test-output' / f'desktop-{i}.png'
                target.parent.mkdir(exist_ok=True)
                self.assertTrue(self.window.grab().save(str(target)))
        self.assertFalse(self.window.train_button.isEnabled())
        self.assertFalse(self.window.infer_button.isEnabled())

    def test_data_registration_and_plan(self):
        path = Path(self.tmp.name) / 'data.zip'; path.write_bytes(archive())
        self.window.data_name.setText('합성 테스트'); self.window.zip_path.setText(str(path))
        self.window.import_button.click(); self.wait_task()
        self.assertEqual(self.window.datasets.rowCount(), 1)
        self.assertEqual(self.window.dataset_choices.count(), 1)
        self.window.dataset_choices.item(0).setCheckState(Qt.CheckState.Checked)
        model = self.service.store.item('models', 'base'); model.mkdir()
        write(model / 'manifest.json', {'id': 'base', 'name': '테스트 구성', 'has_head': False})
        self.window.refresh(); self.window.plan_button.click(); self.wait_task()
        self.assertIn('학습 1장 / 개발 1장 / 평가 1장', self.window.plan_label.text())
        self.assertEqual(self.window.settings()['batch'], 8)

    def test_registration_error_not_freeze(self):
        self.window.model_folder.setText('../escape'); self.window.inspect_button.click(); self.wait_task()
        self.assertIn('확인 필요', self.window.message.text())
        self.assertTrue(self.window.inspect_button.isEnabled())
        self.assertFalse(self.window.register_button.isEnabled())

    def test_model_delete_and_restore_ui(self):
        for key in ('active','unused'):
            folder=self.service.store.item('models',key);folder.mkdir()
            write(folder/'manifest.json',dict(id=key,name=key,has_head=True,engine='legacy_dino_rbf_v1'))
        self.service.activate('active');self.window.refresh()
        row=next(i for i in range(self.window.models.rowCount()) if self.window.models.item(i,3).text()=='unused')
        self.window.models.selectRow(row)
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):self.window.delete_model_button.click()
        self.assertFalse(self.service.store.model_deleted('unused'))
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.window.delete_model_button.click();self.wait_task()
        self.assertEqual(self.window.models.rowCount(),1)
        self.assertEqual(self.window.train_model.findData('unused'),-1)
        self.assertEqual(self.window.infer_model.findData('unused'),-1)
        self.window.model_trash_toggle.click();self.window.deleted_models.selectRow(0)
        self.window.restore_model_button.click();self.wait_task()
        self.assertEqual(self.window.models.rowCount(),2)
        self.assertGreaterEqual(self.window.train_model.findData('unused'),0)
        self.assertEqual(self.window.deleted_models.rowCount(),0)
        self.assertEqual(self.service.snapshot()['active']['model_id'],'active')

    def test_plan_omits_verbose_provenance_without_modifying_record(self):
        plan=dict(counts=dict(train=1,dev=0,eval=0),epochs=10,base_eval_seen=0,
                  base_provenance='원본 PATTERN_SEARCH_20261007 배포 가중치. 원본 배포 기록은 평가 446장 중 223장의 학습 재사용을 명시함.')
        original=dict(plan)
        self.window.show_plan(plan)
        self.assertNotIn('PATTERN_SEARCH',self.window.plan_label.text())
        self.assertIn('마지막 에폭',self.window.plan_label.text())
        self.assertEqual(plan,original)

    def test_retention_and_permanent_delete_confirmation(self):
        folder=self.service.store.item('models','unused');folder.mkdir()
        write(folder/'manifest.json',dict(id='unused',name='삭제 테스트',has_head=True))
        (folder/'head.safetensors').write_bytes(b'test fixture')
        self.service.set_model_deleted('unused',True);self.window.refresh()
        self.window.retention_enabled.setCurrentIndex(1);self.window.retention_days.setValue(14)
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):self.window.retention_save.click()
        self.assertFalse(self.service.model_retention()['enabled'])
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.window.retention_save.click();self.wait_task()
        self.assertEqual(self.service.model_retention()['days'],14)
        self.assertTrue(self.service.model_retention()['enabled'])
        self.window.model_trash_toggle.click();self.window.deleted_models.selectRow(0)
        self.assertNotIn('없음',self.window.deleted_models.item(0,2).text())
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):self.window.purge_model_button.click()
        self.assertTrue((folder/'head.safetensors').exists())
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.window.purge_model_button.click();self.wait_task()
        self.assertFalse((folder/'head.safetensors').exists())
        self.assertEqual(self.window.deleted_models.rowCount(),0)
        self.assertIn('영구 삭제 완료',self.window.message.text())

    def test_bert5_description_input_switch(self):
        for key,engine in [('image','legacy_dino_rbf_v1'),('text','bert5_fusion_v1')]:
            folder=self.service.store.item('models',key);folder.mkdir()
            write(folder/'manifest.json',dict(id=key,name=key,engine=engine,has_head=True))
        self.window.refresh();self.window.nav.setCurrentRow(3)
        self.window.infer_model.setCurrentIndex(self.window.infer_model.findData('text'))
        self.qt.processEvents();self.assertTrue(self.window.description_panel.isVisible())
        self.assertGreaterEqual(self.window.train_model.findData('text'),0)
        self.window.infer_model.setCurrentIndex(self.window.infer_model.findData('image'))
        self.qt.processEvents();self.assertFalse(self.window.description_panel.isVisible())

    def test_dataset_delete_confirmation_and_restore(self):
        item=self.service.store.import_zip(archive(),'등록 삭제 확인')
        self.window.refresh();self.window.datasets.selectRow(0);self.qt.processEvents()
        self.assertTrue(self.window.delete_data_button.isEnabled())
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):
            self.window.delete_data_button.click()
        self.assertEqual(len(self.service.snapshot()['datasets']),1)
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes):
            self.window.delete_data_button.click();self.wait_task()
        self.assertEqual(self.window.datasets.rowCount(),0)
        self.assertEqual(self.window.dataset_choices.count(),0)
        self.window.trash_toggle.click()
        self.assertTrue(self.window.trash_panel.isVisible())
        self.assertEqual(self.window.deleted_datasets.rowCount(),1)
        self.window.deleted_datasets.selectRow(0);self.window.restore_data_button.click();self.wait_task()
        self.assertEqual(self.window.datasets.rowCount(),1)
        self.assertEqual(self.window.dataset_choices.count(),1)
        self.assertEqual(self.window.deleted_datasets.rowCount(),0)
        self.assertEqual(self.service.snapshot()['datasets'][0]['id'],item['id'])

    def test_single_image_data_registration_and_reset(self):
        import io,zipfile
        source=Path(self.tmp.name)/'data';source.mkdir()
        with zipfile.ZipFile(io.BytesIO(archive())) as z: z.extractall(source)
        import csv
        rows=list(csv.DictReader((source/'labels.csv').read_text().splitlines()))
        self.window.data_name.setText('단일 이미지')
        with patch('desktop.QFileDialog.getOpenFileName',return_value=(str(source/rows[0]['image']),'')):
            self.window.data_file_button.click()
        self.assertTrue(self.window.single_data_panel.isVisible())
        for i in range(5): self.window.data_labels.item(i).setCheckState(Qt.CheckState.Checked)
        self.window.data_split.setCurrentIndex(2)
        self.window.import_button.click();self.wait_task()
        self.assertEqual(self.service.store.list('datasets')[0]['counts'],{'eval':1})
        self.window.zip_path.setText(str(source/rows[1]['image']))
        self.assertEqual(self.window.data_label_count.text(),'선택한 정답: 0 / 5')
        with patch('desktop.QFileDialog.getExistingDirectory',return_value=str(source)):
            self.window.data_folder_button.click()
        self.assertFalse(self.window.single_data_panel.isVisible())
        self.window.import_button.click();self.wait_task()
        self.assertEqual(self.window.datasets.rowCount(),2)

    def test_simple_training_and_optional_advanced(self):
        from core import TRAIN_DEFAULTS
        self.window.nav.setCurrentRow(1); self.qt.processEvents()
        self.assertFalse(self.window.advanced_panel.isVisible())
        self.window.epochs.setValue(99)
        self.assertEqual(self.window.settings()['epochs'],TRAIN_DEFAULTS['epochs'])
        self.window.advanced_button.click(); self.assertTrue(self.window.advanced_panel.isVisible())
        self.window.epochs.setValue(3); self.assertEqual(self.window.settings()['epochs'],3)
        self.window.advanced_button.click()
        for key,value in TRAIN_DEFAULTS.items(): self.assertEqual(self.window.settings()[key],value)

    def test_folder_picker_and_missing_weights(self):
        source=Path(self.tmp.name)/'weights'; source.mkdir(); (source/'image_residual.pt').write_bytes(b'fixture')
        with patch('desktop.QFileDialog.getExistingDirectory',return_value=str(source)):
            self.window.folder_button.click()
        self.window.inspect_button.click(); self.wait_task()
        self.assertIn('dinov3_end_to_end_strict.pt',self.window.model_check_text.text())
        self.assertFalse(self.window.register_button.isEnabled())

    def test_registration_wizard_confirm(self):
        from tests.test_core import model_source
        source=model_source(self.tmp.name)
        self.window.model_folder.setText(str(source)); self.window.inspect_button.click(); self.wait_task()
        self.assertTrue(self.window.register_button.isEnabled())
        with patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.Yes), patch.object(self.service,'import_model',return_value={'id':'fixture'}) as register:
            self.window.register_button.click(); self.wait_task()
        self.assertEqual(register.call_args[0][0]['source'],str(source))
        self.assertIn('등록 완료',self.window.message.text())
        self.assertFalse(self.window.register_button.isEnabled())

    def test_legacy_original_available_for_training(self):
        folder=self.service.store.item('models','legacy'); folder.mkdir()
        write(folder/'manifest.json',dict(id='legacy',name='legacy',has_head=True,engine='legacy_dino_rbf_v1',training_supported=False))
        self.window.refresh()
        self.assertEqual(self.window.train_model.count(),1)
        self.assertEqual(self.window.infer_model.count(),1)

    def test_applied_model_is_next_training_default(self):
        for key in ('original','updated'):
            folder=self.service.store.item('models',key);folder.mkdir()
            write(folder/'manifest.json',dict(id=key,name=key,has_head=True,engine='legacy_dino_rbf_v1'))
        self.service.activate('original');self.window.refresh()
        self.assertEqual(self.window.train_model.currentData(),'original')
        self.service.activate('updated');self.window.refresh()
        self.assertEqual(self.window.train_model.currentData(),'updated')

    def test_train_only_result_has_no_fake_evaluation(self):
        folder=self.service.store.item('runs','trainonly');folder.mkdir()
        write(folder/'status.json',dict(id='trainonly',kind='train',status='complete'))
        metrics={s:dict(sample_count=0,correct_slots=0,possible_slots=0,f1_at_5=None) for s in ('train','dev','eval')}
        metrics['train']=dict(sample_count=1,correct_slots=3,possible_slots=5,f1_at_5=.6)
        write(folder/'result.json',dict(selected_epoch=10,metrics=metrics,training_mode='fixture',benchmark_note='fixture',base_eval_seen=0,base_provenance='fixture',checks={},selection='마지막 에폭'))
        self.window.refresh();self.window.load_result()
        self.assertIn('별도 평가 데이터 없음',self.window.evaluation_summary.text())
        self.assertEqual(self.window.metrics.item(2,3).text(),'—')

    def test_image_preview(self):
        from PIL import Image
        path = Path(self.tmp.name) / 'test.png'; Image.new('RGB', (200, 100), 'red').save(path)
        self.window.image_path.setText(str(path)); self.qt.processEvents()
        self.assertFalse(self.window.preview.pixmap().isNull())
        self.assertEqual(self.window.prediction.rowCount(), 0)

    def test_active_model_selection(self):
        for key in ('old', 'new'):
            folder = self.service.store.item('models', key); folder.mkdir()
            write(folder / 'manifest.json', {'id': key, 'name': key, 'has_head': True})
        self.service.activate('old'); self.window.refresh()
        self.assertEqual(self.window.infer_model.currentData(), 'old')
        self.service.activate('new'); self.window.refresh()
        self.assertEqual(self.window.infer_model.currentData(), 'new')

    def test_evaluation_display_and_export(self):
        # Synthetic UI fixture only, in a temporary store; not a research result.
        folder = self.service.store.item('runs', 'fixture'); folder.mkdir()
        write(folder / 'status.json', {'id': 'fixture', 'kind': 'train', 'status': 'complete', 'name': 'UI test'})
        metrics = {s: {'sample_count': 1, 'correct_slots': 2, 'possible_slots': 5, 'f1_at_5': .4} for s in ('train', 'dev', 'eval')}
        write(folder / 'result.json', {'selected_epoch': 1, 'metrics': metrics, 'training_mode': 'synthetic fixture', 'benchmark_note': 'not research', 'base_eval_seen': 0, 'base_provenance': 'fixture', 'checks': {}})
        raw = b'{"fixture":true}\n'; (folder / 'predictions.jsonl').write_bytes(raw)
        self.window.refresh(); self.window.load_result()
        self.assertEqual(self.window.metrics.item(2, 3).text(), '40.00%')
        self.assertIn('평가 감성어 일치율(F1@5): 40.00%',self.window.evaluation_summary.text())
        self.assertIn('평가 이미지 1장',self.window.evaluation_summary.text())
        destination = Path(self.tmp.name) / 'export.jsonl'
        with patch('desktop.QFileDialog.getSaveFileName', return_value=(str(destination), '')):
            self.window.export_predictions()
        self.assertEqual(destination.read_bytes(), raw)

    def test_inference_does_not_send_gt(self):
        from PIL import Image
        path = Path(self.tmp.name) / 'test.png'; Image.new('RGB', (10, 10)).save(path)
        self.window.image_path.setText(str(path))
        self.window.infer_model.addItem('fixture', 'fixture')
        with patch.object(self.service, 'launch', return_value={'id': 'fixture'}) as launch:
            self.window.start_infer(); self.wait_task()
            kind, payload = launch.call_args[0]
            self.assertEqual(kind, 'infer')
            self.assertEqual(set(payload), {'image_path', 'model_id', 'device'})

    def test_plain_language_prediction_summary(self):
        from core import LABELS
        top=LABELS[:3]+LABELS[5:7]
        result=dict(top5=[dict(label=x,score=1.3176-i*.1) for i,x in enumerate(top)])
        c=dict(status='matched',result=result,labels=LABELS[:5],hits=3,matches=[dict(dataset_name='임시 라벨 표본',split='train')])
        self.window.show_inference_result(c)
        self.assertIn('5개 중 3개 일치',self.window.infer_summary.text())
        self.assertIn('전체 성능이 아닙니다',self.window.infer_summary.text())
        self.assertEqual(self.window.prediction.item(0,2).text(),'일치')
        self.assertEqual(self.window.prediction.item(4,2).text(),'불일치')
        self.assertTrue(self.window.prediction.isColumnHidden(3))
        self.window.score_details_button.click()
        self.assertFalse(self.window.prediction.isColumnHidden(3))
        self.assertEqual(self.window.prediction.item(0,3).text(),'1.3176')
        self.assertIn('확률·정확도가 아닙니다',self.window.score_note.text())
        for status in ('unregistered','conflict'):
            self.window.show_inference_result(dict(c,status=status,labels=[],hits=None))
            self.assertNotIn('3개 일치',self.window.infer_summary.text())
            self.assertEqual(self.window.prediction.item(0,2).text(),'비교 불가')

    def test_confirmed_training_launch(self):
        path = Path(self.tmp.name) / 'data.zip'; path.write_bytes(archive())
        self.service.import_data(path, 'synthetic fixture')
        model = self.service.store.item('models', 'base'); model.mkdir()
        write(model / 'manifest.json', {'id': 'base', 'name': 'fixture', 'has_head': False})
        self.window.refresh(); self.window.dataset_choices.item(0).setCheckState(Qt.CheckState.Checked)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), patch.object(self.service, 'launch', return_value={'id': 'fixture'}) as launch:
            self.window.start_train(); self.wait_task()
            deadline = time.monotonic() + 3
            while not launch.called and time.monotonic() < deadline:
                self.qt.processEvents(); time.sleep(.01)
            self.wait_task()
            self.assertTrue(launch.called)
            self.assertEqual(launch.call_args[0][0], 'train')

    def test_exit_requests_safe_cancel(self):
        folder = self.service.store.item('runs', 'running'); folder.mkdir()
        write(folder / 'status.json', {'id': 'running', 'status': 'training'})
        with patch.object(self.service, 'is_running', return_value=True), patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            self.window.close()
            self.assertTrue(self.window.isVisible())
            self.assertTrue(self.window.pending_close)
            self.assertTrue((folder / 'cancel.request').exists())


if __name__ == '__main__': unittest.main()
