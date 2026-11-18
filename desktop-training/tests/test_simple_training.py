"""The default training-and-apply screen; isolated fixtures, no research claims."""
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('PySide6') is not None
if AVAILABLE:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton, QSpinBox, QDoubleSpinBox
    from training_ui import MainWindow, STYLE
    from app_service import App
    from core import TRAIN_DEFAULTS, read, write
    from tests.test_core import archive


@unittest.skipUnless(AVAILABLE, 'Native Qt runtime not installed')
class SimpleTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])
        cls.qt.setStyle('Fusion'); cls.qt.setStyleSheet(STYLE)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent))
        self.service = App(Path(self.tmp.name) / 'workspace')
        self.window = MainWindow(self.service); self.window.show(); self.qt.processEvents()

    def wait_task(self):
        deadline = time.monotonic() + 10
        while self.window.task is not None and time.monotonic() < deadline:
            self.qt.processEvents(); time.sleep(.01)
        self.assertIsNone(self.window.task)

    def tearDown(self):
        self.wait_task(); self.window.close(); self.window.deleteLater(); self.qt.processEvents(); self.tmp.cleanup()

    def model(self, key):
        folder = self.service.store.item('models', key); folder.mkdir()
        write(folder / 'manifest.json', dict(id=key, name=key, has_head=True, engine='dino_head_v1'))
        return key

    def data(self):
        source = Path(self.tmp.name) / 'new_patterns.zip'; source.write_bytes(archive())
        with patch('training_ui.QFileDialog.getOpenFileName', return_value=(str(source), '')):
            self.window.simple_file.click(); self.wait_task()
        return self.window.simple_dataset.currentData()

    def completed(self, key='complete', score=.8, n=2, base='base', before=.9):
        model = self.model('model_' + key)
        run = self.service.store.item('runs', key); run.mkdir()
        write(run / 'status.json', dict(id=key, kind='train', name=key, status='complete', progress=100))
        result = dict(model_id=model, selected_epoch=3, metrics={'eval': dict(sample_count=n,
            f1_at_5=score if n else None, baseline=dict(f1_at_5=before if n else None))})
        write(run / 'result.json', result)
        write(run / 'config.json', dict(model_id=base))
        self.window.simple_run_id = key; self.window.refresh()
        return model

    def test_one_screen_without_management_or_hyperparameters(self):
        self.assertFalse(hasattr(self.window, 'nav'))
        self.assertFalse(hasattr(self.window, 'management_button'))
        self.assertFalse(hasattr(self.window, 'advanced_enabled'))
        self.assertEqual(self.window.findChildren(QSpinBox), [])
        self.assertEqual(self.window.findChildren(QDoubleSpinBox), [])
        self.assertEqual(self.window.scroll.horizontalScrollBar().maximum(), 0)
        self.assertEqual(self.window.scroll.verticalScrollBar().maximum(), 0)
        visible = [w.text() for w in self.window.findChildren(QPushButton) if w.isVisible()]
        self.assertIn('학습 시작', visible); self.assertIn('선택 가중치 다른 이름으로 저장…', visible)
        self.assertFalse(any('설정' in t or '추론' in t or '보관함' in t for t in visible))
        self.assertFalse(self.window.simple_start.isEnabled()); self.assertFalse(self.window.simple_apply.isEnabled())
        self.assertFalse(self.window.simple_save.isEnabled())

    def test_import_requires_labels_and_cancel_has_no_side_effect(self):
        with patch('training_ui.QFileDialog.getOpenFileName', return_value=('', '')):
            self.window.simple_file.click()
        self.assertEqual(self.service.store.list('datasets'), [])
        key = self.data()
        self.assertTrue(key); self.assertIn('학습 1장 · 검증 1장 · 평가 1장', self.window.simple_data_note.text())
        with patch('training_ui.QFileDialog.getOpenFileName', return_value=(str(Path(self.tmp.name) / 'image.png'), '')):
            self.window.simple_file.click(); self.wait_task()
        self.assertEqual(len(self.service.store.list('datasets')), 1)
        self.assertIn('확인 필요', self.window.message.text())
        self.assertEqual(self.window.simple_dataset.currentData(), key)

    def test_start_fixed_settings_and_selected_base_preserve_splits(self):
        base = self.model('base'); self.service.activate(base); key = self.data()
        other = self.model('other'); self.window.refresh()
        self.window.base_choice.setCurrentIndex(self.window.base_choice.findData(other))
        self.assertTrue(self.window.simple_start.isEnabled())
        self.assertFalse(hasattr(self.window, 'epochs')); self.assertFalse(hasattr(self.window, 'batch'))
        self.assertFalse(hasattr(self.window, 'lr'))
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes), \
             patch.object(self.service, 'launch', return_value={'id': 'new_run'}) as launch:
            self.window.simple_start.click(); self.wait_task()
            deadline = time.monotonic() + 4
            while not launch.called and time.monotonic() < deadline:
                self.qt.processEvents(); time.sleep(.01)
            self.wait_task()
            self.assertTrue(launch.called)
            kind, payload = launch.call_args[0]
            self.assertEqual(kind, 'train'); self.assertEqual(payload['model_id'], other)
            self.assertEqual(payload['dataset_ids'], [key]); self.assertEqual(payload['device'], 'cpu')
            self.assertEqual({k: payload[k] for k in TRAIN_DEFAULTS}, dict(epochs=10, batch=8, lr=.0001))
        self.assertEqual([r['split'] for r in self.service.store.rows([key])], ['train','dev','eval'])
        self.assertEqual(read(self.service.store.root / 'active.json')['model_id'], base)

    def test_compare_select_apply_and_revert_are_explicit(self):
        base = self.model('base'); self.service.activate(base)
        candidate = self.completed()
        self.assertEqual(self.window.comparison.item(0, 1).text(), '90.00%')
        self.assertEqual(self.window.comparison.item(1, 1).text(), '80.00%')
        self.assertEqual(self.window.comparison.item(1, 2).text(), '-10.00%p')
        self.assertIn('기존 가중치가 더 높습니다', self.window.simple_result_note.text())
        self.assertEqual(self.window.selected_weight, base)
        self.assertFalse(self.window.simple_apply.isEnabled())
        self.window.comparison.selectRow(1)
        self.assertEqual(self.window.selected_weight, candidate)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
            self.window.simple_apply.click()
        self.assertEqual(read(self.service.store.root / 'active.json')['model_id'], base)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            self.window.simple_apply.click(); self.wait_task()
        self.assertEqual(read(self.service.store.root / 'active.json')['model_id'], candidate)
        self.assertTrue((self.service.store.item('models', base) / 'manifest.json').is_file())
        self.assertFalse(self.window.simple_apply.isEnabled()); self.assertEqual(self.window.simple_apply.text(), '현재 적용 중')
        self.assertEqual(App(self.service.store.root).snapshot()['active']['model_id'], candidate)
        self.assertEqual(self.window.base_choice.currentData(), candidate)
        self.window.comparison.selectRow(0)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            self.window.simple_apply.click(); self.wait_task()
        self.assertEqual(read(self.service.store.root / 'active.json')['model_id'], base)

    def test_comparison_uses_run_base_not_current_active_or_manifest_score(self):
        base = self.model('base'); other = self.model('different_dataset')
        self.service.activate(other)
        candidate = self.completed(score=.94)
        self.assertEqual(self.window.row_models, [base, candidate])
        self.assertEqual(self.window.comparison.item(0,1).text(), '90.00%')
        self.assertEqual(self.window.comparison.item(1,2).text(), '+4.00%p')
        self.assertIn('새 가중치가 더 높습니다', self.window.simple_result_note.text())
        self.assertEqual(self.service.snapshot()['active']['model_id'], other)
        self.window.comparison.selectRow(1); self.window.refresh()
        self.assertEqual(self.window.selected_weight, candidate)

    def test_equal_missing_baseline_and_no_eval(self):
        self.model('base'); self.completed('equal', score=.9)
        self.assertIn('성능이 같습니다', self.window.simple_result_note.text())
        self.completed('missing', before=None)
        self.assertEqual(self.window.comparison.item(0,1).text(), '—')
        self.assertIn('성능 차이는 확인할 수 없습니다', self.window.simple_result_note.text())
        self.completed('no_eval', n=0)
        self.assertIn('별도 평가 데이터 없음', self.window.simple_result_note.text())
        self.assertEqual(self.window.comparison.item(1,1).text(), '—')

    def test_failed_running_and_missing_result_never_use_stale_candidate(self):
        self.model('base'); self.completed('old')
        run = self.service.store.item('runs', 'failed'); run.mkdir()
        write(run/'status.json', dict(id='failed', kind='train', name='failed', status='failed', message='fixture error'))
        self.window.simple_run_id = 'failed'; self.window.refresh()
        self.assertIsNone(self.window.simple_result); self.assertFalse(self.window.simple_apply.isEnabled())
        self.assertFalse(self.window.simple_save.isEnabled()); self.assertEqual(self.window.comparison.rowCount(), 0)
        self.assertIn('완료되지 않아', self.window.simple_result_note.text())
        write(run/'status.json', dict(id='failed', kind='train', name='running', status='training', progress=20, eta_seconds=240))
        self.window.refresh()
        self.assertFalse(self.window.simple_start.isEnabled()); self.assertFalse(self.window.simple_save.isEnabled())
        self.assertIn('4분', self.window.simple_status.text())
        self.window.simple_cancel.click(); self.assertTrue((run/'cancel.request').exists())
        with self.assertRaisesRegex(ValueError, '실행 중'): self.service.activate('model_old')
        write(run/'status.json', dict(id='failed', kind='train', status='complete'))
        self.window.refresh()
        self.assertFalse(self.window.simple_save.isEnabled()); self.assertIn('파일이 없어', self.window.simple_result_note.text())

    def test_deleted_candidate_unselectable_old_weight_preserved(self):
        base = self.model('base'); self.service.activate(base); candidate = self.completed()
        self.service.set_model_deleted(candidate, True); self.window.refresh()
        self.assertIsNone(self.window.simple_result)
        self.assertEqual(self.window.row_models, [base, None])
        self.assertEqual(self.window.selected_weight, base)
        self.assertFalse(self.window.simple_apply.isEnabled()); self.assertTrue(self.window.simple_save.isEnabled())

    def test_save_chosen_model_not_active_includes_explicit_overwrite(self):
        base = self.model('base'); self.service.activate(base); candidate = self.completed()
        self.window.comparison.selectRow(1)
        target = Path(self.tmp.name) / 'chosen.zip'; target.write_bytes(b'previous')
        with patch('training_ui.QFileDialog.getSaveFileName', return_value=(str(target), '')), \
             patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes) as question, \
             patch.object(self.service, 'export_model', return_value={'bytes':10,'path':str(target)}) as export:
            self.window.simple_save.click(); self.wait_task()
            export.assert_called_once_with(candidate, str(target), True)
            self.assertIn('교체', question.call_args.args[2])
        self.assertEqual(self.service.snapshot()['active']['model_id'], base)

    def test_save_picker_cancel_confirmation_cancel_and_suffix(self):
        self.model('base'); self.completed()
        with patch.object(self.service, 'export_model', return_value={'bytes':10,'path':'ok'}) as export:
            with patch('training_ui.QFileDialog.getSaveFileName', return_value=('', '')):
                self.window.simple_save.click()
            export.assert_not_called()
            target = Path(self.tmp.name) / 'new_weights'
            with patch('training_ui.QFileDialog.getSaveFileName', return_value=(str(target), '')), \
                 patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.No):
                self.window.simple_save.click()
            export.assert_not_called()
            with patch('training_ui.QFileDialog.getSaveFileName', return_value=(str(target), '')), \
                 patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
                self.window.simple_save.click(); self.wait_task()
            export.assert_called_once_with('base', str(target)+'.zip', False)

    def test_busy_disables_selection_export_and_import(self):
        self.model('base'); self.completed()
        self.window.task = object()
        try:
            self.window.update_simple_training()
            for widget in (self.window.simple_start, self.window.simple_save, self.window.simple_apply,
                           self.window.base_choice, self.window.comparison, self.window.simple_file):
                self.assertFalse(widget.isEnabled())
            with patch.object(self.service, 'export_model') as export:
                self.window.save_model_as(); export.assert_not_called()
        finally: self.window.task = None


if __name__ == '__main__': unittest.main()
