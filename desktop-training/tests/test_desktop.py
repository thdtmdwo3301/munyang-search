"""Native single-screen UI checks; engine/storage/archive regressions live separately."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec('PySide6') is not None
if AVAILABLE:
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication, QGroupBox, QComboBox, QMessageBox
    from training_ui import MainWindow, STYLE
    from app_service import App
    from core import write, read


@unittest.skipUnless(AVAILABLE, 'Native Qt runtime not installed')
class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])
        cls.qt.setStyle('Fusion'); cls.qt.setStyleSheet(STYLE)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent))
        self.service = App(Path(self.tmp.name)/'workspace')
        self.window = MainWindow(self.service); self.window.show(); self.qt.processEvents()

    def tearDown(self):
        self.window.close(); self.window.deleteLater(); self.qt.processEvents(); self.tmp.cleanup()

    def test_only_three_sections_no_alternate_management_mode(self):
        self.assertEqual([w.title() for w in self.window.findChildren(QGroupBox)],
            ['1. 데이터 추가', '2. 기존 가중치에서 학습', '3. 성능 비교 · 가중치 선택 · 저장'])
        self.assertFalse(hasattr(self.window, 'toggle_management'))
        self.assertFalse(hasattr(self.window, 'build_inference'))
        self.assertFalse(hasattr(self.window, 'toggle_advanced'))
        self.assertEqual(self.window.menuBar().actions(), [])

    def test_small_window_scroll_not_clipped_controls(self):
        self.window.resize(820,660); self.qt.processEvents()
        self.assertEqual(self.window.scroll.horizontalScrollBar().maximum(),0)
        self.window.scroll.ensureWidgetVisible(self.window.simple_save)
        self.qt.processEvents()
        self.assertTrue(self.window.simple_save.visibleRegion().isEmpty() is False)

    def test_no_engine_environment_is_clear_and_training_blocked(self):
        with patch.object(self.service, 'snapshot', return_value=dict(datasets=[], models=[], runs=[],
                   dependencies={'torch':False}, active=None)):
            self.window.refresh()
        self.assertIn('설치가 필요', self.window.simple_status.text())
        self.assertFalse(self.window.simple_start.isEnabled())

    def test_stale_process_error_and_existing_storage_preserved(self):
        before = dict(self.service.store.locations)
        run = self.service.store.item('runs','interrupted'); run.mkdir()
        write(run/'status.json',dict(id='interrupted',kind='train',name='중단된 기록',status='interrupted',message='중단됨'))
        self.window.refresh()
        self.assertIn('중단', self.window.simple_status.text())
        self.assertFalse(self.window.simple_save.isEnabled())
        self.assertEqual(self.service.store.locations,before)
        self.assertEqual(read(run/'status.json')['status'],'interrupted')

    def test_close_during_file_work_and_training_is_not_forced(self):
        self.window.task=object()
        self.window.close(); self.assertTrue(self.window.isVisible()); self.window.task=None
        with patch.object(self.service,'is_running',return_value=True), \
             patch.object(QMessageBox,'question',return_value=QMessageBox.StandardButton.No):
            self.window.close(); self.assertTrue(self.window.isVisible())
        self.assertFalse(self.window.pending_close)


if __name__=='__main__': unittest.main()
