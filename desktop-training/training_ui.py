"""Native Qt Widgets UI. No HTML, WebView, HTTP listener or browser."""
import sys
import traceback
import time
import re
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QEvent
from PySide6.QtGui import QFont, QPalette, QColor, QContextMenuEvent
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea,
    QGroupBox, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QFileDialog,
    QMessageBox, QProgressBar, QMenu)
from app_service import App, WorkspaceLock, ROOT, RUNNING
from core import TRAIN_DEFAULTS, local_only, read, supports_training

STATE_NAMES = dict(queued='준비', loading='모델 로딩', extracting='특징 추출', training='학습',
                   evaluating='평가', saving='저장', complete='완료', failed='오류', cancelled='중단', interrupted='중단됨')
STYLE = '''
QWidget { background: #f5f7fa; color: #25354a; font-size: 14px; }
QMainWindow { background: #f5f7fa; }
QLabel { background: transparent; }
QLabel#title { font-size: 26px; font-weight: 600; }
QLabel#brand { font-size: 21px; font-weight: 600; }
QLabel#hint { color: #637389; }
QGroupBox { background: white; border: 1px solid #d8e0e9; border-radius: 5px; margin-top: 18px; padding: 20px 15px 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 6px; font-weight: 600; }
QGroupBox#simpleGroup { margin-top: 10px; padding: 8px 12px 6px; }
QPushButton { background: white; border: 1px solid #bccbda; border-radius: 4px; padding: 8px 17px; }
QPushButton:hover { background: #edf3fa; border-color: #356da3; }
QPushButton[primary="true"] { background: #285f98; color: white; border-color: #285f98; }
QPushButton:disabled { background: #edf0f4; color: #738093; border-color: #d8e0e9; }
QComboBox { background: white; color: #25354a; border: 1px solid #bccbda; border-radius: 3px; padding: 6px; min-height: 22px; }
QComboBox QAbstractItemView { background: white; color: #25354a; selection-background-color: #dceafa; selection-color: #173b62; }
QListWidget, QTableWidget, QPlainTextEdit { background: white; color: #25354a; border: 1px solid #d8e0e9; selection-background-color: #e1ecfa; selection-color: #173b62; }
QListWidget#nav { background: #f5f7fa; border: none; font-size: 16px; }
QListWidget#nav::item { padding: 17px 12px; }
QListWidget#nav::item:selected { background: #e1ecfa; color: #225c98; }
QHeaderView::section { background: #f2f5f9; color: #53657b; padding: 9px; border: none; border-bottom: 1px solid #d8e0e9; }
QProgressBar { background: #e5ebf2; border: none; border-radius: 4px; text-align: center; min-height: 22px; }
QProgressBar::chunk { background: #8ab0d4; border-radius: 4px; }
QScrollArea { border: none; }
QMenu { background: white; color: #25354a; border: 1px solid #bccbda; padding: 4px; }
QMenu::item { padding: 8px 20px; }
QMenu::item:selected { background: #e1ecfa; }
QMenu::item:disabled { color: #8994a3; }
'''


def label(text='', hint=False):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(True)
    if hint: widget.setObjectName('hint')
    return widget


def button(text, callback, primary=False):
    widget = QPushButton(text)
    widget.setProperty('primary', primary)
    widget.clicked.connect(callback)
    return widget


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    widget.verticalHeader().hide()
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    widget.setMinimumHeight(180)
    return widget


def populate(widget, rows):
    widget.setRowCount(len(rows))
    for i, row in enumerate(rows):
        for j, value in enumerate(row): widget.setItem(i, j, QTableWidgetItem(str(value)))


def percent(value):
    return '—' if value is None else f'{value * 100:.2f}%'


class Task(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    def __init__(self, fn, parent=None):
        super().__init__(parent); self.fn = fn
    def run(self):
        try: self.succeeded.emit(self.fn())
        except Exception as e: self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self, service):
        super().__init__()
        self.service = service; self.state = {}; self.task = None; self.callback = None
        self.pending_close = False; self.queued_simple_train = None
        self.simple_dataset_id = None; self.simple_run_id = None; self.base_id = None
        self.simple_result = None; self.simple_result_key = None; self.result_cache = None
        self.choice_context = None; self.selected_weight = None; self.row_models = []
        self.context_targets = {}; self.context_menu = None
        self.setWindowTitle('문양검색 · 추가 학습'); self.resize(1060, 910); self.setMinimumSize(820, 660)
        body = QWidget(); self.setCentralWidget(body)
        outer = QVBoxLayout(body); outer.setContentsMargins(24, 18, 24, 12)
        brand = label('문양검색 · 추가 학습'); brand.setObjectName('brand'); outer.addWidget(brand)
        outer.addWidget(label('데이터 추가 → 학습 → 성능 비교 → 가중치 선택·저장', True))
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); outer.addWidget(self.scroll, 1)
        page = QWidget(); self.scroll.setWidget(page); layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 2, 5, 4); layout.setSpacing(8)
        self.build_workflow(layout); layout.addStretch()
        for combo, kind in ((self.simple_dataset, 'dataset'), (self.base_choice, 'model'), (self.simple_run, 'run')):
            combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            combo.customContextMenuRequested.connect(lambda pos, c=combo, k=kind: self.show_item_menu(k, c.currentData(), c.mapToGlobal(pos)))
            for watched in (combo.view(), combo.view().viewport()):
                watched.installEventFilter(self); self.context_targets[watched] = (combo, kind)
        self.comparison.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.comparison.customContextMenuRequested.connect(self.comparison_menu)
        self.message = label(''); outer.addWidget(self.message)
        self.timer = QTimer(self); self.timer.setInterval(5000); self.timer.timeout.connect(self.poll); self.timer.start()
        self.cleanup_last = None
        self.cleanup_timer = QTimer(self); self.cleanup_timer.setInterval(60000)
        self.cleanup_timer.timeout.connect(self.cleanup_models); self.cleanup_timer.start()
        self.refresh(); QTimer.singleShot(0, self.cleanup_models)

    def group(self, layout, title):
        box = QGroupBox(title); box.setObjectName('simpleGroup')
        inner = QVBoxLayout(box); inner.setSpacing(7); layout.addWidget(box); return inner

    def build_workflow(self, layout):
        box = self.group(layout, '1. 데이터 추가')
        row = QHBoxLayout()
        self.simple_folder = button('데이터 폴더 선택', lambda: self.pick_simple_data(True))
        self.simple_file = button('ZIP / CSV 선택', lambda: self.pick_simple_data(False))
        row.addWidget(self.simple_folder); row.addWidget(self.simple_file)
        self.simple_dataset = QComboBox(); row.addWidget(self.simple_dataset, 1); box.addLayout(row)
        self.simple_dataset.currentIndexChanged.connect(self.simple_dataset_changed)
        box.addWidget(label('이미지와 정답 labels.csv가 함께 필요합니다.', True))
        self.simple_data_note = label('', True); box.addWidget(self.simple_data_note)

        box = self.group(layout, '2. 기존 가중치에서 학습')
        row = QHBoxLayout(); row.addWidget(label('기준 가중치'))
        self.base_choice = QComboBox(); row.addWidget(self.base_choice, 1); box.addLayout(row)
        self.base_choice.currentIndexChanged.connect(self.base_changed)
        self.simple_base = label('', True); box.addWidget(self.simple_base)
        row = QHBoxLayout(); row.addWidget(label('실행 장치'))
        self.simple_device = QComboBox(); self.simple_device.addItem('CPU', 'cpu')
        self.simple_device.addItem('NVIDIA GPU 0', 'cuda:0'); row.addWidget(self.simple_device)
        row.addStretch(); self.simple_start = button('학습 시작', self.start_simple_training, True)
        row.addWidget(self.simple_start); self.simple_cancel = button('학습 중단', self.cancel_simple_training)
        row.addWidget(self.simple_cancel); box.addLayout(row)
        box.addWidget(label('학습 설정은 고정되어 있습니다. 기존 가중치는 그대로 보관합니다.', True))
        self.simple_progress = QProgressBar(); box.addWidget(self.simple_progress)
        self.simple_status = label(''); box.addWidget(self.simple_status)

        box = self.group(layout, '3. 성능 비교 · 가중치 선택 · 저장')
        row = QHBoxLayout(); row.addWidget(label('학습 결과'))
        self.simple_run = QComboBox(); row.addWidget(self.simple_run, 1); box.addLayout(row)
        self.simple_run.currentIndexChanged.connect(self.simple_run_changed)
        self.simple_result_note = label(''); box.addWidget(self.simple_result_note)
        self.comparison = table(['가중치 (행을 눌러 선택)', '평가 감성어 일치율 (F1@5)', '기존 대비'])
        self.comparison.setMinimumHeight(0); self.comparison.setFixedHeight(132); box.addWidget(self.comparison)
        self.comparison.itemSelectionChanged.connect(self.weight_changed)
        self.selected_note = label('', True); box.addWidget(self.selected_note)
        row = QHBoxLayout()
        self.simple_apply = button('선택 가중치 적용', self.apply_simple_model)
        self.simple_save = button('선택 가중치 다른 이름으로 저장…', self.save_model_as, True)
        row.addWidget(self.simple_apply); row.addWidget(self.simple_save, 1); box.addLayout(row)
        box.addWidget(label('적용: 다음 학습의 기본 가중치로 사용  ·  다른 이름으로 저장: 선택한 가중치를 ZIP으로 내보내기', True))
        box.addWidget(label('데이터·가중치·학습 결과 목록의 항목을 우클릭하면 저장·삭제할 수 있습니다.', True))

    def eventFilter(self, watched, event):
        target = self.context_targets.get(watched)
        if target and event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.RightButton:
            combo, kind = target; view = combo.view()
            point = event.position().toPoint()
            if watched is view: point = view.viewport().mapFrom(view, point)
            index = view.indexAt(point)
            if index.isValid():
                key = combo.itemData(index.row())  # Clicked row, NOT the currently selected row.
                global_pos = view.viewport().mapToGlobal(point)
                self.show_item_menu(kind, key, global_pos, combo)
            return True
        if target and event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.RightButton:
            return True  # A right-click opens a menu, never selects/closes a combo row.
        if target and event.type() == QEvent.Type.ContextMenu:
            # Windows can also send this after the right-button release. The menu
            # was already opened for the clicked row on press; do not retarget it.
            if event.reason() != QContextMenuEvent.Reason.Keyboard: return True
            combo, kind = target; index = combo.view().currentIndex()
            if index.isValid():
                key = combo.itemData(index.row())
                pos = combo.view().viewport().mapToGlobal(combo.view().visualRect(index).center())
                self.show_item_menu(kind, key, pos, combo)
            return True
        return super().eventFilter(watched, event)

    def comparison_menu(self, pos):
        index = self.comparison.indexAt(pos)
        if index.isValid() and index.row() < len(self.row_models):
            self.show_item_menu('model', self.row_models[index.row()], self.comparison.viewport().mapToGlobal(pos))

    def show_item_menu(self, kind, key, global_pos, combo=None):
        if not key: return
        collection = {'dataset':'datasets', 'model':'models', 'run':'runs'}[kind]
        item = next((m for m in self.state.get(collection, []) if m['id'] == key), None)
        if not item: return
        save_key = key
        if kind == 'run':
            if item.get('kind') != 'train': return
            save_key = None
            path = local_only(self.service.store.item('runs', key)/'result.json')
            if item.get('status') == 'complete' and path.is_file():
                try:
                    model_id = read(path).get('model_id')
                    if any(m['id']==model_id and m.get('has_head') for m in self.state.get('models', [])):
                        save_key = model_id
                except (OSError, ValueError, AttributeError): pass  # A broken/missing result must remain dismissible.
        # Keep the dropdown as the parent popup, like a submenu. Closing it here
        # makes the list vanish on right-click and loses the user's place.
        menu = QMenu(combo.view() if combo is not None else self)
        menu.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose); menu.setToolTipsVisible(True)
        def perform(action):
            if combo is not None: combo.hidePopup()  # Only after choosing a command.
            action()
        prefix = {'dataset':'데이터: ', 'model':'가중치: ', 'run':'학습 기록: '}[kind]
        heading = menu.addAction(prefix+item.get('name',key)); heading.setEnabled(False)
        menu.addSeparator()
        save = menu.addAction('다른 이름으로 저장…')
        save.triggered.connect(lambda checked=False, k=save_key, t=kind: perform(
            lambda: self.save_dataset_as(k) if t=='dataset' else self.save_model_as(k)))
        remove = menu.addAction('삭제…'); remove.triggered.connect(lambda checked=False, k=key, t=kind: perform(lambda: self.delete_item(t, k)))
        save.setEnabled(not self.busy()); remove.setEnabled(not self.busy())
        if kind == 'run':
            save.setEnabled(save.isEnabled() and bool(save_key))
            if not save_key: save.setToolTip('저장할 새 가중치가 없거나 삭제되었습니다. 학습 기록은 삭제할 수 있습니다.')
            remove.setEnabled(remove.isEnabled() and item.get('status') in ('complete','failed','cancelled','interrupted'))
            remove.setToolTip('학습 결과 목록에서 이 기록만 삭제합니다. 가중치·원본 데이터·내부 결과와 로그는 보존합니다.')
        if kind == 'model':
            if key == (self.state.get('active') or {}).get('model_id'):
                remove.setEnabled(False); remove.setToolTip('현재 적용 중입니다. 다른 가중치를 먼저 적용하세요.')
            elif any(key in (m.get('base_model'), m.get('backbone_model')) for m in self.state.get('models', []) if m['id'] != key):
                remove.setEnabled(False); remove.setToolTip('다른 가중치가 참조하는 원본입니다.')
            save.setEnabled(save.isEnabled() and bool(item.get('has_head')))
        self.context_menu = menu; menu.popup(global_pos)

    def delete_item(self, kind, key):
        if self.busy(): return
        collection = {'dataset':'datasets', 'model':'models', 'run':'runs'}[kind]
        item = next((m for m in self.state.get(collection, []) if m['id'] == key), None)
        if not item: return
        if kind == 'dataset':
            text = f'데이터: {item["name"]} · {item["count"]}장\n\n등록 목록에서 삭제할까요?\n원본 파일과 과거 학습 기록은 보존합니다. 저장 공간은 줄어들지 않습니다.'
        elif kind == 'run':
            text = f'학습 기록: {item.get("name",key)}\n\n학습 결과 목록에서 이 기록을 삭제할까요?\n가중치와 원본 데이터는 삭제하지 않습니다. 내부 결과·로그는 보존합니다.\n가중치 삭제는 기준 가중치 목록에서 별도로 할 수 있습니다.'
        else:
            policy = self.state.get('retention', {})
            text = f'가중치: {item["name"]}\n\n사용 목록에서 삭제할까요?\n학습·성능 기록은 보존합니다. '
            text += f'기존 보관 정책에 따라 {policy["days"]}일 후, 프로그램 실행 중 안전 검사 후 영구 삭제됩니다.' if policy.get('enabled') else '자동 영구 삭제가 꺼져 있어 파일은 보관됩니다.'
        if QMessageBox.question(self, '삭제 확인', text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes: return
        action = {'dataset':self.service.set_dataset_deleted, 'model':self.service.set_model_deleted,
                  'run':self.service.set_run_deleted}[kind]
        def done(result):
            if kind=='dataset' and self.simple_dataset_id==key: self.simple_dataset_id=None
            if kind=='model' and self.base_id==key: self.base_id=None
            if kind=='run' and self.simple_run_id==key:
                self.simple_run_id=None; self.simple_result_key=None; self.result_cache=None
            self.message.setText('학습 결과 목록에서 기록을 삭제했습니다. 가중치·데이터·내부 기록은 그대로 보존했습니다.' if kind=='run'
                                 else '목록에서 삭제했습니다. 원본과 학습 기록은 기존 보관 정책에 따라 유지됩니다.')
        self.run_task(lambda: action(key, True), done)

    def save_dataset_as(self, key):
        if self.busy(): return
        data = next((d for d in self.state.get('datasets', []) if d['id']==key), None)
        if not data: return
        name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', data['name']).strip(' .')[:80] or 'dataset'
        path, _ = QFileDialog.getSaveFileName(self, '데이터 다른 이름으로 저장',
            str(self.service.store.root.parent/(name+'.zip')), '데이터 ZIP (*.zip)',
            options=QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.DontConfirmOverwrite)
        if not path: return
        try:
            target = local_only(path)
            if target.suffix.lower()!='.zip': target=target.with_name(target.name+'.zip')
            overwrite=target.exists()
            if overwrite and QMessageBox.question(self, '기존 파일 교체',
                str(target)+'\n\n같은 이름의 ZIP을 교체할까요?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes: return
            self.run_task(lambda: self.service.export_dataset(key, str(target), overwrite),
                lambda r: self.message.setText('데이터 저장 완료 · 이미지·라벨·분할을 그대로 저장했습니다.\n'+r['path']))
        except Exception as e: self.error(str(e))

    @staticmethod
    def fill_combo(combo, choices, selected, placeholder):
        entries = [(placeholder, None)] + list(choices)
        previous = combo.blockSignals(True)
        if entries != [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]:
            combo.clear()
            for title, key in entries: combo.addItem(title, key)
        index = max(0, combo.findData(selected))
        if combo.currentIndex() != index: combo.setCurrentIndex(index)
        combo.blockSignals(previous)
        return combo.currentData()

    def busy(self):
        return (self.task is not None or self.queued_simple_train is not None or self.service.is_running()
                or any(r.get('status') in RUNNING for r in self.state.get('runs', [])))

    def refresh(self):
        try:
            self.state = self.service.snapshot(); self.update_simple_training()
        except Exception as e: self.error(str(e))

    def simple_dataset_changed(self):
        self.simple_dataset_id = self.simple_dataset.currentData(); self.update_simple_training()

    def base_changed(self):
        self.base_id = self.base_choice.currentData(); self.update_simple_training()

    def simple_run_changed(self):
        self.simple_run_id = self.simple_run.currentData(); self.update_simple_training()

    def update_simple_training(self):
        datasets = self.state.get('datasets', []); models = self.state.get('models', [])
        runs = [r for r in self.state.get('runs', []) if r.get('kind') == 'train']
        active = (self.state.get('active') or {}).get('model_id')
        self.simple_dataset_id = self.fill_combo(self.simple_dataset,
            [(f'{d["name"]} · {d["count"]}장', d['id']) for d in datasets], self.simple_dataset_id, '등록된 데이터 선택')
        data = next((d for d in datasets if d['id'] == self.simple_dataset_id), None)
        if data:
            c = data['counts']
            self.simple_data_note.setText(f'학습 {c.get("train", 0)}장 · 검증 {c.get("dev", 0)}장 · 평가 {c.get("eval", 0)}장')
        else: self.simple_data_note.setText('아직 선택한 데이터가 없습니다.')
        available = [m for m in models if m.get('has_head') and supports_training(m)]
        if self.base_id is None: self.base_id = active
        self.base_id = self.fill_combo(self.base_choice,
            [(m['name'], m['id']) for m in available], self.base_id, '학습에 사용할 가중치 선택')
        active_model = next((m for m in models if m['id'] == active), None)
        self.simple_base.setText('현재 적용: ' + active_model['name'] if active_model else
            '적용된 가중치가 없습니다. 설치 후 기준 가중치를 선택하세요.')
        active_run = next((r for r in runs if r.get('status') in RUNNING), None)
        if active_run: self.simple_run_id = active_run['id']
        if self.simple_run_id is None and runs: self.simple_run_id = runs[0]['id']
        self.simple_run_id = self.fill_combo(self.simple_run,
            [(f'{r.get("name", r["id"])} · {STATE_NAMES.get(r["status"], r["status"])}', r['id']) for r in runs],
            self.simple_run_id, '학습 기록 선택')
        status = next((r for r in runs if r['id'] == self.simple_run_id), None)
        idle = not self.busy(); ready = all(self.state.get('dependencies', {}).values())
        for w in (self.simple_folder, self.simple_file, self.simple_dataset, self.base_choice, self.simple_device, self.simple_run):
            w.setEnabled(idle)
        self.simple_start.setEnabled(idle and ready and bool(data and data['counts'].get('train') and self.base_id))
        self.simple_cancel.setVisible(bool(active_run)); self.simple_cancel.setEnabled(self.task is None and bool(active_run))
        self.simple_progress.setValue(int(status.get('progress', 0)) if status else 0)
        if status:
            eta = status.get('eta_seconds')
            timing = f'남은 학습 약 {max(1, round(eta/60))}분 (평가·저장 제외)' if eta else ('예상시간 측정 중' if status['status'] in RUNNING else '')
            self.simple_status.setText(f'{STATE_NAMES.get(status["status"], status["status"])} · {status.get("message", "")}\n{timing}'.strip())
        else:
            self.simple_status.setText('데이터를 선택한 뒤 학습을 시작하세요.' if ready else
                '실행 환경 설치가 필요합니다. install-windows.bat 또는 install-linux.sh를 실행하세요.')
        self.show_comparison(status, models, active)

    def show_comparison(self, status, models, active):
        self.simple_result = None; rows = []; self.row_models = []
        note = '학습이 끝나면 기존 가중치와 새 가중치의 성능을 비교합니다.'
        if status and status['status'] == 'complete':
            folder = self.service.store.item('runs', status['id'])
            result_path = local_only(folder / 'result.json'); config_path = local_only(folder / 'config.json')
            if result_path.is_file() and config_path.is_file():
                key = (status['id'], result_path.stat().st_mtime_ns, config_path.stat().st_mtime_ns)
                if key != self.simple_result_key:
                    self.result_cache = (read(result_path), read(config_path)); self.simple_result_key = key
                result, config = self.result_cache
                # Compare against this run's starting checkpoint, not today's active model.
                old_id, new_id = config.get('model_id'), result.get('model_id')
                metric = result.get('metrics', {}).get('eval', {})
                n = metric.get('sample_count', 0)
                score = metric.get('f1_at_5') if n else None
                before = metric.get('baseline', {}).get('f1_at_5') if n else None
                delta = (score - before) * 100 if score is not None and before is not None else None
                if delta is not None:
                    verdict = '새 가중치가 더 높습니다.' if delta > 1e-9 else ('기존 가중치가 더 높습니다.' if delta < -1e-9 else '두 가중치의 성능이 같습니다.')
                    note = f'같은 평가 이미지 {n}장 비교 · {verdict}'
                elif n and score is not None: note = f'평가 이미지 {n}장 · 기존 결과가 없어 성능 차이는 확인할 수 없습니다.'
                else: note = '별도 평가 데이터 없음 · 성능 개선 여부는 확인되지 않았습니다.'
                rows = [('기존 가중치', percent(before), '—'),
                        ('새 가중치', percent(score), f'{delta:+.2f}%p' if delta is not None else '—')]
                present = {m['id'] for m in models if m.get('has_head')}
                self.row_models = [old_id if old_id in present else None, new_id if new_id in present else None]
                if new_id in present: self.simple_result = result
                if None in self.row_models: note += '  삭제되거나 없는 가중치는 선택할 수 없습니다.'
            else: note = '결과 또는 학습 설정 파일이 없어 가중치를 비교할 수 없습니다.'
        elif status and status['status'] not in RUNNING:
            note = '학습이 완료되지 않아 선택할 새 가중치가 없습니다. 위의 상태를 확인하세요.'
        context = (status['id'] if status else None, tuple(self.row_models))
        if context != self.choice_context:
            self.choice_context = context
            self.selected_weight = active if active in self.row_models else next((k for k in self.row_models if k), None)
        if self.selected_weight not in self.row_models: self.selected_weight = None
        self.comparison.blockSignals(True); self.comparison.setRowCount(len(rows)); self.comparison.clearSelection()
        for i, values in enumerate(rows):
            self.comparison.setRowHeight(i, 37)
            for j, value in enumerate(values):
                item = QTableWidgetItem(value)
                if not self.row_models[i]: item.setFlags(Qt.ItemFlag.NoItemFlags)
                self.comparison.setItem(i, j, item)
        if self.selected_weight: self.comparison.selectRow(self.row_models.index(self.selected_weight))
        self.comparison.blockSignals(False); self.simple_result_note.setText(note); self.update_weight_actions()

    def weight_changed(self):
        selected = self.comparison.selectionModel().selectedRows()
        row = selected[0].row() if selected else -1
        self.selected_weight = self.row_models[row] if 0 <= row < len(self.row_models) else None
        self.update_weight_actions()

    def update_weight_actions(self):
        model = next((m for m in self.state.get('models', []) if m['id'] == self.selected_weight and m.get('has_head')), None)
        active = (self.state.get('active') or {}).get('model_id')
        applied = bool(model and model['id'] == active)
        self.selected_note.setText('선택: ' + model['name'] + (' · 현재 적용 중' if applied else '') if model else '비교표에서 사용할 가중치를 선택하세요.')
        self.comparison.setEnabled(not self.busy())
        self.simple_apply.setText('현재 적용 중' if applied else '선택 가중치 적용')
        self.simple_apply.setEnabled(not self.busy() and bool(model) and not applied)
        self.simple_save.setEnabled(not self.busy() and bool(model))

    def pick_simple_data(self, folder):
        if self.busy(): return
        if folder:
            path = QFileDialog.getExistingDirectory(self, '이미지와 labels.csv가 있는 폴더', str(self.service.store.root),
                options=QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.DontResolveSymlinks)
        else:
            path, _ = QFileDialog.getOpenFileName(self, '학습 데이터 선택', str(self.service.store.root),
                '학습 데이터 (*.zip *.csv)', options=QFileDialog.Option.DontUseNativeDialog)
        if not path: return
        def done(result):
            self.simple_dataset_id = result['id']
            self.message.setText(f'{result["count"]}장 등록 완료. 학습 시작을 눌러 주세요.')
        self.run_task(lambda: self.service.import_data(path, Path(path).stem), done)

    def start_simple_training(self):
        if self.busy(): return
        key = self.simple_dataset.currentData()
        active = self.base_choice.currentData()
        if not key or not active: self.error('학습 데이터와 기준 가중치를 먼저 선택하세요.'); return
        name = self.simple_dataset.currentText().split(' · ')[0]
        payload = dict(dataset_ids=[key], model_id=active, name=name+' 추가 학습',
                       device=self.simple_device.currentData(), **TRAIN_DEFAULTS)
        def confirm(plan):
            c = plan['counts']
            text = f'학습 {c["train"]}장으로 선택한 기준 가중치를 추가 학습합니다.\n'
            text += '검증 결과가 가장 좋은 가중치를 저장합니다.' if c['dev'] else '검증 데이터가 없어 학습을 마친 마지막 가중치를 저장합니다. 성능 개선은 확인할 수 없습니다.'
            text += '\n기존 가중치는 유지하며, 완료 후 사용할 가중치를 직접 선택합니다.\n\n시작할까요?'
            if QMessageBox.question(self, '학습 시작', text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
                self.queued_simple_train = payload
                QTimer.singleShot(100, self.launch_simple_training)
        self.run_task(lambda: self.service.store.training_plan(payload), confirm)

    def launch_simple_training(self):
        if self.task is not None:
            QTimer.singleShot(100, self.launch_simple_training); return
        payload = self.queued_simple_train; self.queued_simple_train = None
        if not payload: return
        def done(result):
            self.simple_run_id = result['id']; self.simple_result_key = None
            self.message.setText('학습을 시작했습니다. 완료되면 아래에서 성능을 비교하고 가중치를 선택하세요.')
        self.run_task(lambda: self.service.launch('train', payload), done)

    def cancel_simple_training(self):
        status = next((r for r in self.state.get('runs', []) if r.get('status') in RUNNING and r.get('kind') == 'train'), None)
        if status:
            try:
                self.service.cancel(status['id']); self.message.setText('현재 처리 단위가 끝나면 안전하게 중단합니다.')
            except Exception as e: self.error(str(e))

    def apply_simple_model(self):
        if self.busy() or not self.selected_weight: return
        key = self.selected_weight
        text = self.selected_note.text() + '\n\n이 가중치를 다음 학습의 기본값으로 적용할까요?\n이전 가중치는 그대로 보관합니다.'
        if QMessageBox.question(self, '가중치 적용', text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            def done(result):
                self.base_id = key; self.message.setText('선택한 가중치를 적용했습니다. 다음 학습은 이 가중치에서 시작합니다.')
            self.run_task(lambda: self.service.activate(key), done)

    def save_model_as(self, key=None):
        if key is None or isinstance(key, bool): key = self.selected_weight
        if self.busy() or not key: return
        model = next((m for m in self.state.get('models', []) if m['id'] == key), None)
        if not model: return
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', model.get('name', key)).strip(' .')[:80] or 'model'
        path, _ = QFileDialog.getSaveFileName(self, '가중치 다른 이름으로 저장',
            str(self.service.store.root.parent/(safe_name+'.zip')), '가중치 백업 ZIP (*.zip)',
            options=QFileDialog.Option.DontUseNativeDialog|QFileDialog.Option.DontConfirmOverwrite)
        if not path: return
        try:
            target = local_only(path)
            if target.suffix.lower() != '.zip': target = target.with_name(target.name+'.zip')
            overwrite = target.exists()
            text = ('선택 가중치와 필요한 백본을 함께 저장합니다. 모델에 따라 수 GB 이상이며 저장에 시간이 걸릴 수 있습니다.\n'
                    '등록 데이터와 현재 적용 모델은 변경하지 않습니다.\n\n저장 위치: '+str(target))
            if overwrite: text += '\n\n같은 이름의 기존 ZIP을 교체합니다. 계속할까요?'
            if QMessageBox.question(self, '가중치 저장 확인', text,
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes: return
            self.run_task(lambda: self.service.export_model(key, str(target), overwrite),
                lambda r: self.message.setText(f'가중치 저장 완료 · {r["bytes"]/1024**3:.2f} GB\n{r["path"]}'))
            self.message.setText('가중치 저장 중 · 필요한 원본 모델을 함께 복사하고 무결성을 확인합니다. 창을 닫지 마세요.')
        except Exception as e: self.error(str(e))

    def run_task(self, fn, callback):
        if self.task is not None: return
        self.callback = callback; self.task = Task(fn, self)
        self.task.succeeded.connect(self.task_success); self.task.failed.connect(self.task_error); self.task.finished.connect(self.task_finished)
        self.message.setText('처리 중입니다. 잠시 기다려 주세요.'); self.update_simple_training(); self.task.start()

    @Slot(object)
    def task_success(self, result):
        try:
            if self.callback: self.callback(result)
        except Exception as e: self.error(str(e))

    @Slot(str)
    def task_error(self, message): self.error(message)

    @Slot()
    def task_finished(self):
        task = self.task; self.task = None; self.callback = None
        task.deleteLater(); self.refresh()

    def error(self, text):
        self.message.setText('확인 필요: ' + text)

    def cleanup_models(self):
        if self.busy() or self.pending_close:return
        if self.cleanup_last is not None and time.monotonic()-self.cleanup_last<3600:return
        if not self.state.get('retention',{}).get('enabled'):return
        self.cleanup_last=time.monotonic()
        def report(result):
            if result['deferred']:
                self.cleanup_last=None;return
            text=f'자동 정리: {len(result["deleted"])}개 가중치 영구 삭제.'
            if result['blocked']:text+=' 삭제 보류: '+'; '.join(r['id']+' — '+r['reason'] for r in result['blocked'])
            self.message.setText(text if result['deleted'] or result['blocked'] else '')
        self.run_task(self.service.cleanup_expired_models,report)

    def poll(self):
        if self.service.is_running() or self.pending_close or any(r['status'] in RUNNING for r in self.state.get('runs', [])):
            self.refresh()
        if self.pending_close and not self.service.is_running() and self.task is None: self.close()

    def closeEvent(self, event):
        if self.task is not None or self.queued_simple_train is not None:
            self.message.setText('파일 처리 중입니다. 작업이 끝난 뒤 창을 닫아 주세요.'); event.ignore(); return
        if self.service.is_running():
            event.ignore()
            if self.pending_close: return
            if QMessageBox.question(self, '작업 중', '작업 중입니다. 안전하게 중단한 뒤 프로그램을 종료할까요?') == QMessageBox.StandardButton.Yes:
                for status in self.service.store.list('runs'):
                    if status['status'] in RUNNING:
                        try: self.service.cancel(status['id'])
                        except ValueError: pass
                self.pending_close = True; self.message.setText('안전한 중단을 기다리는 중입니다. 완료되면 창이 닫힙니다.')
            return
        self.timer.stop();self.cleanup_timer.stop(); event.accept()


def main():
    app = QApplication(sys.argv); app.setApplicationName('PatternTraining'); app.setStyle('Fusion')
    palette = QPalette(); palette.setColor(QPalette.ColorRole.Window, QColor('#f5f7fa')); palette.setColor(QPalette.ColorRole.WindowText, QColor('#25354a'))
    palette.setColor(QPalette.ColorRole.Base, QColor('white')); palette.setColor(QPalette.ColorRole.Text, QColor('#25354a'))
    palette.setColor(QPalette.ColorRole.Button, QColor('white')); palette.setColor(QPalette.ColorRole.ButtonText, QColor('#25354a'))
    app.setPalette(palette); app.setFont(QFont('Malgun Gothic' if sys.platform == 'win32' else 'Noto Sans CJK KR', 10)); app.setStyleSheet(STYLE)
    lock = None
    try:
        local_only(ROOT); lock = WorkspaceLock(ROOT / 'workspace')
        window = MainWindow(App()); window.show(); return app.exec()
    except Exception as e:
        traceback.print_exc(); QMessageBox.critical(None, '시작 오류', str(e)); return 1
    finally:
        if lock: lock.close()


if __name__ == '__main__':
    sys.exit(main())
