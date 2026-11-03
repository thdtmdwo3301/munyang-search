"""Native Qt Widgets UI. No HTML, WebView, HTTP listener or browser."""
import json
import sys
import traceback
import time
import re
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont, QPalette, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QStackedWidget, QScrollArea,
    QGroupBox, QFormLayout, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QFileDialog,
    QMessageBox, QProgressBar, QPlainTextEdit, QSplitter)
from app_service import App, WorkspaceLock, ROOT, RUNNING
from core import LABELS, TRAIN_DEFAULTS, local_only, read, supports_training

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
QPushButton { background: white; border: 1px solid #bccbda; border-radius: 4px; padding: 10px 17px; }
QPushButton:hover { background: #edf3fa; border-color: #356da3; }
QPushButton[primary="true"] { background: #285f98; color: white; border-color: #285f98; }
QPushButton:disabled { background: #edf0f4; color: #738093; border-color: #d8e0e9; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: white; color: #25354a; border: 1px solid #bccbda; border-radius: 3px; padding: 8px; min-height: 22px; }
QComboBox QAbstractItemView { background: white; color: #25354a; selection-background-color: #dceafa; selection-color: #173b62; }
QListWidget, QTableWidget, QPlainTextEdit { background: white; color: #25354a; border: 1px solid #d8e0e9; selection-background-color: #e1ecfa; selection-color: #173b62; }
QListWidget#nav { background: #f5f7fa; border: none; font-size: 16px; }
QListWidget#nav::item { padding: 17px 12px; }
QListWidget#nav::item:selected { background: #e1ecfa; color: #225c98; }
QHeaderView::section { background: #f2f5f9; color: #53657b; padding: 9px; border: none; border-bottom: 1px solid #d8e0e9; }
QProgressBar { background: #e5ebf2; border: none; border-radius: 4px; text-align: center; min-height: 22px; }
QProgressBar::chunk { background: #8ab0d4; border-radius: 4px; }
QScrollArea { border: none; }
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
        self.infer_id = None; self.result_id = None; self.active_id = None; self.pending_close = False
        self.infer_images = []; self.infer_index = 0; self.infer_output = None
        self.infer_completed_id = None; self.infer_comparisons = {}; self.input_model_id = None
        self.model_check = None; self.advanced_enabled = False
        self.setWindowTitle('문양검색 · 학습 관리')
        self.resize(1180, 850); self.setMinimumSize(900, 680)
        body = QWidget(); self.setCentralWidget(body); outer = QVBoxLayout(body); outer.setContentsMargins(22, 18, 22, 12)
        header = QHBoxLayout(); brand = label('문양검색  |  학습 관리'); brand.setObjectName('brand')
        self.active_label = label('적용 모델 없음', True)
        header.addWidget(brand); header.addStretch(); header.addWidget(self.active_label); outer.addLayout(header)
        area = QHBoxLayout(); outer.addLayout(area, 1)
        self.nav = QListWidget(); self.nav.setObjectName('nav'); self.nav.setFixedWidth(175)
        self.nav.addItems(['데이터 등록', '학습', '평가', '추론', '가중치 보관함'])
        area.addWidget(self.nav); self.stack = QStackedWidget(); area.addWidget(self.stack, 1)
        self.build_data(); self.build_training(); self.build_evaluation(); self.build_inference(); self.build_models()
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex); self.nav.setCurrentRow(0)
        self.message = label(''); outer.addWidget(self.message)
        footer = QHBoxLayout(); self.storage = label(str(self.service.store.root), True)
        self.storage.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        footer.addWidget(self.storage, 1); footer.addWidget(button('저장 경로 복사', lambda: QApplication.clipboard().setText(str(self.service.store.root))))
        footer.addWidget(button('새로고침', self.refresh)); outer.addLayout(footer)
        self.timer = QTimer(self); self.timer.setInterval(5000); self.timer.timeout.connect(self.poll); self.timer.start()
        self.refresh()
        self.cleanup_last=None
        self.cleanup_timer=QTimer(self);self.cleanup_timer.setInterval(60000)
        self.cleanup_timer.timeout.connect(self.cleanup_models);self.cleanup_timer.start()
        QTimer.singleShot(0,self.cleanup_models)

    def page(self, title, subtitle):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(22, 22, 14, 22); layout.setSpacing(18)
        heading = label(title); heading.setObjectName('title'); layout.addWidget(heading); layout.addWidget(label(subtitle, True))
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(page); self.stack.addWidget(scroll)
        return layout

    def group(self, layout, title):
        box = QGroupBox(title); inner = QVBoxLayout(box); inner.setSpacing(12); layout.addWidget(box); return inner

    def file_row(self, layout, description, pattern, save=False):
        row = QHBoxLayout(); field = QLineEdit(); field.setReadOnly(True); field.setPlaceholderText(description)
        row.addWidget(field, 1)
        row.addWidget(button('파일 선택', lambda: self.pick_file(field, pattern)))
        layout.addLayout(row); return field

    def pick_file(self, field, pattern):
        path, _ = QFileDialog.getOpenFileName(self, '로컬 파일 선택', str(self.service.store.root), pattern,
                                             options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try: local_only(path); field.setText(path)
            except Exception as e: self.error(str(e))

    def build_data(self):
        layout = self.page('문양 데이터를 업데이트해 주세요', '이미지와 정답을 등록합니다. 원설명문은 없어도 됩니다.')
        box = self.group(layout, '새 데이터 등록')
        self.data_name = QLineEdit(); self.data_name.setMaxLength(100); self.data_name.setPlaceholderText('예: 2026년 10월 추가 문양')
        box.addWidget(label('데이터 이름')); box.addWidget(self.data_name)
        self.zip_path = QLineEdit(); self.zip_path.setReadOnly(True)
        self.zip_path.setPlaceholderText('ZIP · 정답 CSV · 이미지 한 장 · 데이터 폴더')
        box.addWidget(self.zip_path)
        pickers=QHBoxLayout()
        self.data_file_button=button('파일 선택 (ZIP / CSV / 이미지)',self.pick_data_file)
        self.data_folder_button=button('폴더 선택',self.pick_data_folder)
        pickers.addWidget(self.data_file_button); pickers.addWidget(self.data_folder_button); box.addLayout(pickers)
        box.addWidget(label('여러 이미지: labels.csv와 이미지가 있는 폴더, CSV 또는 ZIP을 선택하세요.\n이미지 한 장: 아래에서 실제 정답 5개를 직접 선택합니다. 정답 없이 예측만 하려면 추론 화면을 이용하세요.', True))
        self.single_data_panel=QGroupBox('이미지 한 장의 정답 정보'); single=QVBoxLayout(self.single_data_panel)
        single.addWidget(label('실제 정답 감성어 5개를 선택하세요. 모델 예측을 정답으로 자동 입력하지 않습니다.',True))
        self.data_labels=QListWidget(); self.data_labels.setMaximumHeight(155)
        for name in LABELS:
            item=QListWidgetItem(name); item.setFlags(item.flags()|Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked); self.data_labels.addItem(item)
        single.addWidget(self.data_labels)
        self.data_label_count=label('선택한 정답: 0 / 5'); single.addWidget(self.data_label_count)
        self.data_labels.itemChanged.connect(self.update_data_label_count)
        self.data_split=QComboBox()
        for title,key in [('학습용','train'),('검증용 (모델 선택)','dev'),('평가용 (테스트)','eval')]: self.data_split.addItem(title,key)
        form=QFormLayout(); form.addRow('데이터 용도',self.data_split); single.addLayout(form)
        box.addWidget(self.single_data_panel); self.single_data_panel.hide()
        self.zip_path.textChanged.connect(self.data_path_changed)
        row = QHBoxLayout(); row.addWidget(button('정답 목록 양식 저장', self.save_template)); row.addStretch()
        self.import_button = button('데이터 확인 및 등록', self.import_data, True); row.addWidget(self.import_button); box.addLayout(row)
        box = self.group(layout, '등록된 데이터'); self.datasets = table(['이름', '전체', '학습', '검증', '평가(테스트)']); box.addWidget(self.datasets)
        self.datasets.itemSelectionChanged.connect(self.enable_dataset_actions)
        actions=QHBoxLayout();self.delete_data_button=button('선택 데이터 삭제',self.delete_dataset)
        self.trash_toggle=button('삭제 보관함 보기',self.toggle_dataset_trash)
        actions.addWidget(self.delete_data_button);actions.addWidget(self.trash_toggle);box.addLayout(actions)
        box.addWidget(label('삭제는 등록 목록에서 제외하는 기능입니다. 원본 파일·기존 학습 기록은 유지하며 보관함에서 복원할 수 있습니다. 저장 공간은 줄어들지 않습니다.',True))
        self.trash_panel=QGroupBox('삭제 보관함');trash=QVBoxLayout(self.trash_panel)
        self.deleted_datasets=table(['이름','이미지 수']);trash.addWidget(self.deleted_datasets)
        self.deleted_datasets.itemSelectionChanged.connect(self.enable_dataset_actions)
        self.restore_data_button=button('선택 데이터 복원',self.restore_dataset);trash.addWidget(self.restore_data_button)
        box.addWidget(self.trash_panel);self.trash_panel.hide()
        layout.addWidget(button('학습 설정 확인 →', lambda: self.nav.setCurrentRow(1), True)); layout.addStretch()

    def device_combo(self):
        box = QComboBox(); box.addItem('CPU', 'cpu'); box.addItem('NVIDIA GPU 0', 'cuda:0'); return box

    def build_training(self):
        layout = self.page('학습 설정', '등록한 데이터와 기준 가중치로 새 모델 버전을 학습합니다.')
        self.engine_notice = label('', True); layout.addWidget(self.engine_notice)
        box = self.group(layout, '학습 데이터'); self.dataset_choices = QListWidget(); self.dataset_choices.setMinimumHeight(120); box.addWidget(self.dataset_choices)
        box = self.group(layout, '모델과 실행 설정'); form = QFormLayout(); box.addLayout(form)
        self.run_name = QLineEdit('문양 추가 학습'); self.run_name.setMaxLength(100); form.addRow('학습 이름', self.run_name)
        self.train_model = QComboBox(); form.addRow('기준 가중치', self.train_model)
        form.addRow('학습 방식', label('기존 분류/융합부 이어서 학습 · 특징 추출부 고정'))
        self.train_device = self.device_combo(); form.addRow('실행 장치', self.train_device)
        box.addWidget(label('검증 데이터가 있으면 최고 성능 에폭을 저장합니다. 없으면 지정한 마지막 에폭을 실행 확인용으로 저장하며 성능 개선으로 판단하지 않습니다. A/B 모델은 기존 두 분류 헤드만 학습하고 RBF는 유지합니다.', True))
        box.addWidget(label('5종 BERT 융합 모델은 labels.csv의 description 설명문이 모든 이미지에 필요합니다. 이미지·텍스트 인코더와 앙상블 비율은 고정하고 기존 융합부를 학습합니다. 원기록 80%는 이미지+원설명문 조건이며 현재 데이터 성능을 보장하지 않습니다.',True))
        self.advanced_button = button('고급 설정 열기', self.toggle_advanced); box.addWidget(self.advanced_button)
        self.advanced_panel = QWidget(); advanced = QFormLayout(self.advanced_panel); box.addWidget(self.advanced_panel)
        self.epochs = QSpinBox(); self.epochs.setRange(1, 500); self.epochs.setValue(TRAIN_DEFAULTS['epochs']); advanced.addRow('에폭', self.epochs)
        self.epochs.setToolTip('에폭: 전체 학습 데이터를 몇 번 반복해서 볼지 정합니다. 많다고 항상 성능이 좋아지는 것은 아닙니다.')
        self.batch = QComboBox()
        for n in (1, 2, 4, 8, 16, 32): self.batch.addItem(str(n), n)
        self.batch.setCurrentIndex(self.batch.findData(TRAIN_DEFAULTS['batch'])); advanced.addRow('배치 크기', self.batch)
        self.batch.setToolTip('배치 크기: 한 번에 처리하는 이미지 수입니다. 메모리가 부족하면 줄이세요. 처리 속도와 학습 결과에 영향을 줄 수 있습니다.')
        self.lr = QDoubleSpinBox(); self.lr.setDecimals(7); self.lr.setRange(.0000001, .01); self.lr.setSingleStep(.00001); self.lr.setValue(TRAIN_DEFAULTS['lr']); advanced.addRow('학습률', self.lr)
        self.lr.setToolTip('학습률: 모델을 한 단계에서 얼마나 바꿀지 정하는 값입니다. 특별한 이유가 없다면 기본값을 유지하세요.')
        advanced.addRow(label('기본값: 에폭 10 · 배치 크기 8 · 학습률 0.0001. 현재 프로그램의 시작값이며 기존 최고 성능을 재현한 최적값은 아닙니다. 고급 설정을 닫으면 기본값으로 돌아갑니다.', True))
        self.advanced_panel.hide()
        self.plan_label = label('검증셋으로 에폭을 선택한 뒤 평가(테스트)합니다. 기존 가중치는 덮어쓰지 않습니다.', True); box.addWidget(self.plan_label)
        row = QHBoxLayout(); self.plan_button = button('설정 확인', self.check_plan); row.addWidget(self.plan_button); row.addStretch()
        self.train_button = button('학습 시작', self.start_train, True); row.addWidget(self.train_button); box.addLayout(row)
        box = self.group(layout, '실행 현황'); self.job_choice = QComboBox(); self.job_choice.currentIndexChanged.connect(self.show_job); box.addWidget(self.job_choice)
        self.progress = QProgressBar(); box.addWidget(self.progress); self.job_text = label('아직 실행한 학습이 없습니다.'); box.addWidget(self.job_text)
        self.cancel_button = button('안전하게 중단', self.cancel_job); box.addWidget(self.cancel_button)
        box.addWidget(label('예상시간: 초기 측정 후 남은 학습 배치 시간을 표시합니다. 특징 추출·평가·저장은 별도입니다.', True)); layout.addStretch()

    def build_evaluation(self):
        layout = self.page('학습 결과', '같은 분할에서 기준 모델과 새 모델의 Top-5 성능을 비교합니다.')
        box = self.group(layout, '완료된 학습'); self.result_choice = QComboBox(); box.addWidget(self.result_choice)
        box.addWidget(button('결과 보기', self.load_result))
        self.evaluation_summary=label('평가 결과가 아직 없습니다.'); self.evaluation_summary.setObjectName('title');layout.addWidget(self.evaluation_summary)
        layout.addWidget(label('감성어 일치율(F1@5): 이미지마다 예측한 5개와 등록 정답 5개가 얼마나 일치하는지 전체 평가 이미지에서 계산합니다. 학습 성능과 평가 성능은 다릅니다.',True))
        self.metrics = table(['구분', '이미지 수', '기존 일치율', '새 모델 일치율', '맞힌 감성어 / 전체']); layout.addWidget(self.metrics)
        self.result_text = QPlainTextEdit(); self.result_text.setReadOnly(True); self.result_text.setPlaceholderText('실제 학습이 완료되면 결과를 확인할 수 있습니다.'); layout.addWidget(self.result_text)
        self.export_button = button('표본별 Top-5 저장', self.export_predictions); self.export_button.setEnabled(False); layout.addWidget(self.export_button)

    def build_inference(self):
        layout = self.page('문양 추론', '이미지 전용 모델은 이미지만, 5종 BERT 융합 모델은 이미지와 설명문을 사용합니다. 정답은 입력하지 않습니다.')
        box = self.group(layout, '입력과 모델'); form = QFormLayout(); self.infer_model = QComboBox(); self.infer_device = self.device_combo()
        form.addRow('추론 모델', self.infer_model); form.addRow('실행 장치', self.infer_device); box.addLayout(form)
        self.image_path = QLineEdit(); self.image_path.setReadOnly(True); self.image_path.setPlaceholderText('문양 이미지 · 여러 장 선택 가능')
        box.addWidget(self.image_path)
        self.image_path.textChanged.connect(self.single_inference_path_changed)
        pickers = QHBoxLayout()
        self.infer_files_button = button('이미지 선택 (여러 장 가능)', self.pick_inference_files)
        self.infer_folder_button = button('이미지 폴더 선택', self.pick_inference_folder)
        pickers.addWidget(self.infer_files_button); pickers.addWidget(self.infer_folder_button); box.addLayout(pickers)
        box.addWidget(label('폴더 바로 아래의 JPG·PNG·WebP를 파일명 순으로 읽습니다. 하위 폴더는 제외합니다.\n최대 1,000장 · 한 장 16MB · 합계 512MB. 한 장씩 순차 처리하므로 CPU에서도 사용할 수 있습니다.', True))
        navigation = QHBoxLayout()
        self.infer_previous = button('◀ 이전', lambda: self.move_inference(-1))
        self.infer_position = label('0 / 0'); self.infer_position.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.infer_next = button('다음 ▶', lambda: self.move_inference(1))
        navigation.addWidget(self.infer_previous); navigation.addWidget(self.infer_position, 1); navigation.addWidget(self.infer_next)
        self.infer_previous.setEnabled(False); self.infer_next.setEnabled(False)
        self.description_panel=QGroupBox('5종 BERT 모델용 설명문');description_layout=QVBoxLayout(self.description_panel)
        self.infer_description=QPlainTextEdit();self.infer_description.setMaximumHeight(100)
        self.infer_description.setPlaceholderText('이미지의 원설명문을 입력하세요. 감성어 정답 5개를 대신 넣지 마세요.')
        self.infer_description.textChanged.connect(self.inference_description_changed)
        description_layout.addWidget(self.infer_description)
        description_layout.addWidget(label('이 모델의 원기록 F1@5 80%는 이미지+원설명문 조건입니다. 설명문 자동 생성 기능이 아닙니다.',True))
        description_layout.addWidget(label('현재 표시된 이미지의 설명문만 입력합니다. 이전·다음으로 이동해 각 이미지의 설명문을 입력한 뒤 전체 추론을 실행하세요.', True))
        box.addWidget(self.description_panel);self.description_panel.hide()
        self.infer_model.currentIndexChanged.connect(self.update_input_mode)
        self.preview = label('이미지를 선택해 주세요.'); self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter); self.preview.setMinimumHeight(180)
        actions = QHBoxLayout()
        self.infer_button = button('추론 실행', self.start_infer, True); actions.addWidget(self.infer_button)
        self.infer_cancel = button('추론 중단', self.cancel_inference); self.infer_cancel.setEnabled(False); actions.addWidget(self.infer_cancel); box.addLayout(actions)
        self.infer_progress = label(''); box.addWidget(self.infer_progress)
        layout.addLayout(navigation)
        self.infer_summary=label('이미지를 선택하고 추론을 실행해 주세요.');layout.addWidget(self.infer_summary)
        results = QHBoxLayout(); results.addWidget(self.preview, 1)
        self.prediction = table(['순위', '예측 감성어', '등록 정답과 비교', '상세 점수']);self.prediction.setColumnHidden(3,True); results.addWidget(self.prediction, 2); layout.addLayout(results)
        self.score_details_button=button('상세 점수 보기',self.toggle_score_details);layout.addWidget(self.score_details_button)
        self.score_note=label('상세 점수는 예측 순위를 정하는 값이며 확률·정확도가 아닙니다. 1을 넘을 수도 있습니다.',True);layout.addWidget(self.score_note);self.score_note.hide()
        self.infer_status = label('정답은 추론 완료 후 비교에만 사용하며 모델 입력에는 전달하지 않습니다.', True); layout.addWidget(self.infer_status); layout.addStretch()

    def build_models(self):
        layout = self.page('가중치 보관함', '기준 모델과 추가 학습한 모델을 버전별로 보관합니다.')
        box = self.group(layout, '로컬 모델 등록'); box.addWidget(label('가중치가 모인 폴더를 선택하세요. 원본은 그대로 두고 필요한 파일만 보관함으로 복사합니다.', True))
        row=QHBoxLayout(); self.model_folder=QLineEdit(); self.model_folder.setReadOnly(True); self.model_folder.setPlaceholderText('로컬 가중치 폴더를 선택해 주세요')
        self.model_folder.textChanged.connect(self.clear_model_check); row.addWidget(self.model_folder,1)
        self.folder_button=button('폴더 선택',self.pick_model_folder); row.addWidget(self.folder_button); box.addLayout(row)
        self.inspect_button=button('파일 확인',self.inspect_model); box.addWidget(self.inspect_button)
        self.model_check_text=label('폴더 선택 → 파일 확인 → 등록 순서로 진행합니다.',True); box.addWidget(self.model_check_text)
        self.register_button = button('확인한 모델 등록', self.register_model,True); box.addWidget(self.register_button)
        self.load_archive_button = button('저장한 가중치 ZIP 불러오기', self.load_model_archive); box.addWidget(self.load_archive_button)
        box.addWidget(label('DINO 또는 A/B+RBF 모델의 기존 분류 헤드를 추가 학습할 수 있습니다. 원본 특징 추출부는 고정합니다. RBF 파일 하나만으로는 등록할 수 없습니다. 등록 시 실제 로딩을 검사합니다.', True))
        self.models = table(['이름', '상태', '평가 F1@5', '모델 ID']); layout.addWidget(self.models)
        self.models.itemSelectionChanged.connect(self.enable_model_actions)
        self.activate_button = button('선택 모델 적용', self.activate_model, True); layout.addWidget(self.activate_button)
        self.save_model_button = button('선택 가중치 다른 이름으로 저장…', self.save_model_as); layout.addWidget(self.save_model_button)
        layout.addWidget(label('선택 가중치와 실행에 필요한 원본 모델을 ZIP 하나로 저장합니다. 등록 이미지·데이터는 포함하지 않습니다. 다른 PC에서도 위의 ZIP 불러오기로 등록할 수 있습니다.', True))
        actions=QHBoxLayout();self.delete_model_button=button('선택 가중치 삭제',self.delete_model)
        self.model_trash_toggle=button('삭제 가중치 보관함 보기',self.toggle_model_trash)
        actions.addWidget(self.delete_model_button);actions.addWidget(self.model_trash_toggle);layout.addLayout(actions)
        layout.addWidget(label('삭제 보관함에서는 복원할 수 있습니다. 영구 삭제하면 실제 파일이 지워져 공간을 확보하며 복원할 수 없습니다. 현재 적용 모델과 다른 모델이 참조하는 원본은 보호합니다.',True))
        policy=self.group(layout,'가중치 자동 정리')
        row=QHBoxLayout();self.retention_enabled=QComboBox()
        self.retention_enabled.addItem('자동 삭제 사용 안 함',False);self.retention_enabled.addItem('기간 후 자동 영구 삭제',True)
        self.retention_days=QSpinBox();self.retention_days.setRange(1,365);self.retention_days.setValue(7);self.retention_days.setSuffix('일 보관')
        self.retention_save=button('보관 기간 적용',self.save_retention)
        row.addWidget(self.retention_enabled);row.addWidget(self.retention_days);row.addWidget(self.retention_save);policy.addLayout(row)
        self.retention_status=label('',True);policy.addWidget(self.retention_status)
        policy.addWidget(label('삭제 보관함의 가중치만 정리합니다. 프로그램 시작 시와 실행 중 매시간 확인하며 학습·추론 중에는 미룹니다. 프로그램이 꺼져 있으면 다음 실행 때 처리합니다.',True))
        self.model_trash_panel=QGroupBox('삭제 가중치 보관함');trash=QVBoxLayout(self.model_trash_panel)
        self.deleted_models=table(['이름','모델 ID','영구 삭제 예정']);trash.addWidget(self.deleted_models)
        self.deleted_models.itemSelectionChanged.connect(self.enable_model_actions)
        self.restore_model_button=button('선택 가중치 복원',self.restore_model);trash.addWidget(self.restore_model_button)
        self.purge_model_button=button('선택 가중치 즉시 영구 삭제',self.purge_model);trash.addWidget(self.purge_model_button)
        layout.addWidget(self.model_trash_panel);self.model_trash_panel.hide()
        layout.addWidget(label('성능 결과·학습 이력은 영구 삭제 후에도 남습니다. 삭제한 가중치로 재추론할 수는 없습니다. 새 모델을 자동 적용하지 않습니다.', True)); layout.addStretch()

    def select_options(self, combo, items):
        prior = combo.currentData(); combo.blockSignals(True); combo.clear()
        for item in items: combo.addItem(item.get('name') or item['id'], item['id'])
        index = combo.findData(prior)
        if index >= 0: combo.setCurrentIndex(index)
        combo.blockSignals(False)

    @Slot()
    def refresh(self):
        if self.task is not None: return
        try:
            self.state = self.service.snapshot()
            datasets, models, runs = (self.state[x] for x in ('datasets', 'models', 'runs'))
            checked = {self.dataset_choices.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.dataset_choices.count()) if self.dataset_choices.item(i).checkState() == Qt.CheckState.Checked}
            self.dataset_choices.clear()
            for d in datasets:
                item = QListWidgetItem(f'{d["name"]}   ·   {d["count"]}장')
                item.setData(Qt.ItemDataRole.UserRole, d['id']); item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if d['id'] in checked else Qt.CheckState.Unchecked); self.dataset_choices.addItem(item)
            selected_data=self.selected_dataset_id(self.datasets)
            selected_deleted=self.selected_dataset_id(self.deleted_datasets)
            populate(self.datasets, [[d['name'], d['count'], d['counts'].get('train', 0), d['counts'].get('dev', 0), d['counts'].get('eval', 0)] for d in datasets])
            for i,d in enumerate(datasets):
                self.datasets.item(i,0).setData(Qt.ItemDataRole.UserRole,d['id'])
                if d['id']==selected_data:self.datasets.selectRow(i)
            deleted=self.state.get('deleted_datasets',[])
            populate(self.deleted_datasets,[[d['name'],d['count']] for d in deleted])
            for i,d in enumerate(deleted):
                self.deleted_datasets.item(i,0).setData(Qt.ItemDataRole.UserRole,d['id'])
                if d['id']==selected_deleted:self.deleted_datasets.selectRow(i)
            self.select_options(self.train_model, [m for m in models if supports_training(m)]); self.select_options(self.infer_model, [m for m in models if m.get('has_head')])
            self.select_options(self.job_choice, [r for r in runs if r.get('kind') == 'train'])
            self.select_options(self.result_choice, [r for r in runs if r.get('kind') == 'train' and r['status'] == 'complete'])
            active = (self.state.get('active') or {}).get('model_id')
            active_model = next((m for m in models if m['id'] == active), None)
            self.active_label.setText('적용 모델 · ' + active_model['name'] if active_model else '적용 모델 없음')
            if active != self.active_id:
                self.active_id = active
                if self.infer_model.findData(active) >= 0: self.infer_model.setCurrentIndex(self.infer_model.findData(active))
                if self.train_model.findData(active) >= 0: self.train_model.setCurrentIndex(self.train_model.findData(active))
            selected_model = self.models.item(self.models.currentRow(), 3)
            selected_id = selected_model.text() if selected_model else None
            deleted_model_id=self.model_key(self.deleted_models,1)
            populate(self.models, [[m['name'], ('현재 적용 · ' if m['id'] == active else '')+('추가 학습 가능' if supports_training(m) else '추론 전용'), percent(m.get('evaluation_f1')), m['id']] for m in models])
            for i, m in enumerate(models):
                if m['id'] == selected_id: self.models.selectRow(i)
            deleted_models=self.state.get('deleted_models',[])
            def deadline(m):
                record=m.get('trash',{})
                if record.get('status')=='purging':return '삭제 중단 · 재시도 필요'
                value=record.get('purge_after')
                if not value:return '자동 삭제 없음'
                try:return datetime.fromisoformat(value).astimezone().strftime('%Y-%m-%d %H:%M')
                except ValueError:return '예정일 확인 필요'
            populate(self.deleted_models,[[m.get('name',m['id']),m['id'],deadline(m)] for m in deleted_models])
            policy=self.state['retention']
            if not hasattr(self,'retention_loaded'):
                self.retention_enabled.setCurrentIndex(self.retention_enabled.findData(policy['enabled']))
                self.retention_days.setValue(policy['days']);self.retention_loaded=True
            self.retention_status.setText(f'현재 적용: {policy["days"]}일 후 영구 삭제' if policy['enabled'] else '현재 적용: 자동 삭제 꺼짐 · 기간을 확인한 뒤 적용하세요.')
            for i,m in enumerate(deleted_models):
                if m['id']==deleted_model_id:self.deleted_models.selectRow(i)
            ready = all(self.state['dependencies'].values())
            self.engine_notice.setText('학습 환경이 설치되어 있습니다. 실행 시 모델·장치 호환성을 확인합니다.' if ready else '학습 환경 설치 필요: install-engine-windows.bat · 데이터 등록과 화면은 지금 사용할 수 있습니다.')
            self.enable_actions(); self.show_job()
            self.update_input_mode()
            if self.infer_id:
                status = next((r for r in runs if r['id'] == self.infer_id), None)
                if status:
                    self.infer_progress.setText(status.get('message', ''))
                    if status['status'] not in RUNNING:
                        finished_id = self.infer_id; self.infer_id = None
                        path = self.service.store.item('runs', finished_id)/'result.json'
                        if path.is_file():
                            self.infer_output = read(path); self.infer_completed_id = finished_id
                            self.infer_comparisons = {}; self.show_current_inference()
                            self.stack.widget(3).ensureWidgetVisible(self.prediction, 0, 35)
                        else: self.infer_summary.setText('추론 결과가 없습니다. 위의 작업 상태를 확인하세요.')
                        self.enable_actions()
        except Exception as e: self.error(str(e))

    def enable_actions(self):
        idle = self.task is None; ready = all(self.state.get('dependencies', {}).values())
        running = self.service.is_running()
        self.actions_waiting = running
        for b in (self.import_button, self.plan_button, self.activate_button, self.advanced_button,
                  self.data_file_button,self.data_folder_button,self.single_data_panel,self.data_name): b.setEnabled(idle)
        for b in (self.folder_button,self.inspect_button,self.load_archive_button): b.setEnabled(idle and not running)
        self.register_button.setEnabled(idle and not running and bool(self.model_check and not self.model_check['missing']))
        self.train_button.setEnabled(idle and ready and not running and self.train_model.count() > 0 and self.dataset_choices.count() > 0)
        self.infer_button.setEnabled(idle and ready and not running and self.infer_model.count() > 0)
        self.activate_button.setEnabled(idle and not running and not self.infer_id)
        for widget in (self.infer_files_button, self.infer_folder_button, self.infer_model, self.infer_device, self.infer_description):
            widget.setEnabled(idle and not running and not self.infer_id)
        self.infer_cancel.setEnabled(idle and running and bool(self.infer_id))
        self.enable_dataset_actions()
        self.enable_model_actions()

    def model_key(self,widget,column=3):
        item=widget.item(widget.currentRow(),column)
        return item.text() if item else None

    def enable_model_actions(self):
        if not hasattr(self,'restore_model_button'):return
        idle=self.task is None and not self.service.is_running()
        self.delete_model_button.setEnabled(idle and bool(self.model_key(self.models)))
        self.save_model_button.setEnabled(idle and bool(self.model_key(self.models)))
        self.restore_model_button.setEnabled(idle and bool(self.model_key(self.deleted_models,1)))
        if hasattr(self,'purge_model_button'):
            self.purge_model_button.setEnabled(idle and bool(self.model_key(self.deleted_models,1)))
            self.retention_save.setEnabled(idle)

    def toggle_model_trash(self):
        show=self.model_trash_panel.isHidden();self.model_trash_panel.setVisible(show)
        self.model_trash_toggle.setText('삭제 가중치 보관함 닫기' if show else '삭제 가중치 보관함 보기')

    def delete_model(self):
        key=self.model_key(self.models)
        model=next((m for m in self.state.get('models',[]) if m['id']==key),None)
        if not model:return
        policy=self.state.get('retention',{})
        expiry=f'{policy["days"]}일 후 실제 파일이 영구 삭제됩니다.' if policy.get('enabled') else '자동 삭제는 꺼져 있습니다. 보관함에서 즉시 영구 삭제할 수 있습니다.'
        message=f'가중치 “{model.get("name",key)}”를 삭제 보관함으로 보낼까요?\n모델 ID: {key}\n학습·추론 선택 목록에서 제외합니다. 영구 삭제 전까지 복원할 수 있습니다.\n{expiry}'
        if QMessageBox.question(self,'가중치 삭제',message,QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return
        self.run_task(lambda:self.service.set_model_deleted(key,True),lambda r:self.message.setText(f'{r["name"]} 가중치를 목록에서 삭제했습니다. 삭제 가중치 보관함에서 복원할 수 있습니다.'))

    def restore_model(self):
        key=self.model_key(self.deleted_models,1)
        if not key:return
        self.run_task(lambda:self.service.set_model_deleted(key,False),lambda r:self.message.setText(f'{r["name"]} 가중치를 복원했습니다. 자동 적용하지는 않습니다.'))

    def save_retention(self):
        enabled=self.retention_enabled.currentData();days=self.retention_days.value()
        message=(f'삭제 보관함의 가중치를 {days}일 후 영구 삭제하도록 설정할까요?\n이미 보관 중인 가중치도 지금부터 {days}일 후로 예정일을 다시 설정합니다.\n모델 폴더와 해당 학습의 임시 가중치를 삭제하며 복원할 수 없습니다. 성능 기록은 남습니다.'
                 if enabled else '자동 영구 삭제를 끌까요? 이미 영구 삭제된 파일은 복원되지 않습니다.')
        if QMessageBox.question(self,'자동 삭제 설정',message,QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return
        self.run_task(lambda:self.service.set_model_retention(enabled,days),lambda r:self.message.setText('가중치 보관 기간 설정을 저장했습니다.'))

    def purge_model(self):
        key=self.model_key(self.deleted_models,1)
        model=next((m for m in self.state.get('deleted_models',[]) if m['id']==key),None)
        if not model:return
        message=f'“{model.get("name",key)}” 가중치를 즉시 영구 삭제할까요?\n모델 ID: {key}\n모델 폴더의 파일과 해당 학습의 임시 가중치가 실제로 삭제되며 복원할 수 없습니다.\n성능 결과·학습 기록·등록 데이터·외부 원본은 남습니다.'
        if QMessageBox.question(self,'복원 불가 · 영구 삭제',message,QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return
        self.run_task(lambda:self.service.purge_model(key),lambda r:self.message.setText(f'{r["name"]} 영구 삭제 완료 · 파일 크기 합계 {r["removed_bytes"]/1024**2:.1f} MiB 삭제. 복원할 수 없습니다.'))

    def cleanup_models(self):
        if self.task is not None or self.service.is_running() or self.pending_close:return
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

    def selected_dataset_id(self,table_widget):
        item=table_widget.item(table_widget.currentRow(),0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def enable_dataset_actions(self):
        if not hasattr(self,'restore_data_button'): return
        idle=self.task is None and not self.service.is_running()
        self.delete_data_button.setEnabled(idle and bool(self.selected_dataset_id(self.datasets)))
        self.restore_data_button.setEnabled(idle and bool(self.selected_dataset_id(self.deleted_datasets)))

    def toggle_dataset_trash(self):
        show=self.trash_panel.isHidden();self.trash_panel.setVisible(show)
        self.trash_toggle.setText('삭제 보관함 닫기' if show else '삭제 보관함 보기')

    def delete_dataset(self):
        key=self.selected_dataset_id(self.datasets)
        dataset=next((d for d in self.state.get('datasets',[]) if d['id']==key),None)
        if not dataset:return
        message=f'등록 데이터 “{dataset["name"]}” ({dataset["count"]}장)을 삭제 보관함으로 보낼까요?\n선택한 등록 묶음 전체가 학습 목록에서 제외됩니다.\n원본 파일·가중치·기존 학습 결과는 그대로 유지되며 복원할 수 있습니다.\n이미 학습된 내용이 가중치에서 제거되는 것은 아닙니다.'
        if QMessageBox.question(self,'등록 데이터 삭제',message,QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return
        self.run_task(lambda:self.service.set_dataset_deleted(key,True),lambda r:self.message.setText(f'{r["name"]} · {r["count"]}장 목록에서 삭제했습니다. 삭제 보관함에서 복원할 수 있습니다.'))

    def restore_dataset(self):
        key=self.selected_dataset_id(self.deleted_datasets)
        if not key:return
        self.run_task(lambda:self.service.set_dataset_deleted(key,False),lambda r:self.message.setText(f'{r["name"]} · {r["count"]}장 복원했습니다.'))

    def run_task(self, fn, callback):
        if self.task is not None: return
        self.callback = callback; self.task = Task(fn, self)
        self.task.succeeded.connect(self.task_success); self.task.failed.connect(self.task_error); self.task.finished.connect(self.task_finished)
        self.message.setText('처리 중입니다. 잠시 기다려 주세요.'); self.enable_actions(); self.task.start()

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

    def settings(self):
        values=dict(epochs=self.epochs.value(),batch=self.batch.currentData(),lr=self.lr.value()) if self.advanced_enabled else dict(TRAIN_DEFAULTS)
        return dict(dataset_ids=[self.dataset_choices.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.dataset_choices.count()) if self.dataset_choices.item(i).checkState() == Qt.CheckState.Checked],
                    model_id=self.train_model.currentData(), name=self.run_name.text(),device=self.train_device.currentData(),**values)

    def toggle_advanced(self):
        self.advanced_enabled=not self.advanced_enabled
        self.advanced_panel.setVisible(self.advanced_enabled)
        self.advanced_button.setText('고급 설정 닫기 · 기본값 사용' if self.advanced_enabled else '고급 설정 열기')
        if not self.advanced_enabled:
            self.epochs.setValue(TRAIN_DEFAULTS['epochs']); self.batch.setCurrentIndex(self.batch.findData(TRAIN_DEFAULTS['batch'])); self.lr.setValue(TRAIN_DEFAULTS['lr'])

    def show_plan(self, plan):
        c = plan['counts']; selection='검증 데이터에서 가장 잘 나온 버전을 선택합니다.' if c['dev'] else '검증 데이터 없음: 마지막 에폭을 저장합니다. 최고 성능·개선 여부는 판단할 수 없습니다.'
        self.plan_label.setText(f'학습 {c["train"]}장 / 검증 {c["dev"]}장 / 평가(테스트) {c["eval"]}장 · 최대 {plan["epochs"]}회 반복 학습\n{selection}\n기존 모델은 보존합니다. 적용은 학습 완료 후 직접 선택합니다.\n기준 모델 평가 이미지 학습 노출: 기록상 {plan["base_eval_seen"]}장')

    def check_plan(self):
        payload = self.settings(); self.run_task(lambda: self.service.store.training_plan(payload), self.show_plan)

    def start_train(self):
        payload = self.settings()
        def confirm(plan):
            self.show_plan(plan)
            if QMessageBox.question(self, '학습 시작', self.plan_label.text() + '\n\n이 설정으로 새 버전을 학습할까요?') == QMessageBox.StandardButton.Yes:
                # Defer until the preflight thread has finished and released the UI task slot.
                self.queued_train = payload
                QTimer.singleShot(100, self.launch_confirmed_train)
        self.run_task(lambda: self.service.store.training_plan(payload), confirm)

    def launch_confirmed_train(self):
        if self.task is not None:
            QTimer.singleShot(100, self.launch_confirmed_train); return
        payload = self.queued_train; self.queued_train = None
        self.run_task(lambda: self.service.launch('train', payload), lambda r: self.message.setText('학습을 시작했습니다. 기록과 새 가중치는 별도로 보관됩니다.'))

    @Slot()
    def show_job(self):
        status = next((r for r in self.state.get('runs', []) if r['id'] == self.job_choice.currentData()), None)
        self.cancel_button.setEnabled(self.task is None and bool(status and status['status'] in RUNNING))
        if not status: self.progress.setValue(0); self.job_text.setText('아직 실행한 학습이 없습니다.'); return
        self.progress.setValue(int(status.get('progress', 0)))
        eta = status.get('eta_seconds')
        eta_text = ('남은 학습 약 ' + str(max(1, round(eta / 60))) + '분') if eta else ('예상시간 측정 중' if status['status'] in RUNNING else '')
        self.job_text.setText(f'{STATE_NAMES.get(status["status"], status["status"])} · {status.get("message", "")}\n{eta_text}')

    def cancel_job(self):
        try: self.service.cancel(self.job_choice.currentData()); self.message.setText('현재 처리 단위가 끝나면 안전하게 중단합니다.')
        except Exception as e: self.error(str(e))

    def import_data(self):
        path, name = self.zip_path.text(), self.data_name.text()
        if not path: self.error('데이터 파일 또는 폴더를 선택해 주세요.'); return
        labels=[self.data_labels.item(i).text() for i in range(self.data_labels.count()) if self.data_labels.item(i).checkState()==Qt.CheckState.Checked]
        split=self.data_split.currentData()
        self.run_task(lambda: self.service.import_data(path, name, labels=labels, split=split), lambda r: self.message.setText(f'{r["count"]}장 등록 완료. 이미지·정답을 프로그램 폴더에 복사 보관했습니다. 원본은 변경하지 않았습니다.'))

    def pick_data_file(self):
        self.pick_file(self.zip_path,'데이터 파일 (*.zip *.csv *.jpg *.jpeg *.png *.webp)')

    def pick_data_folder(self):
        path=QFileDialog.getExistingDirectory(self,'labels.csv가 있는 로컬 폴더 선택',str(self.service.store.root),
            options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try: local_only(path); self.zip_path.setText(path)
            except Exception as e: self.error(str(e))

    def data_path_changed(self,path):
        # A different image must never inherit another image's manually entered GT.
        for i in range(self.data_labels.count()): self.data_labels.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.data_split.setCurrentIndex(0)
        self.single_data_panel.setVisible(Path(path).suffix.lower() in {'.jpg','.jpeg','.png','.webp'})

    def update_data_label_count(self):
        count=sum(self.data_labels.item(i).checkState()==Qt.CheckState.Checked for i in range(self.data_labels.count()))
        self.data_label_count.setText(f'선택한 정답: {count} / 5')

    def save_model_as(self):
        key = self.model_key(self.models)
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

    def load_model_archive(self):
        path, _ = QFileDialog.getOpenFileName(self, '저장한 가중치 ZIP 불러오기',
            str(self.service.store.root.parent), '가중치 백업 ZIP (*.zip)', options=QFileDialog.Option.DontUseNativeDialog)
        if not path: return
        if QMessageBox.question(self, '가중치 불러오기',
            '이 프로그램에서 저장한 신뢰할 수 있는 가중치 ZIP만 선택하세요.\n'
            '파일 무결성과 실제 모델 로딩을 검사한 뒤 보관함에 추가합니다.\n'
            '기존 가중치는 덮어쓰지 않으며 현재 적용 모델도 바꾸지 않습니다. 계속할까요?',
            QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes: return
        def done(result):
            self.message.setText(f'{result["name"]} 불러오기 완료. 보관함에서 선택 후 필요한 경우 모델 적용을 눌러 주세요.')
        self.run_task(lambda: self.service.import_model_archive(path), done)
        self.message.setText('가중치 불러오는 중 · 압축을 풀고 실제 로딩을 검사합니다. 창을 닫지 마세요.')

    def register_model(self):
        if not self.model_check or self.model_check['missing']:
            self.error('먼저 폴더를 선택하고 파일 확인을 해 주세요.'); return
        checked=self.model_check
        if QMessageBox.question(self,'가중치 등록','원본을 보존하고 프로그램 폴더에 복사한 뒤 실제 로딩을 검사합니다.\n수 분 이상 걸릴 수 있습니다. 계속할까요?')!=QMessageBox.StandardButton.Yes: return
        def done(result):
            self.clear_model_check()
            self.message.setText('모델 등록 완료. 로딩 검사는 통과했으며 연구 성능 재현은 별도입니다. 사용할 모델을 선택하고 적용해 주세요.')
        self.run_task(lambda:self.service.import_model(checked),done)

    def clear_model_check(self):
        self.model_check=None
        if hasattr(self,'model_check_text'): self.model_check_text.setText('파일을 확인해 주세요.')
        if hasattr(self,'register_button'): self.register_button.setEnabled(False)

    def pick_model_folder(self):
        path=QFileDialog.getExistingDirectory(self,'로컬 가중치 폴더 선택',str(self.service.store.root),
            options=QFileDialog.Option.DontUseNativeDialog|QFileDialog.Option.ShowDirsOnly|QFileDialog.Option.DontResolveSymlinks)
        if path:
            try: self.model_folder.setText(str(local_only(path)))
            except Exception as e: self.error(str(e))

    def inspect_model(self):
        path=self.model_folder.text().strip()
        if not path: self.error('가중치 폴더를 선택해 주세요.'); return
        self.clear_model_check()
        def done(result):
            self.model_check=result
            if result['missing']: self.model_check_text.setText('부족한 파일:\n'+'\n'.join('• '+x for x in result['missing']))
            else:
                purpose='기존 DINO A/B+RBF · 분류 헤드 추가 학습 가능' if result['engine']=='legacy_dino_rbf_v1' else 'DINO 모델 · 추가 학습 가능'
                self.model_check_text.setText(f'{purpose}\n파일 구성 확인 완료 · {result["bytes"]/1024**2:.1f} MB\n등록 시 텐서 구조와 실제 로딩을 검사합니다. 성능 검증을 뜻하지는 않습니다.')
        self.run_task(lambda:self.service.inspect_model(path),done)

    def activate_model(self):
        item = self.models.item(self.models.currentRow(), 3)
        if not item: self.error('적용할 모델을 선택해 주세요.'); return
        key = item.text()
        model=next((m for m in self.state.get('models',[]) if m['id']==key),{})
        warning='\n이 모델은 별도 평가 성능이 없습니다. 성능 개선이 확인된 모델은 아닙니다.' if model.get('evaluation_f1') is None else '\n평가 조건과 기존 모델 대비 결과를 확인한 뒤 적용하세요.'
        if QMessageBox.question(self, '모델 적용', '선택한 모델을 추론과 다음 추가 학습의 기본 모델로 적용할까요? 이전 가중치는 유지됩니다.'+warning) == QMessageBox.StandardButton.Yes:
            self.run_task(lambda: self.service.activate(key), lambda r: self.message.setText('추론 및 다음 추가 학습의 기본 모델을 변경했습니다.'))

    def save_template(self):
        path, _ = QFileDialog.getSaveFileName(self, '정답 양식 저장', str(self.service.store.root / 'labels.csv'), 'CSV (*.csv)', options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                local_only(path).write_text('id,image,labels,split,description\npattern_001,images/pattern_001.png,' + ';'.join(LABELS[:5]) + ',train,\n', encoding='utf-8-sig')
                self.message.setText('양식 저장 완료. 예시 정답을 실제 GT로 교체해 주세요.')
            except Exception as e: self.error(str(e))

    def reset_inference_results(self):
        self.infer_output = None; self.infer_completed_id = None; self.infer_comparisons = {}
        self.prediction.setRowCount(0);self.infer_summary.setText('선택한 이미지로 추론을 실행해 주세요.')
        self.infer_status.setText('정답은 추론 완료 후 비교에만 사용합니다.')
        self.infer_progress.setText('')

    def single_inference_path_changed(self, path):
        if path: self.set_inference_images([path])

    def set_inference_images(self, paths):
        if self.infer_id: return
        try:
            paths = self.service.inference_paths(paths)
            self.infer_images = [dict(image_path=path, description='') for path in paths]
            self.infer_index = 0; self.reset_inference_results(); self.show_current_inference()
            self.infer_button.setText(f'전체 {len(paths)}장 추론 실행' if len(paths)>1 else '추론 실행')
        except Exception as e: self.error(str(e))

    def pick_inference_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '추론할 이미지 선택', str(self.service.store.root),
            '이미지 (*.png *.jpg *.jpeg *.webp *.PNG *.JPG *.JPEG *.WEBP)', options=QFileDialog.Option.DontUseNativeDialog)
        if paths: self.set_inference_images(paths)

    def pick_inference_folder(self):
        folder = QFileDialog.getExistingDirectory(self, '이미지 폴더 선택 (하위 폴더 제외)', str(self.service.store.root),
            options=QFileDialog.Option.DontUseNativeDialog)
        if folder:
            try: self.set_inference_images(self.service.inference_paths(folder=folder))
            except Exception as e: self.error(str(e))

    def move_inference(self, delta):
        index = self.infer_index + delta
        if 0 <= index < len(self.infer_images):
            self.infer_index = index; self.show_current_inference()

    def inference_description_changed(self):
        if self.infer_images and not self.infer_id:
            self.infer_images[self.infer_index]['description'] = self.infer_description.toPlainText()
            self.reset_inference_results()

    def show_current_inference(self):
        if not self.infer_images: return
        item = self.infer_images[self.infer_index]
        self.image_path.blockSignals(True); self.image_path.setText(item['image_path']); self.image_path.blockSignals(False)
        self.infer_description.blockSignals(True); self.infer_description.setPlainText(item['description']); self.infer_description.blockSignals(False)
        self.infer_position.setText(f'{self.infer_index+1} / {len(self.infer_images)} · {Path(item["image_path"]).name}')
        self.infer_previous.setEnabled(self.infer_index > 0); self.infer_next.setEnabled(self.infer_index+1 < len(self.infer_images))
        path = item['image_path']
        if self.infer_completed_id:
            folder = self.service.store.item('runs', self.infer_completed_id)
            path = folder/'inputs'/f'{self.infer_index:04d}.image' if self.infer_output.get('kind') == 'infer_batch' else folder/'input.image'
        self.preview_image(path)
        self.prediction.setRowCount(0)
        self.infer_status.setText('정답은 추론 완료 후 비교에만 사용합니다.')
        if not self.infer_output:
            self.infer_summary.setText('추론 진행 중입니다.' if self.infer_id else '선택한 이미지로 추론을 실행해 주세요.'); return
        batch = self.infer_output.get('kind') == 'infer_batch'
        output = self.infer_output['items'][self.infer_index] if batch else self.infer_output
        if batch and output['status'] != 'complete':
            self.infer_summary.setText('이 이미지의 추론 중 오류가 발생했습니다.' if output['status']=='failed' else '이 이미지는 아직 추론되지 않았습니다.')
            self.infer_status.setText(output.get('error', '완료된 다른 이미지는 이전·다음으로 확인할 수 있습니다.')); return
        try:
            if self.infer_index not in self.infer_comparisons:
                self.infer_comparisons[self.infer_index] = self.service.inference_comparison(self.infer_completed_id, self.infer_index if batch else None)
            self.show_inference_result(self.infer_comparisons[self.infer_index])
        except Exception as e: self.error(str(e))

    def cancel_inference(self):
        if self.infer_id:
            try:
                self.service.cancel(self.infer_id)
                self.infer_progress.setText('현재 이미지 처리가 끝나면 중단합니다. 완료된 이미지 결과는 보존합니다.')
            except Exception as e: self.error(str(e))

    def preview_image(self, path):
        try:
            from PIL import Image, ImageOps
            path = local_only(path)
            if path.stat().st_size > 16 * 1024**2: raise ValueError('이미지는 16MB 이하로 선택하세요.')
            with Image.open(path) as image:
                if image.width * image.height > 40_000_000: raise ValueError('이미지 크기 제한을 초과했습니다.')
                image = ImageOps.exif_transpose(image).convert('RGB'); image.thumbnail((440, 240))
                qimage = QImage(image.tobytes(), image.width, image.height, image.width * 3, QImage.Format.Format_RGB888).copy()
            self.preview.setPixmap(QPixmap.fromImage(qimage))
        except Exception as e: self.preview.clear(); self.error(str(e))

    def start_infer(self):
        payload = dict(image_path=self.image_path.text(), model_id=self.infer_model.currentData(), device=self.infer_device.currentData())
        model=next((m for m in self.state.get('models',[]) if m['id']==payload['model_id']),{})
        if model.get('engine')=='bert5_fusion_v1':payload['description']=self.infer_description.toPlainText()
        if not payload['image_path']: self.error('이미지를 선택해 주세요.'); return
        if len(self.infer_images) > 1:
            payload.pop('image_path'); payload.pop('description', None)
            payload['images'] = [dict(image_path=i['image_path'], **({'description':i['description']} if model.get('engine')=='bert5_fusion_v1' else {})) for i in self.infer_images]
        def started(result):
            self.reset_inference_results(); self.infer_id = result['id']; self.infer_summary.setText('예측 중입니다…');self.infer_progress.setText('모델을 불러오는 중…'); self.message.setText('추론을 시작했습니다.')
        self.run_task(lambda: self.service.launch('infer', payload), started)

    def update_input_mode(self):
        if not hasattr(self,'description_panel'):return
        model=next((m for m in self.state.get('models',[]) if m['id']==self.infer_model.currentData()),{})
        self.description_panel.setVisible(model.get('engine')=='bert5_fusion_v1')
        if self.input_model_id != self.infer_model.currentData():
            self.input_model_id = self.infer_model.currentData()
            if not self.infer_id: self.reset_inference_results()

    def toggle_score_details(self):
        show=self.prediction.isColumnHidden(3)
        self.prediction.setColumnHidden(3,not show);self.score_note.setVisible(show)
        self.score_details_button.setText('상세 점수 숨기기' if show else '상세 점수 보기')

    def show_inference_result(self,comparison):
        matched=comparison['status']=='matched'; labels=comparison['labels']
        predictions=comparison['result']['top5']
        populate(self.prediction,[[i+1,p['label'],('일치' if p['label'] in labels else '불일치') if matched else '비교 불가',f'{p["score"]:.4f}'] for i,p in enumerate(predictions)])
        if matched:
            self.infer_summary.setText(f'이 이미지: 등록 정답 5개 중 {comparison["hits"]}개 일치\n한 장의 결과이며 모델 전체 성능이 아닙니다.')
            for i,p in enumerate(predictions):
                if p['label'] in labels:
                    for col in (1,2):
                        self.prediction.item(i,col).setBackground(QColor('#e3f3e9'))
                        self.prediction.item(i,col).setForeground(QColor('#185c37'))
            sources=' / '.join(dict.fromkeys(m['dataset_name']+' · '+{'train':'학습용','dev':'검증용','eval':'평가용(테스트)'}.get(m['split'],m['split']) for m in comparison['matches']))
            self.infer_status.setText('등록 정답: '+' · '.join(labels)+'\n비교 자료: '+sources+'\n등록 라벨 기준의 비교입니다. AI 임시 라벨은 사람 검수 정답이 아닙니다. 학습에 사용한 이미지라면 독립 평가가 아닙니다.')
        elif comparison['status']=='conflict':
            self.infer_summary.setText('같은 이미지에 서로 다른 정답이 등록되어 있어 일치율을 계산하지 않았습니다.')
            self.infer_status.setText('데이터 등록 내용을 확인하세요. 예측 결과는 그대로 표시합니다.')
        else:
            self.infer_summary.setText('예측 결과 — 등록된 정답이 없어 성능 계산 불가')
            self.infer_status.setText('감성어 5개는 모델의 예측입니다. 정답과 함께 데이터를 등록한 뒤 다시 추론하면 일치 개수를 확인할 수 있습니다.')

    def load_result(self):
        try:
            key = self.result_choice.currentData()
            if not key: raise ValueError('완료된 학습이 없습니다.')
            result = read(self.service.store.item('runs', key) / 'result.json'); rows = []
            for split, name in [('train', '학습'), ('dev', '검증'), ('eval', '평가(테스트)')]:
                m = result['metrics'][split]; rows.append([name, m['sample_count'], percent(m.get('baseline', {}).get('f1_at_5')), percent(m['f1_at_5']), f'{m["correct_slots"]} / {m["possible_slots"]}'])
            populate(self.metrics, rows); self.result_id = key; self.export_button.setEnabled(True)
            m=result['metrics']['eval'];baseline=m.get('baseline',{}).get('f1_at_5')
            delta='기존 모델과의 비교 없음' if baseline is None or m['f1_at_5'] is None else f'기존 모델 대비 {(m["f1_at_5"]-baseline)*100:+.2f}%p'
            self.evaluation_summary.setText(f'평가 감성어 일치율(F1@5): {percent(m["f1_at_5"])}\n평가 이미지 {m["sample_count"]}장 · {delta}' if m['sample_count'] else '학습 완료 · 별도 평가 데이터 없음\n성능 개선 여부는 판단할 수 없습니다.')
            # Translate historical display text in memory; never rewrite saved results.
            selection_text = result.get('selection', '검증 데이터 기준 선택').replace('개발', '검증').replace('dev F1@5', '검증 F1@5')
            self.result_text.setPlainText(f'저장 에폭: {result["selected_epoch"]}\n{selection_text}\n{result["training_mode"]}\n{result["benchmark_note"]}\n기준 모델 평가 이미지 학습 노출: 기록상 {result["base_eval_seen"]}장\n{result["base_provenance"]}\n재현 검증: {json.dumps(result["checks"], ensure_ascii=False)}')
        except Exception as e: self.error(str(e))

    def export_predictions(self):
        if not self.result_id: return
        path, _ = QFileDialog.getSaveFileName(self, '표본별 예측 저장', str(self.service.store.root / 'predictions.jsonl'), 'JSON Lines (*.jsonl)', options=QFileDialog.Option.DontUseNativeDialog)
        if path:
            try:
                source = self.service.store.item('runs', self.result_id) / 'predictions.jsonl'
                local_only(path).write_bytes(source.read_bytes()); self.message.setText('표본별 예측을 저장했습니다.')
            except Exception as e: self.error(str(e))

    def poll(self):
        if self.service.is_running() or self.infer_id or self.pending_close or getattr(self, 'actions_waiting', False) or any(r['status'] in RUNNING for r in self.state.get('runs', [])):
            self.refresh()
        if self.pending_close and not self.service.is_running() and self.task is None: self.close()

    def closeEvent(self, event):
        if self.task is not None:
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
