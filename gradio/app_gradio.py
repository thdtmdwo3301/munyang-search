"""
전통문양 감성분류 인퍼런스 데모 (val F1@5=0.8000, 5개 모델 앙상블).
이미지 + description을 넣으면 top-5 감성 라벨과 점수를 보여준다.
데이터셋에 있는 이미지(파일명이 원본 그대로인 경우)를 올리면 description을 자동으로 채운다.

의존성: ../inference/(config.json, weights/) 를 그대로 불러와 씀 (가중치 중복 보관 안 함).
description 자동완성 기능은 원본 ETRI 데이터셋(../train/dataset.py 가 참조하는 경로)이
로컬에 있어야 동작함 — 없어도 모델 추론 자체는 정상 동작함(자동완성만 비활성).

사용법:
  python app_gradio.py
"""
import html as _html_lib
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(HERE.parent / "inference"))
from infer import EnsembleModel  # noqa: E402

FONT_DIR = Path("/usr/share/fonts/truetype/nanum")
FONT_REGULAR = str(FONT_DIR / "NanumGothic.ttf")
FONT_BOLD = str(FONT_DIR / "NanumGothicBold.ttf")

PILL_COLORS = {"blue": (66, 133, 244), "green": (52, 168, 83), "red": (234, 67, 53)}

sys.path.insert(0, str(HERE.parent / "train"))
import dataset as D  # noqa: E402

print("모델 로딩 중...")
model = EnsembleModel()
print("로딩 완료.")

print("description/GT라벨 조회용 파일명 매핑 만드는 중...")
_recs = D.load_records()
FILENAME_TO_DESC = {Path(r["image_path"]).name: r["description"] for r in _recs}
FILENAME_TO_EMOTIONS = {Path(r["image_path"]).name: r["emotions"] for r in _recs}
print(f"매핑 {len(FILENAME_TO_DESC)}개 준비 완료.")

PILL_CSS = """
<style>
.pill-box { font-family: inherit; background: #ffffff; border: 1px solid #e5e5e5; border-radius: 10px; padding: 12px 14px; }
.pill-section-title { font-weight: 600; font-size: 13px; margin: 4px 0 8px 0; }
.pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
.pill { display: inline-block; padding: 5px 12px; border-radius: 999px; color: white; font-size: 12px; font-weight: 500; }
.pill-blue { background: #4285F4; }
.pill-green { background: #34A853; }
.pill-red { background: #EA4335; }
.pill-score { font-size: 10px; opacity: 0.85; margin-left: 3px; }
.pill-summary { font-size: 12px; margin-top: 4px; line-height: 1.5; }
</style>
"""


def _pill(text, color, score=None):
    score_html = f'<span class="pill-score">{score:.3f}</span>' if score is not None else ""
    return f'<span class="pill pill-{color}">{text}{score_html}</span>'


def _text_w(draw, txt, font):
    return draw.textbbox((0, 0), txt, font=font)[2]


def _wrap_text(draw, text, font, max_w):
    lines, cur = [], ""
    for ch in text.replace("\n", " "):
        if _text_w(draw, cur + ch, font) > max_w and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def render_description_image(description, width=560):
    """Description 텍스트만 따로 pill-box 스타일 PNG로 렌더링."""
    pad_x = 24
    pad_y = 10
    title_gap = 24
    title_font = ImageFont.truetype(FONT_BOLD, 17)
    desc_font = ImageFont.truetype(FONT_REGULAR, 14)
    desc_line_h = 20
    inner_w = width - 2 * pad_x

    tmp = Image.new("RGB", (10, 10))
    tdraw = ImageDraw.Draw(tmp)
    lines = _wrap_text(tdraw, description, desc_font, inner_w)

    height = pad_y + title_gap + len(lines) * desc_line_h + pad_y
    img = Image.new("RGB", (width, int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([1, 1, width - 2, int(height) - 2], radius=14, outline=(229, 229, 229), width=2)

    y = pad_y
    draw.text((pad_x, y), "Description", font=title_font, fill=(30, 30, 30))
    y += title_gap
    for line in lines:
        draw.text((pad_x, y), line, font=desc_font, fill=(90, 90, 90))
        y += desc_line_h
    return img


def render_result_image(sections, summary=None, width=560):
    """sections: [(title, [(text, color_key), ...]), ...] -> 스크린샷과 동일한 스타일의 PNG(PIL Image)."""
    pad = 24
    title_font = ImageFont.truetype(FONT_BOLD, 17)
    pill_font = ImageFont.truetype(FONT_REGULAR, 16)
    summary_font = ImageFont.truetype(FONT_REGULAR, 16)

    pill_h = 40
    pill_gap_x, pill_gap_y = 12, 12
    section_gap = 22
    inner_w = width - 2 * pad

    tmp = Image.new("RGB", (10, 10))
    tdraw = ImageDraw.Draw(tmp)

    def text_w(txt, font):
        return tdraw.textbbox((0, 0), txt, font=font)[2]

    # 레이아웃 계산 (줄바꿈 포함)
    layout = []  # ('title', text) / ('row', [...]) / ('gap', None) / ('summary', text)
    for title, items in sections:
        layout.append(("title", title))
        row, row_w = [], 0
        for text, color in items:
            w = text_w(text, pill_font) + 36
            if row and row_w + pill_gap_x + w > inner_w:
                layout.append(("row", row))
                row, row_w = [], 0
            row.append((text, color, w))
            row_w += w + pill_gap_x
        if row:
            layout.append(("row", row))
        layout.append(("gap", None))
    if summary:
        layout.append(("summary", summary))

    height = pad
    for kind, val in layout:
        if kind == "title":
            height += 24
        elif kind == "row":
            height += pill_h + pill_gap_y
        elif kind == "gap":
            height += section_gap - pill_gap_y
        elif kind == "summary":
            height += 26
    height += pad

    img = Image.new("RGB", (width, int(height)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([1, 1, width - 2, int(height) - 2], radius=14, outline=(229, 229, 229), width=2)

    y = pad
    for kind, val in layout:
        if kind == "title":
            draw.text((pad, y), val, font=title_font, fill=(30, 30, 30))
            y += 30
        elif kind == "row":
            x = pad
            for text, color, w in val:
                rgb = PILL_COLORS[color]
                draw.rounded_rectangle([x, y, x + w, y + pill_h], radius=pill_h // 2, fill=rgb)
                tw = text_w(text, pill_font)
                draw.text((x + (w - tw) / 2, y + (pill_h - 18) / 2), text, font=pill_font, fill=(255, 255, 255))
                x += w + pill_gap_x
            y += pill_h + pill_gap_y
        elif kind == "gap":
            y += section_gap - pill_gap_y
        elif kind == "summary":
            draw.text((pad, y), val, font=summary_font, fill=(30, 30, 30))
            y += 26
    return img


def autofill_description(image_path):
    """업로드한 이미지 파일명이 데이터셋에 있는 원본이면 description을 자동으로 채워준다."""
    if not image_path:
        return gr.update()
    name = Path(image_path).name
    desc = FILENAME_TO_DESC.get(name)
    if desc is None:
        return gr.update()  # 데이터셋에 없는(새) 이미지면 그대로 둠
    return gr.update(value=desc)


PAGE_CSS = """
<style>
  html, body { margin: 0; padding: 0; background: #f7f7f8;
    font-family: -apple-system, "Malgun Gothic", "Nanum Gothic", sans-serif; }
  .page-wrap { max-width: 420px; margin: 24px auto; padding: 0 16px; }
</style>
"""


def _image_to_data_uri(image_path):
    import base64
    ext = Path(image_path).suffix.lower().lstrip(".") or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:image/{ext};base64,{b64}"


def _pil_to_data_uri(img):
    import base64
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _save_html_file(inner_html, name="result"):
    """예측 결과 HTML을 화면에 보이는 카드와 동일하게(배경/여백 포함) 독립 .html 파일로 저장."""
    doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>예측 결과</title>"
        f"{PAGE_CSS}</head><body><div class='page-wrap'>{inner_html}</div></body></html>"
    )
    path = f"/tmp/_gradio_{name}.html"
    Path(path).write_text(doc, encoding="utf-8")
    return path


def _build_card_html(image_path, description, labels, gt, caption=None):
    """이미지+Description+예측+GT 카드 하나를 HTML(문자열)로 만든다. sections/summary도 같이 반환."""
    img_uri = _image_to_data_uri(image_path)
    html = ['<div class="pill-box">']
    if caption:
        html.append(f'<div class="pill-section-title">{_html_lib.escape(caption)}</div>')
    html.append(f'<img src="{img_uri}" style="max-width:100%;border-radius:8px;margin-bottom:12px;display:block;">')
    html.append('<div class="pill-section-title">Description</div>')
    html.append(f'<div class="pill-summary" style="margin-bottom:10px;">{_html_lib.escape(description)}</div>')
    html.append('<div class="pill-section-title">모델 예측</div>')
    html.append('<div class="pill-row">' + "".join(_pill(l, "blue") for l in labels) + "</div>")

    sections = [("모델 예측", [(l, "blue") for l in labels])]
    summary = None
    if gt:
        pred_set = set(labels)
        hit = sum(1 for g in gt if g in pred_set)
        html.append('<div class="pill-section-title">GT 라벨 (초록=맞춤 / 빨강=틀림)</div>')
        html.append('<div class="pill-row">' + "".join(
            _pill(g, "green" if g in pred_set else "red") for g in gt) + "</div>")
        html.append(f'<div class="pill-summary">정답 수: {hit} / {len(gt)}</div>')
        sections.append(("GT 라벨 (초록=맞춤 / 빨강=틀림)", [
            (g, "green" if g in pred_set else "red") for g in gt]))
        summary = f"정답 수: {hit} / {len(gt)}"
    else:
        html.append('<div class="pill-summary">(데이터셋에 없는 이미지라 GT 라벨 없음)</div>')
    html.append("</div>")
    return "".join(html), sections, summary


def predict(image_path, description):
    if not image_path:
        empty = PILL_CSS + '<div class="pill-box">이미지를 업로드하세요.</div>'
        return empty, None, None, gr.update(value=None), gr.update(value=None)
    if not description or not description.strip():
        empty = PILL_CSS + '<div class="pill-box">description을 입력하세요.</div>'
        return empty, None, None, gr.update(value=None), gr.update(value=None)

    labels, scores = model.predict(image_path, description, topk=5)
    gt = FILENAME_TO_EMOTIONS.get(Path(image_path).name)

    card_html, sections, summary = _build_card_html(image_path, description, labels, gt)
    full_html = PILL_CSS + card_html

    result_image = render_result_image(sections, summary=summary)
    desc_image = render_description_image(description)
    html_path = _save_html_file(full_html)

    desc_img_uri = _pil_to_data_uri(desc_image)
    desc_only_html = f'<img src="{desc_img_uri}" style="max-width:100%;display:block;">'
    desc_html_path = _save_html_file(desc_only_html, name="description")

    return full_html, desc_image, result_image, gr.update(value=html_path), gr.update(value=desc_html_path)


def _save_zip_of_htmls(named_htmls):
    """{파일명(확장자 제외): inner_html} 딕셔너리를 이미지별 개별 .html로 묶은 zip 하나로 저장."""
    import zipfile
    path = "/tmp/_gradio_batch_result.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for stem, inner_html in named_htmls.items():
            doc = (
                "<!DOCTYPE html><html><head><meta charset='utf-8'><title>예측 결과</title>"
                f"{PAGE_CSS}</head><body><div class='page-wrap'>{PILL_CSS}{inner_html}</div></body></html>"
            )
            zf.writestr(f"{stem}.html", doc)
    return path


def batch_predict(files):
    """여러 이미지를 한 번에 예측해서 카드들을 HTML로 바로 이어붙여 보여준다 (데이터셋 이미지만 지원).
    다운로드는 이미지 하나당 html 파일 하나씩, zip으로 묶어서 제공한다."""
    if not files:
        empty = PILL_CSS + '<div class="pill-box">이미지를 올려주세요.</div>'
        return empty, gr.update(value=None)
    cards_html, skipped, named_htmls = [], [], {}
    for f in files:
        path = f if isinstance(f, str) else getattr(f, "name", None)
        if not path:
            continue
        name = Path(path).name
        desc = FILENAME_TO_DESC.get(name)
        if desc is None:
            skipped.append(name)
            continue
        labels, scores = model.predict(path, desc, topk=5)
        gt = FILENAME_TO_EMOTIONS.get(name)
        card_html, _, _ = _build_card_html(path, desc, labels, gt, caption=name)
        cards_html.append(card_html)
        named_htmls[Path(name).stem] = card_html

    zip_path = _save_zip_of_htmls(named_htmls) if named_htmls else None

    out = [PILL_CSS]
    if skipped:
        skipped_str = ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else "")
        out.append(f'<div class="pill-summary" style="margin-bottom:12px;">'
                    f'{len(skipped)}개는 데이터셋에 없는 이미지라 건너뜀: {_html_lib.escape(skipped_str)}</div>')
    out.append(f'<div style="display:flex;flex-direction:column;gap:16px;">{"".join(cards_html)}</div>')
    full_html = "".join(out)
    return full_html, gr.update(value=zip_path)


with gr.Blocks(title="전통문양 감성분류 (F1@5=0.8022)") as demo:
    gr.Markdown(
        "# 전통문양 감성 분류 데모\n"
        "이미지(DINOv3, frozen) + 텍스트 4개 모델(klue-roberta / xlm-roberta / kcbert / klue-bert) "
        "앙상블. val F1@5 = **0.8022**."
    )
    with gr.Tabs():
        with gr.Tab("단일 이미지"):
            gr.Markdown(
                "이미지를 업로드하면 description이 자동으로 채워지고(데이터셋 원본 이미지인 경우) "
                "바로 예측까지 실행됩니다. description을 직접 수정한 뒤 다시 예측하려면 예측 버튼을 누르세요."
            )
            with gr.Row():
                with gr.Column():
                    image_in = gr.Image(type="filepath", label="이미지")
                    desc_in = gr.Textbox(
                        label="Description (문양 설명, 데이터셋 이미지면 자동 채워짐)", lines=6,
                        placeholder="예: 이 문양은 조선시대 사방탁자에 조각된 매화와 새문양 중 일부이다. 자연적인 형태의 매화 나무에...",
                    )
                    btn = gr.Button("예측", variant="primary")
                with gr.Column():
                    out_html = gr.HTML(label="예측 결과")
                    out_html_file = gr.DownloadButton(label="결과 HTML 다운로드 (이미지+Description+예측+GT)", value=None)
                    out_desc_html_file = gr.DownloadButton(label="Description만 HTML 다운로드", value=None)
                    out_desc_image = gr.Image(label="Description 이미지 (다운로드 가능)", type="pil", interactive=False)
                    out_image = gr.Image(label="예측/GT 결과 이미지 (다운로드 가능)", type="pil", interactive=False)

            predict_outputs = [out_html, out_desc_image, out_image, out_html_file, out_desc_html_file]
            image_in.upload(autofill_description, inputs=[image_in], outputs=[desc_in]).then(
                predict, inputs=[image_in, desc_in], outputs=predict_outputs)
            btn.click(predict, inputs=[image_in, desc_in], outputs=predict_outputs)

        with gr.Tab("여러 이미지 확인"):
            gr.Markdown(
                "여러 이미지를 한 번에 올리면 각각 예측해서 카드로 보여줍니다. "
                "**데이터셋에 있는 이미지(파일명이 원본 그대로)만** 지원합니다 — "
                "`data_split/train` 또는 `data_split/val` 안의 이미지를 여러 개 선택해서 올려보세요."
            )
            batch_files = gr.File(label="이미지 여러 개 업로드", file_count="multiple", type="filepath")
            batch_btn = gr.Button("일괄 예측", variant="primary")
            batch_html = gr.HTML(label="예측 결과 (카드가 순서대로 이어져서 표시됨)")
            batch_html_file = gr.DownloadButton(label="이미지별 결과 HTML 모음 (zip) 다운로드", value=None)
            batch_outputs = [batch_html, batch_html_file]
            batch_files.upload(batch_predict, inputs=[batch_files], outputs=batch_outputs)
            batch_btn.click(batch_predict, inputs=[batch_files], outputs=batch_outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
