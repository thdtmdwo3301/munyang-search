"""
이미지에 대한 description(묘사) 텍스트만 입력받아서 저장하는 가벼운 도구.
모델 추론 없음 - GPU 필요 없음.

사용법:
  python collect_description.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr

OUT_PATH = Path(__file__).resolve().parent / "collected_descriptions.jsonl"


def save_description(image, description):
    if image is None:
        return "이미지를 업로드하세요.", None
    if not description or not description.strip():
        return "description을 입력하세요.", None

    ts = datetime.now(timezone.utc).isoformat()
    img_path = f"/tmp/_collected_{ts.replace(':', '-')}.jpg"
    image.save(img_path)

    row = {"timestamp": ts, "image_path": img_path, "description": description.strip()}
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = sum(1 for _ in open(OUT_PATH, encoding="utf-8")) if OUT_PATH.exists() else 0
    return f"저장 완료 (누적 {total}건) → {OUT_PATH}", ""


with gr.Blocks(title="Description 수집") as demo:
    gr.Markdown(
        "# 이미지 Description 수집\n"
        "이미지를 업로드하고 묘사(description) 텍스트만 입력해서 저장합니다. "
        "모델 예측은 하지 않습니다.\n\n"
        f"저장 위치: `{OUT_PATH}`"
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="이미지")
            desc_in = gr.Textbox(label="Description", lines=6, placeholder="이 이미지에 대한 설명을 입력하세요...")
            btn = gr.Button("저장", variant="primary")
        with gr.Column():
            status = gr.Textbox(label="상태", interactive=False)

    btn.click(save_description, inputs=[image_in, desc_in], outputs=[status, desc_in])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
