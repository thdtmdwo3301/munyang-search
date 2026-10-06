"""Single configurable model inference; missing modes never use dummy modalities."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoImageProcessor, AutoTokenizer
from PIL import Image
from . import build_model, build_component, load_checkpoint

def main():
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--mode",choices=["image","text","multimodal"],required=True)
    p.add_argument("--image",type=Path)
    p.add_argument("--text")
    p.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    a=p.parse_args()
    required={"image":{"image"},"text":{"text"},"multimodal":{"image","text"}}[a.mode]
    supplied=({"image"} if a.image is not None else set())|({"text"} if a.text is not None else set())
    if supplied!=required: p.error(f"{a.mode} requires exactly {sorted(required)}")
    from .paths import load_model_config
    config=load_model_config(a.config)
    checkpoint=Path(config["checkpoint"])
    if not checkpoint.is_absolute(): checkpoint=a.config.parent/checkpoint
    if not checkpoint.is_file(): p.error(f"Matching trained checkpoint required: {checkpoint}")
    model=build_model(config["model"])
    if model.classifier.required_modalities!=required: p.error("Mode does not match configured classifier")
    load_checkpoint(model,checkpoint,config.get("checkpoint_layout","modular"))
    model.to(a.device).eval(); inputs={}
    feature_encoder = build_component(config.get("feature_image_encoder"))
    if feature_encoder is not None:
        if model.image_encoder is not None or "image" not in required:
            p.error("External image feature encoder conflicts with the selected model/mode")
        feature_encoder.to(a.device).eval()
    if "image" in required:
        processor=AutoImageProcessor.from_pretrained(config["image_processor"],local_files_only=True)
        with Image.open(a.image) as im:
            pixels=processor(images=im.convert("RGB"),return_tensors="pt")["pixel_values"].to(a.device)
        if feature_encoder is None:
            inputs["pixel_values"]=pixels
        else:
            with torch.inference_mode():
                if str(a.device).startswith("cuda"):
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        inputs["image_features"]=feature_encoder(pixels).float()
                else:
                    inputs["image_features"]=feature_encoder(pixels).float()
    if "text" in required:
        tokenizer=AutoTokenizer.from_pretrained(config["tokenizer"],local_files_only=True)
        tokens=tokenizer([a.text],padding="max_length",truncation=True,
                         max_length=config.get("max_length",384),return_tensors="pt")
        inputs.update({k:tokens[k].to(a.device) for k in ("input_ids","attention_mask")})
    with torch.inference_mode():
        if str(a.device).startswith("cuda"):
            with torch.autocast("cuda",dtype=torch.bfloat16): logits=model(**inputs)
        else: logits=model(**inputs)
        probabilities=torch.sigmoid(logits)[0].float()
    vocab=config["label_vocab"]
    if len(vocab)!=len(probabilities) or len(set(vocab))!=len(vocab): raise ValueError("Invalid label vocabulary")
    scores,indices=probabilities.topk(min(5,len(vocab)))
    print(json.dumps({"mode":a.mode,"model_config":str(a.config),"memory":"off",
          "top5":[{"label":vocab[i],"score":float(v)} for i,v in zip(indices.tolist(),scores.tolist())]},
          ensure_ascii=False,indent=2))

if __name__=="__main__": main()
