"""Check inference image/text pairs against a frozen test-data manifest."""
import hashlib
import json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def check_pairs(records, manifest, data_root):
    manifest,data_root=Path(manifest),Path(data_root).resolve()
    reference=read_rows(manifest)
    expected={r["id"]:r for r in reference}
    ids=[str(r["id"]) for r in records]
    if len(expected)!=len(reference) or len(set(ids))!=len(ids):
        raise ValueError("Duplicate pattern ID in inputs or test-data list")
    if set(ids)!=set(expected):
        raise ValueError("Input IDs do not match the test-data list")
    actual=[]
    for record in records:
        sid=str(record["id"]); ref=expected[sid]
        path=Path(record["image_path"]).resolve()
        if not path.is_relative_to(data_root):
            raise ValueError(f"Image outside data directory: {sid}")
        name=path.relative_to(data_root).as_posix()
        if name!=ref["image_file"]:
            raise ValueError(f"Image filename mismatch: {sid}")
        if not path.is_file():
            raise ValueError(f"Missing image: {sid}")
        if not isinstance(record.get("description"),str) or not record["description"].strip():
            raise ValueError(f"Missing description: {sid}")
        if record["description"]!=ref["description"]:
            raise ValueError(f"Image/description pairing mismatch: {sid}")
        sha=digest(path)
        if sha!=ref["image_sha256"]:
            raise ValueError(f"Image contents changed: {sid}")
        actual.append({"id":sid,"image_file":name,"description":record["description"],"image_sha256":sha})
    audit={"status":"PASS","checked_records":len(actual),"manifest_sha256":digest(manifest),
        "checks":["unique_and_matching_ids","image_filename","image_file_sha256","exact_description"],
        "stage":"before_inference","reference":"frozen annotation-derived validation manifest"}
    return audit,actual

def save_check(output, audit, actual):
    output=Path(output)
    (output/"input_pairs.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in actual),encoding="utf-8")
    audit=dict(audit,input_pairs_sha256=digest(output/"input_pairs.jsonl"))
    (output/"input_check.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")

def verify_saved_check(result, expected_ids, manifest, data_root):
    result=Path(result)
    if not (result/"input_check.json").is_file():
        return None
    saved=json.loads((result/"input_check.json").read_text())
    if saved["status"]!="PASS" or saved["stage"]!="before_inference":
        raise ValueError("Input pairing check did not pass before inference")
    if digest(result/"input_pairs.jsonl")!=saved["input_pairs_sha256"]:
        raise ValueError("Saved inference input list changed")
    if digest(manifest)!=saved["manifest_sha256"]:
        raise ValueError("Test-data list changed since evaluation")
    pairs=read_rows(result/"input_pairs.jsonl")
    if [r["id"] for r in pairs]!=expected_ids:
        raise ValueError("Prediction IDs do not match checked input pairs")
    current,_=check_pairs([{"id":r["id"],"image_path":str(Path(data_root)/r["image_file"]),
        "description":r["description"]} for r in pairs],manifest,data_root)
    if current["checked_records"]!=saved["checked_records"]:
        raise ValueError("Checked input count mismatch")
    return saved
