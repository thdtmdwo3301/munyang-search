"""Create the initial fixed evaluation manifest from original annotations."""
import json
from pathlib import Path
from train import dataset
from evct.input_pairs import digest
ROOT=Path(__file__).resolve().parents[1]

def main():
    data=ROOT/"data"
    dest=data/"evaluation_manifest.jsonl"
    if dest.exists():
        raise FileExistsError("Existing test-data list will not be overwritten")
    dataset.ANN=data/"annotations";dataset.IMG=data/"images";dataset.ANNDB=data/"annotations-db"
    records=dataset.load_records()
    _,validation=dataset.split_records(records,seed=42)
    assert len(records)==4400 and len(validation)==446
    rows=[{"id":str(r["id"]),"image_file":Path(r["image_path"]).relative_to(data).as_posix(),
           "description":r["description"],"image_sha256":digest(r["image_path"])} for r in validation]
    dest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    (data/"evaluation_manifest_info.json").write_text(json.dumps({
        "source":"original annotations; fixed validation split seed=42",
        "records":446,"independently_supplied_external_list":False,
        "manifest_sha256":digest(dest)},ensure_ascii=False,indent=2))
    print("시험자료 목록: 446건")
if __name__=="__main__":main()
