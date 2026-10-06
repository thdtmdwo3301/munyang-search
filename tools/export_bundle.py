"""Build code-only and USB bundles; never include venv, caches or credentials."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={"venv",".venv",".cache","__pycache__",".git","outputs"}

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(8*1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--code-only",action="store_true",help="Export code and small runtime support files without weights or images")
    a=p.parse_args()
    output=a.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Export outside the project directory")
    output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for path in sorted(ROOT.rglob("*")):
        rel=path.relative_to(ROOT)
        if any(part in EXCLUDE for part in rel.parts) or path.suffix==".pyc" or not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError(f"Do not export symlinks: {rel}")
        paths.append(path)
    support=lambda rel: rel.parts[:2]==("assets","text") or str(rel) in {"data/evaluation_manifest.jsonl","data/evaluation_manifest_info.json"}
    if a.code_only:
        paths=[path for path in paths if path.relative_to(ROOT).parts[0] not in {"assets","data"} or support(path.relative_to(ROOT))]
    manifest={str(path.relative_to(ROOT)):{"size":path.stat().st_size,"sha256":sha256(path)}
        for path in paths if path.relative_to(ROOT).parts[0] in {"assets","data"}}
    if a.code_only:
        existing=json.loads((ROOT/"ASSET_MANIFEST.json").read_text())
        existing.update(manifest)
        manifest=existing
    (ROOT/"ASSET_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    paths=[p for p in paths if p.name!="ASSET_MANIFEST.json"]+[ROOT/"ASSET_MANIFEST.json"]
    exported=[]
    bundles=(("pattern-search-code.zip",False),) if a.code_only else (("pattern-search-code.zip",False),("pattern-search-usb.zip",True))
    for name,include_assets in bundles:
        target=output/name
        if target.exists():
            raise FileExistsError(target)
        with zipfile.ZipFile(target,"x",allowZip64=True) as z:
            for path in paths:
                rel=path.relative_to(ROOT)
                if not include_assets and rel.parts[0] in {"assets","data"} and not support(rel):
                    continue
                compress=zipfile.ZIP_STORED if path.suffix in {".pt",".safetensors",".npz",".jpg",".png"} else zipfile.ZIP_DEFLATED
                z.write(path,"pattern-search/"+str(rel),compress_type=compress)
        exported.append({"file":name,"size":target.stat().st_size,"sha256":sha256(target)})
    (output/"SHA256.json").write_text(json.dumps(exported,indent=2))
    print(json.dumps(exported,indent=2))

if __name__=="__main__":main()
