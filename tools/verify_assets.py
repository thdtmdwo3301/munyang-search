"""Verify transported assets using the bundled SHA-256 manifest."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/"ASSET_MANIFEST.json").read_text())
    total=0
    for name,record in manifest.items():
        path=ROOT/name
        if not path.is_file() or path.stat().st_size!=record["size"]:
            raise RuntimeError(f"Missing or incorrect asset: {name}")
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
        if h.hexdigest()!=record["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {name}")
        total+=1
    print(f"파일 무결성: {total:,}/{total:,} 정상")

if __name__=="__main__":main()
