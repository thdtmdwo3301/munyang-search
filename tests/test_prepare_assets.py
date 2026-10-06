import hashlib
import tempfile
import unittest
from pathlib import Path
from tools.prepare_assets import restore_split_weight, is_lfs_pointer

class SplitWeightTests(unittest.TestCase):
    def make_parts(self, root):
        chunks = [b"first-weight-data", b"second-weight-data"]
        record = {"file": "xlmr.pt", "size": sum(map(len, chunks)),
                  "sha256": hashlib.sha256(b"".join(chunks)).hexdigest(), "parts": []}
        for i, chunk in enumerate(chunks, 1):
            name = f"xlmr.pt.part{i}"
            (root/name).write_bytes(chunk)
            record["parts"].append({"file": name, "size": len(chunk),
                                    "sha256": hashlib.sha256(chunk).hexdigest()})
        return record

    def test_restore_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); record=self.make_parts(root)
            self.assertTrue(restore_split_weight(root, record))
            self.assertEqual(hashlib.sha256((root/"xlmr.pt").read_bytes()).hexdigest(), record["sha256"])
            self.assertFalse(restore_split_weight(root, record))

    def test_corruption_does_not_publish_partial_weight(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); record=self.make_parts(root)
            f=root/"xlmr.pt.part2";f.write_bytes(b"x"*f.stat().st_size)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                restore_split_weight(root, record)
            self.assertFalse((root/"xlmr.pt").exists())
            self.assertEqual(list(root.glob("*.restore")), [])

    def test_lfs_pointer_requires_download(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); record=self.make_parts(root)
            f=root/"xlmr.pt.part1";f.write_text("version https://git-lfs.github.com/spec/v1\n")
            self.assertTrue(is_lfs_pointer(f))
            with self.assertRaisesRegex(FileNotFoundError, "git lfs pull"):
                restore_split_weight(root, record)
