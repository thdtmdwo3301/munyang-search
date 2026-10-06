import json,tempfile,unittest
from pathlib import Path
from evct.input_pairs import check_pairs,digest,save_check,verify_saved_check
from evct.runtime import ROOT,configure

class InputPairTests(unittest.TestCase):
    def setUp(self):
        configure()
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/".cache/tmp")
        self.root=Path(self.temp.name);(self.root/"images").mkdir()
        self.rows=[]
        for sid,text in (("a","first image description"),("b","second image description")):
            p=self.root/"images"/(sid+".jpg");p.write_bytes(sid.encode())
            self.rows.append({"id":sid,"image_file":"images/"+sid+".jpg","description":text,"image_sha256":digest(p)})
        self.manifest=self.root/"manifest.jsonl"
        self.manifest.write_text("".join(json.dumps(r)+"\n" for r in self.rows))
        self.actual=[{"id":r["id"],"image_path":str(self.root/r["image_file"]),"description":r["description"]} for r in self.rows]
    def tearDown(self):self.temp.cleanup()
    def test_correct_pairs_and_saved_check(self):
        audit,actual=check_pairs(self.actual,self.manifest,self.root)
        save_check(self.root,audit,actual)
        self.assertEqual(verify_saved_check(self.root,["a","b"],self.manifest,self.root)["checked_records"],2)
    def test_swapped_descriptions_are_rejected(self):
        self.actual[0]["description"],self.actual[1]["description"]=self.actual[1]["description"],self.actual[0]["description"]
        with self.assertRaisesRegex(ValueError,"pairing mismatch"):check_pairs(self.actual,self.manifest,self.root)
    def test_duplicate_ids_are_rejected(self):
        self.actual[1]["id"]="a"
        with self.assertRaisesRegex(ValueError,"Duplicate"):check_pairs(self.actual,self.manifest,self.root)
    def test_changed_image_is_rejected(self):
        (self.root/"images/a.jpg").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"contents changed"):check_pairs(self.actual,self.manifest,self.root)
if __name__=="__main__":unittest.main()
