import json
import tempfile
import unittest
from pathlib import Path
from evct.paths import load_model_config, portable_model_config
from evct.runtime import ROOT, configure

class PortableTests(unittest.TestCase):
    def setUp(self):
        configure()

    def test_model_paths_follow_configuration_not_working_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".cache/tmp") as temp:
            base=Path(temp)
            (base/"configs").mkdir()
            (base/"models").mkdir()
            config={"checkpoint":"../models/weights.pt", "image_processor":"../models",
                    "model":{"image_encoder":{"type":"dino","model_name":"../models"},
                             "classifier":{"type":"image_mlp","labels":22}}}
            path=base/"configs/example.json"
            path.write_text(json.dumps(config))
            resolved=load_model_config(path)
            self.assertEqual(resolved["model"]["image_encoder"]["model_name"],str(base/"models"))
            exported=portable_model_config(resolved,base/"trained")
            self.assertEqual(exported["image_processor"],"../models")
            self.assertEqual(exported["checkpoint"],"../models/weights.pt")

    def test_report_rejects_inconsistent_metrics(self):
        from evct.report import render
        with tempfile.TemporaryDirectory(dir=ROOT/".cache/tmp") as temp:
            base=Path(temp); (base/"data/annotations").mkdir(parents=True)
            result=base/"result";result.mkdir()
            ann={"identifier":"sample","contents":{"image":{"file_name":"sample.png"},"emotion":list("abcde")}}
            (base/"data/annotations/sample.json").write_text(json.dumps(ann))
            (result/"results.json").write_text(json.dumps({"validation_records":1,
                "memory_records":0,"image_f1_at_5":1.0,"image_correct_labels":5}))
            row={"id":"sample","predicted_top5":list("abcdf"),"scores":[.9,.8,.7,.6,.5],
                 "correct_count":4,"memory_applied":False}
            (result/"image_predictions.jsonl").write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError,"Metric"):
                render(result,base/"data")

if __name__=="__main__": unittest.main()
