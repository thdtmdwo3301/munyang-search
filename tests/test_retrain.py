import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from evct import build_model, load_checkpoint
from evct.retrain import configure_training, training_mode, save_weights, validate_records

class TinyImage(nn.Module):
    output_dim=6
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,6),nn.Dropout(.5))
    def forward(self,pixel_values):
        return self.net(pixel_values)

class TinyText(nn.Module):
    output_dim=6
    def __init__(self):
        super().__init__()
        self.net=nn.Embedding(12,6)
    def forward(self,input_ids,attention_mask):
        return (self.net(input_ids)*attention_mask.unsqueeze(-1)).sum(1)/attention_mask.sum(1,keepdim=True).clamp_min(1)

class RetrainTests(unittest.TestCase):
    def test_train_save_reload_all_modalities(self):
        for mode in ("image","text","multimodal"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / ".cache/tmp") as td:
                spec={"classifier":{"type":{"image":"image_mlp","text":"text_mlp","multimodal":"attention_fusion"}[mode],
                    "labels":22,"hidden_dim":8,"dropout":.1}}
                inputs={}
                if mode!="text":
                    spec["image_encoder"]={"type":"tests.test_retrain:TinyImage"}
                    inputs["pixel_values"]=torch.randn(3,4)
                if mode!="image":
                    spec["text_encoder"]={"type":"tests.test_retrain:TinyText"}
                    inputs.update(input_ids=torch.randint(0,12,(3,5)),attention_mask=torch.ones(3,5))
                if mode=="multimodal": spec["classifier"]["num_heads"]=2
                model=build_model(spec)
                params=configure_training(model,freeze_image=True)
                before={k:v.clone() for k,v in model.state_dict().items()}
                training_mode(model)
                if model.image_encoder: self.assertFalse(model.image_encoder.training)
                optim=torch.optim.AdamW(params,lr=.01)
                loss=nn.functional.binary_cross_entropy_with_logits(model(**inputs),torch.zeros(3,22))
                loss.backward(); optim.step()
                self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items() if k.startswith("classifier.")))
                if model.image_encoder:
                    self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items() if k.startswith("image_encoder.")))
                model.eval(); expected=model(**inputs).detach()
                path=Path(td)/"model.pt"; save_weights(model,path)
                reloaded=build_model(spec); load_checkpoint(reloaded,path); reloaded.eval()
                torch.testing.assert_close(reloaded(**inputs),expected,rtol=0,atol=0)

    def test_reject_overlap_and_invalid_ground_truth(self):
        labels=list("abcdef")
        def row(i): return {"id":i,"emotions":labels[:5],"description":"text"}
        validate_records([row("train")],[row("val")],labels,{"text"})
        with self.assertRaisesRegex(ValueError,"overlap"):
            validate_records([row("same")],[row("same")],labels,{"text"})
        bad=row("bad"); bad["emotions"]=["a"]*5
        with self.assertRaisesRegex(ValueError,"GT"):
            validate_records([bad],[row("val")],labels,{"text"})

if __name__=="__main__": unittest.main()
