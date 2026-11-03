"""Tiny randomly initialized local backbone: functional test, not research performance."""
import importlib.util, io, json, tempfile, unittest
from pathlib import Path
from core import Store,LABELS,write,read
from tests.test_core import archive

AVAILABLE=all(importlib.util.find_spec(x) for x in ('torch','torchvision','transformers','safetensors','numpy'))

@unittest.skipUnless(AVAILABLE,'ML runtime not installed; no real-model validation claimed')
class EngineTests(unittest.TestCase):
    def test_checkpoint_path_metadata_is_inert(self):
        import torch
        from worker import load_checkpoint
        from pathlib import PurePath
        with tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent)) as tmp:
            p=Path(tmp)/'fixture.pt'
            torch.save(dict(state_dict={'x':torch.ones(1)},old_path=Path('/unopened/metadata')),p)
            result=load_checkpoint(p)
            self.assertIsInstance(result['old_path'],PurePath)
            self.assertNotIsInstance(result['old_path'],Path)
            self.assertFalse(hasattr(result['old_path'],'open'))
            torch.testing.assert_close(result['state_dict']['x'],torch.ones(1))

    def test_legacy_equations_and_safe_registration_probe(self):
        import torch
        from PIL import Image
        from transformers import DINOv3ViTConfig,DINOv3ViTModel,DINOv3ViTImageProcessorFast
        from worker import LegacyImageEngine,probe_model
        from core import LEGACY_WEIGHTS
        with tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent)) as tmp:
            torch.manual_seed(42); store=Store(Path(tmp)/'workspace'); p=store.item('models','legacy')
            (p/'backbone').mkdir(parents=True); (p/'weights').mkdir()
            cfg=DINOv3ViTConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,image_size=32,patch_size=16,num_register_tokens=4)
            backbone=DINOv3ViTModel(cfg); backbone.save_pretrained(p/'backbone',safe_serialization=True)
            DINOv3ViTImageProcessorFast(size={'height':32,'width':32}).save_pretrained(p/'backbone')
            for filename in LEGACY_WEIGHTS[:2]:
                head=torch.nn.Sequential(torch.nn.LayerNorm(64),torch.nn.Linear(64,1024),torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.Linear(1024,22))
                state={'backbone.'+k:v for k,v in backbone.state_dict().items()}
                state.update({'head.'+k:v for k,v in head.state_dict().items()})
                torch.save(dict(state_dict=state,vocab=LABELS),p/'weights'/filename)
            centers=torch.nn.functional.normalize(torch.randn(7,64),dim=1); weight=torch.randn(22,7)
            torch.save(dict(architecture='dinov3_rbf_neural_residual',centers=centers,state_dict={'weight':weight},gamma=2.,vocab=LABELS,train_records=7,selected_validation_records=0),p/'weights/image_residual.pt')
            spec=dict(engine='legacy_dino_rbf_v1',labels=LABELS,has_head=True)
            write(p/'package.json',spec)
            result=store.register_model('legacy',lambda manifest:probe_model(store.root,'legacy',manifest))
            self.assertTrue(result['load_check']['synthetic_image_only']); self.assertTrue(result['training_supported'])
            engine=LegacyImageEngine(store,'legacy','cpu')
            image=Image.new('RGB',(280,240),(20,100,200)); actual=engine.features([image])
            # Separately implement the original script's equations for comparison.
            pixels=engine.transform(image).unsqueeze(0); base=backbone.eval(); expected=torch.zeros(1,22)
            with torch.inference_mode():
                for filename,ratio in zip(LEGACY_WEIGHTS[:2],(.46,.54)):
                    checkpoint=torch.load(p/'weights'/filename,map_location='cpu',weights_only=True)
                    base.load_state_dict({k[len('backbone.'):]:v for k,v in checkpoint['state_dict'].items() if k.startswith('backbone.')})
                    head.load_state_dict({k[len('head.'):]:v for k,v in checkpoint['state_dict'].items() if k.startswith('head.')}); head.eval()
                    logits=[]
                    for view in (pixels,pixels.flip(-1),pixels.flip(-2),pixels.flip((-1,-2))):
                        hidden=base(pixel_values=view).last_hidden_state
                        logits.append(head(torch.cat([hidden[:,0],hidden[:,5:].mean(1)],dim=1)))
                    expected+=ratio*torch.sigmoid(torch.stack(logits).mean(0))
                batch=engine.processor(images=[image],return_tensors='pt')['pixel_values']
                hidden=base(pixel_values=batch).last_hidden_state
                feat=torch.nn.functional.normalize(torch.cat([hidden[:,0],hidden[:,5:].mean(1)],dim=1),dim=1)
                expected+=torch.exp(-2*(1-feat@centers.T).clamp(min=0))@weight.T
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
            self.assertEqual(len(engine.scores(actual)[0]),22)
            cached=engine.training_features([image])
            torch.testing.assert_close(torch.tensor(engine.scores(cached)),actual,rtol=1e-5,atol=1e-6)
            # Exercise the same subprocess import/inference path as the desktop UI.
            from app_service import App
            app=App(store.root); imported=app.import_model(app.inspect_model(p))
            self.assertEqual(imported['load_check']['status'],'passed')
            image_path=Path(tmp)/'input.png'; image.save(image_path)
            run=app.launch('infer',dict(model_id=imported['id'],image_path=str(image_path),device='cpu'))
            self.assertEqual(app.process.wait(timeout=90),0)
            output=read(store.item('runs',run['id'])/'result.json')
            self.assertFalse(output['GT_input']); self.assertFalse(output['description_input'])
            self.assertEqual(len(output['top5']),5)
            torch.testing.assert_close(torch.tensor(output['scores']),expected[0],rtol=1e-5,atol=1e-6)
            # Real optimizer steps on existing A/B heads; source weights immutable.
            from worker import execute
            from core import sha
            hashes={n:sha(p/'weights'/n) for n in LEGACY_WEIGHTS}
            dataset=store.import_zip(archive(),'three distinct samples')
            plan=store.training_plan(dict(dataset_ids=[dataset['id']],model_id='legacy',epochs=2,batch=1,lr=.001,device='cpu'))
            train_folder=store.item('runs',plan['id']);train_folder.mkdir()
            write(train_folder/'config.json',plan);write(train_folder/'status.json',dict(id=plan['id'],status='queued',kind='train'))
            execute(store.root,plan['id']);trained=read(train_folder/'result.json')
            self.assertTrue(trained['checks']['head_weights_changed'])
            history=read(train_folder/'history.json');best=max(h['dev_f1_at_5'] for h in history)
            self.assertEqual(trained['selected_epoch'],next(h['epoch'] for h in history if h['dev_f1_at_5']==best))
            self.assertEqual(hashes,{n:sha(p/'weights'/n) for n in LEGACY_WEIGHTS})
            # Fresh child engine must use the newly saved heads, not original heads.
            from safetensors.torch import load_file
            child=LegacyImageEngine(store,trained['model_id'],'cpu')
            saved=load_file(str(store.item('models',trained['model_id'])/'head.safetensors'))
            child_cached=child.training_features([image])
            for k,v in child.head.state_dict().items(): torch.testing.assert_close(v.cpu(),saved[k],rtol=0,atol=0)
            torch.testing.assert_close(torch.tensor(child.scores(child_cached)),child.features([image]),rtol=1e-5,atol=1e-6)
            # Single-image second generation continues child weights, without dev/eval.
            row=dict(id='one',image='images/one.png',labels=';'.join(LABELS[:5]),split='train',description='')
            one=store.import_zip(archive(rows=[row]),'one sample')
            next_plan=store.training_plan(dict(dataset_ids=[one['id']],model_id=trained['model_id'],epochs=2,batch=1,device='cpu'))
            next_folder=store.item('runs',next_plan['id']);next_folder.mkdir()
            write(next_folder/'config.json',next_plan);write(next_folder/'status.json',dict(id=next_plan['id'],status='queued',kind='train'))
            execute(store.root,next_plan['id']);second=read(next_folder/'result.json')
            self.assertEqual(second['selected_epoch'],2);self.assertIsNone(second['metrics']['eval']['f1_at_5'])
            manifest=read(store.item('models',second['model_id'])/'manifest.json')
            self.assertEqual(manifest['parent_model'],trained['model_id']);self.assertEqual(manifest['base_model'],'legacy')
            from tests.test_model_archive import assert_real_roundtrip
            assert_real_roundtrip(self, store, second['model_id'], image)

    def test_local_end_to_end(self):
        import torch
        from transformers import DINOv3ViTConfig,DINOv3ViTModel,DINOv3ViTImageProcessorFast
        from worker import execute
        with tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent)) as tmp:
            store=Store(Path(tmp)/'workspace');p=store.item('models','tiny');(p/'backbone').mkdir(parents=True)
            cfg=DINOv3ViTConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,image_size=32,patch_size=16,num_register_tokens=4)
            DINOv3ViTModel(cfg).save_pretrained(p/'backbone',safe_serialization=True)
            DINOv3ViTImageProcessorFast(size={'height':32,'width':32}).save_pretrained(p/'backbone')
            write(p/'package.json',dict(engine='dino_head_v1',labels=LABELS,has_head=False,name='synthetic test',trained_pixel_hashes=[]))
            store.register_model('tiny');d=store.import_zip(archive(),'synthetic')
            plan=store.training_plan(dict(dataset_ids=[d['id']],model_id='tiny',epochs=1,batch=1,lr=.0001,device='cpu'))
            folder=store.item('runs',plan['id']);folder.mkdir();write(folder/'config.json',plan);write(folder/'status.json',dict(id=plan['id'],status='queued',kind='train'))
            execute(store.root,plan['id']);r=read(folder/'result.json')
            self.assertTrue(r['checks']['fresh_load_top5_equal']);self.assertEqual(r['metrics']['eval']['sample_count'],1)
            self.assertTrue((store.item('models',r['model_id'])/'head.safetensors').exists())
            from tests.test_model_archive import assert_real_roundtrip
            from PIL import Image
            assert_real_roundtrip(self, store, r['model_id'], Image.new('RGB', (32, 32), (20, 50, 80)))

if __name__=='__main__': unittest.main()
