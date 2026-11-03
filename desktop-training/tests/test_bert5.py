"""Tiny five-model functional checks, not research performance."""
import importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from core import Store,LABELS,read,write,inspect_model_folder
from tests.test_core import archive
from app_service import App

AVAILABLE=all(importlib.util.find_spec(x) for x in ('torch','transformers','safetensors'))

@unittest.skipUnless(AVAILABLE,'ML runtime missing')
class Bert5Tests(unittest.TestCase):
    def test_training_reload_and_original_weighting(self):
        import torch
        from PIL import Image
        from transformers import DINOv3ViTConfig,DINOv3ViTModel,DINOv3ViTImageProcessorFast,BertConfig,BertModel,BertTokenizerFast
        from bert5_engine import fusion_head
        from worker import make_engine,execute
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent.parent) as tmp:
            store=Store(Path(tmp)/'workspace');p=store.item('models','bert5');(p/'weights').mkdir(parents=True)
            cfg=DINOv3ViTConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,image_size=32,patch_size=16,num_register_tokens=4)
            DINOv3ViTModel(cfg).save_pretrained(p/'backbone');DINOv3ViTImageProcessorFast(size={'height':32,'width':32}).save_pretrained(p/'backbone')
            models=[]
            for tag,weight in zip(('klue','xlmr','kcbert','mbert','kobigbird'),(.3135,.2099,.2143,.2106,.0518)):
                text=p/'text'/tag;text.mkdir(parents=True)
                config=BertConfig(vocab_size=8,hidden_size=16,num_hidden_layers=1,num_attention_heads=4,intermediate_size=32,max_position_embeddings=32)
                config.save_pretrained(text);(text/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\n')
                BertTokenizerFast(vocab_file=str(text/'vocab.txt')).save_pretrained(text)
                state={'text_backbone.'+k:v for k,v in BertModel(config).state_dict().items()};state.update(fusion_head(16,64,32,4).state_dict())
                torch.save(state,p/'weights'/(tag+'.pt'));models.append(dict(tag=tag,max_length=16,ensemble_weight=weight))
            write(p/'fusion_config.json',dict(label_vocab=LABELS,image_dim=64,hidden_dim=32,num_heads=4,models=models))
            spec=dict(id='bert5',name='tiny image+text',engine='bert5_fusion_v1',labels=LABELS,has_head=True,training_supported=True)
            write(p/'manifest.json',spec);write(p/'package.json',spec)
            self.assertEqual(inspect_model_folder(p)['missing'],[])
            spec['weight_hashes'] = inspect_model_folder(p)['hashes']
            write(p/'manifest.json', spec)
            engine=make_engine(store,'bert5','cpu')
            with self.assertRaisesRegex(ValueError,'설명문'):engine.features([Image.new('RGB',(32,32))],[''])
            features=engine.features([Image.new('RGB',(32,32))],['a'])
            logits=engine.training_logits(features)
            expected=sum(s['ensemble_weight']*l.sigmoid() for s,l in zip(models,logits))
            torch.testing.assert_close(torch.tensor(engine.scores(features)),expected)
            data=store.import_zip(archive(),'test')
            plan=store.training_plan(dict(dataset_ids=[data['id']],model_id='bert5',epochs=2,batch=8,lr=.001))
            run=store.item('runs',plan['id']);run.mkdir();write(run/'config.json',plan);write(run/'status.json',dict(id=plan['id'],status='queued'))
            execute(store.root,plan['id']);result=read(run/'result.json')
            self.assertTrue(result['checks']['description_input']);self.assertFalse(result['checks']['GT_input_to_encoder'])
            self.assertEqual(result['metrics']['eval']['sample_count'],1)
            child=read(store.item('models',result['model_id'])/'manifest.json')
            self.assertEqual(child['base_model'],'bert5');self.assertEqual(child['engine'],'bert5_fusion_v1')
            fresh=make_engine(store,child['id'],'cpu');f=fresh.features([Image.new('RGB',(32,32))],['a'])
            self.assertEqual(len(fresh.scores(f)[0]),22)
            history=read(run/'history.json');best=max(history,key=lambda h:h['dev_f1_at_5'])
            self.assertEqual(result['selected_epoch'],best['epoch'])
            from tests.test_model_archive import assert_real_roundtrip
            assert_real_roundtrip(self, store, child['id'], Image.new('RGB', (32, 32)), 'a')

    def test_missing_description_rejected_before_inference_and_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).absolute().parent.parent) as tmp:
            app=App(Path(tmp)/'workspace');p=app.store.item('models','bert5');p.mkdir()
            write(p/'manifest.json',dict(id='bert5',name='five',engine='bert5_fusion_v1',labels=LABELS,has_head=True))
            rows=[dict(id='x',image='images/x.png',labels=';'.join(LABELS[:5]),split='train',description='')]
            data=app.store.import_zip(archive(rows),'test')
            with self.assertRaisesRegex(ValueError,'설명문'):app.store.training_plan(dict(dataset_ids=[data['id']],model_id='bert5'))
            with patch.object(app,'snapshot',return_value=dict(dependencies={'torch':True})):
                with self.assertRaisesRegex(ValueError,'설명문'):app.launch('infer',dict(model_id='bert5',image_path='not-read.png'))
