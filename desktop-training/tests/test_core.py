import base64, csv, io, json, os, tempfile, unittest, zipfile
from pathlib import Path
from PIL import Image
from core import Store, LABELS, TRAIN_DEFAULTS, metric, read, write, local_only, inspect_model_folder, LEGACY_WEIGHTS

def model_source(root,legacy=False):
    p=Path(root)/'source'; (p/'backbone').mkdir(parents=True)
    write(p/'backbone/config.json',dict(model_type='dinov3_vit'))
    write(p/'backbone/preprocessor_config.json',{})
    (p/'backbone/model.safetensors').write_bytes(b'synthetic structural fixture, not a model')
    if legacy:
        (p/'runtime_weights').mkdir()
        for name in LEGACY_WEIGHTS: (p/'runtime_weights'/name).write_bytes(b'synthetic fixture')
    return p

def archive(rows=None,extra=None):
    rows=rows or [dict(id=str(i),image=f'images/{i}.png',labels=';'.join(LABELS[:5]),split=s,description='원설명문') for i,s in enumerate(('train','dev','eval'))]
    buf=io.BytesIO(); csvbuf=io.StringIO(); writer=csv.DictWriter(csvbuf,fieldnames=['id','image','labels','split','description']);writer.writeheader();writer.writerows(rows)
    with zipfile.ZipFile(buf,'w') as z:
        z.writestr('labels.csv',csvbuf.getvalue())
        for i,r in enumerate(rows):
            image=io.BytesIO();Image.new('RGB',(12,12),(i*40,10,20)).save(image,format='PNG'); z.writestr(r['image'],image.getvalue())
        if extra: z.writestr(*extra)
    return buf.getvalue()

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent));self.addCleanup(self.tmp.cleanup);self.store=Store(Path(self.tmp.name)/'workspace')
    def test_import_persists(self):
        blob=archive();d=self.store.import_zip(blob,'테스트 자료');s=Store(self.store.root)
        self.assertEqual(d['counts'],dict(train=1,dev=1,eval=1));self.assertEqual(len(s.rows([d['id']])),3)
        self.assertEqual((s.item('datasets',d['id'])/'original.zip').read_bytes(),blob)
    def test_path_traversal(self):
        with self.assertRaises(ValueError): self.store.import_zip(archive(extra=('../outside.txt','bad')),'bad')
        self.assertEqual(self.store.list('datasets'),[])
    def test_invalid_labels(self):
        r=[dict(id='a',image='images/a.png',labels='강한;강한',split='train',description='')]
        with self.assertRaises(ValueError): self.store.import_zip(archive(r),'bad')
    def test_cross_dataset_duplicate(self):
        a=self.store.import_zip(archive(),'a');b=self.store.import_zip(archive(),'b')
        with self.assertRaises(ValueError): self.store.rows([a['id'],b['id']])
    def test_id_traversal(self):
        with self.assertRaises(ValueError): self.store.item('models','../../escape')
    def test_nas_path(self):
        paths=['//NAS/share'] if os.name=='nt' else ['/home/mount/NAS_MOUNT/x','/workspace/mount/x','//NAS/share']
        for p in paths:
            with self.assertRaises(ValueError): local_only(p)
    def test_model_version_not_overwritten(self):
        p=self.store.item('models','existing');p.mkdir();write(p/'manifest.json',{'id':'existing'})
        with self.assertRaises(ValueError): self.store.register_model('existing')
        self.assertEqual(read(p/'manifest.json'),{'id':'existing'})
    def test_metric_and_ties(self):
        rows=[dict(id='a',split='eval',labels=LABELS[:5])]
        result,p=metric([[1]*22],rows)
        self.assertEqual(result['correct_slots'],5);self.assertEqual(p[0]['top5'],LABELS[:5])
        with self.assertRaises(ValueError): metric([[float('nan')]*22],rows)
    def test_plan_preserves_split(self):
        d=self.store.import_zip(archive(),'data');m=self.store.item('models','base');m.mkdir()
        write(m/'manifest.json',dict(id='base',engine='dino_head_v1',labels=LABELS,trained_pixel_hashes=[],has_head=False))
        p=self.store.training_plan(dict(dataset_ids=[d['id']],model_id='base',epochs=5,batch=8,lr=.0001))
        self.assertEqual([r['split'] for r in p['rows']],['train','dev','eval']);self.assertEqual(p['base_eval_seen'],0)
    def test_missing_training_model_rejected(self):
        rows=[dict(id='a',image='images/a.png',labels=';'.join(LABELS[:5]),split='train',description='')]
        d=self.store.import_zip(archive(rows),'data')
        with self.assertRaises((ValueError,FileNotFoundError)): self.store.training_plan(dict(dataset_ids=[d['id']],model_id='base'))

    def test_model_folder_inspection_and_missing(self):
        p=model_source(self.tmp.name,True); result=inspect_model_folder(p)
        self.assertEqual(result['engine'],'legacy_dino_rbf_v1'); self.assertEqual(result['missing'],[])
        self.assertEqual(len(result['files']),6)
        (p/'runtime_weights'/LEGACY_WEIGHTS[0]).unlink()
        self.assertIn(LEGACY_WEIGHTS[0],inspect_model_folder(p)['missing'])

    def test_partial_rbf_explains_backbone(self):
        p=Path(self.tmp.name)/'partial'; p.mkdir(); (p/'image_residual.pt').write_bytes(b'fixture')
        result=inspect_model_folder(p)
        self.assertIn(LEGACY_WEIGHTS[0],result['missing'])
        self.assertTrue(any('config.json' in name for name in result['missing']))

    def test_model_shards_and_unsafe_paths(self):
        p=model_source(self.tmp.name)
        write(p/'backbone/model.safetensors.index.json',dict(weight_map={'x':'part-1.safetensors'}))
        self.assertIn('backbone/part-1.safetensors',inspect_model_folder(p)['missing'])
        write(p/'backbone/model.safetensors.index.json',dict(weight_map={'x':'../../outside.safetensors'}))
        with self.assertRaises(ValueError): inspect_model_folder(p)

    @unittest.skipIf(os.name=='nt','symlink setup may need privileges')
    def test_model_symlink_rejected(self):
        p=model_source(self.tmp.name); (p/'package.json').symlink_to('/home/mount/NAS_MOUNT/package.json')
        with self.assertRaises(ValueError): inspect_model_folder(p)

    def test_legacy_training_plan_and_single_image(self):
        d=self.store.import_zip(archive(),'data'); p=self.store.item('models','base'); p.mkdir()
        write(p/'manifest.json',dict(id='base',engine='legacy_dino_rbf_v1'))
        plan=self.store.training_plan(dict(dataset_ids=[d['id']],model_id='base'))
        self.assertEqual(plan['engine'],'legacy_dino_rbf_v1');self.assertEqual(plan['selection'],'best_dev_f1')
        row=dict(id='single',image='images/single.png',labels=';'.join(LABELS[:5]),split='train',description='')
        single=self.store.import_zip(archive(rows=[row]),'single')
        plan=self.store.training_plan(dict(dataset_ids=[single['id']],model_id='base'))
        self.assertEqual(plan['selection'],'last_epoch_no_dev');self.assertFalse(plan['evaluation_available'])

if __name__=='__main__': unittest.main()
