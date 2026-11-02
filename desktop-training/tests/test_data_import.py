"""Folder/CSV/image adapters must use exactly the existing dataset validation."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from app_service import App
from core import LABELS
from tests.test_core import archive


class DataImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=str(Path(__file__).absolute().parent.parent))
        self.base=Path(self.tmp.name)
        self.app=App(self.base/'workspace')
        self.source=self.base/'source'; self.source.mkdir()
        with zipfile.ZipFile(io.BytesIO(archive())) as z: z.extractall(self.source)
        self.rows=list(csv.DictReader(io.StringIO((self.source/'labels.csv').read_text(encoding='utf-8-sig'))))

    def tearDown(self): self.tmp.cleanup()

    def rewrite(self,rows):
        with (self.source/'labels.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['id','image','labels','split','description']); w.writeheader(); w.writerows(rows)

    def test_folder_csv_zip_same_records(self):
        (self.source/'unrelated.secret').write_text('must not be copied')
        results=[]
        zpath=self.base/'data.zip'; zpath.write_bytes(archive())
        for path in (self.source,self.source/'labels.csv',zpath):
            item=self.app.import_data(str(path),'등록 테스트')
            results.append(self.app.store.rows([item['id']]))
            with zipfile.ZipFile(self.app.store.item('datasets',item['id'])/'original.zip') as z:
                self.assertNotIn('unrelated.secret',z.namelist())
        keys=('id','labels','split','pixel_sha256','description')
        normalize=lambda rows:[{k:r[k] for k in keys} for r in rows]
        self.assertEqual(normalize(results[0]),normalize(results[1]))
        self.assertEqual(normalize(results[0]),normalize(results[2]))

    def test_single_image_manual_gt(self):
        image=self.source/self.rows[0]['image']; raw=image.read_bytes()
        for labels in (None,LABELS[:4],LABELS[:6],[LABELS[0]]*5):
            with self.assertRaisesRegex(ValueError,'5개'): self.app.import_data(image,'한 장',labels=labels)
        item=self.app.import_data(image,'한 장',labels=LABELS[:5],split='eval')
        self.assertEqual(item['counts'],{'eval':1})
        self.assertEqual(self.app.store.rows([item['id']])[0]['labels'],LABELS[:5])
        self.assertEqual(image.read_bytes(),raw)

    def test_missing_csv_and_image(self):
        empty=self.base/'empty'; empty.mkdir()
        with self.assertRaisesRegex(ValueError,'labels.csv'): self.app.import_data(empty,'없음')
        (self.source/self.rows[0]['image']).unlink()
        with self.assertRaisesRegex(ValueError,'찾을 수'): self.app.import_data(self.source,'없음')
        self.assertEqual(self.app.store.list('datasets'),[])

    def test_escape_and_symlink_rejected(self):
        for value in ('../outside.jpg','/tmp/outside.jpg','C:/outside.jpg','..\\outside.jpg'):
            rows=[dict(self.rows[0],image=value)];self.rewrite(rows)
            with self.assertRaises(ValueError): self.app.import_data(self.source,'위험 경로')
        image=self.source/self.rows[0]['image']
        link=self.source/'link.png'; link.symlink_to(image)
        self.rewrite([dict(self.rows[0],image='link.png')])
        with self.assertRaisesRegex(ValueError,'바로가기'): self.app.import_data(self.source,'링크')
        with self.assertRaisesRegex(ValueError,'NAS'): self.app.import_data('/home/mount/NAS_MOUNT/forbidden','NAS')
        self.assertEqual(self.app.store.list('datasets'),[])

    def test_duplicate_and_invalid_gt_rejected(self):
        self.rewrite([self.rows[0],dict(self.rows[0],id='second',split='eval')])
        with self.assertRaisesRegex(ValueError,'중복'): self.app.import_data(self.source,'중복')
        self.rewrite([dict(self.rows[0],labels='강한')])
        with self.assertRaisesRegex(ValueError,'5개'): self.app.import_data(self.source,'잘못된 GT')


if __name__=='__main__': unittest.main()
