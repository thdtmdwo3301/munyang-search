"""Explicit CPU/CUDA release check using bundled real weights; not a benchmark.

Run from the project root: python -m tests.verify_real_continual
Preserves original workspace and records evidence in a separate local workspace.
"""
import json
import shutil
import tempfile
import time
import argparse
import csv,io,zipfile
from pathlib import Path
from app_service import App
from core import read,write,sha
from worker import head_state_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--model',choices=('bundled_image_v1','bundled_bert5_v1'),default='bundled_image_v1');args=parser.parse_args()
    from safetensors.torch import load_file
    import torch
    root=Path(__file__).absolute().parent.parent
    model_id=args.model;fusion=model_id=='bundled_bert5_v1'
    source=root/'workspace/models'/model_id
    before={rel:sha(source/rel) for rel in read(source/'manifest.json')['weight_hashes']}
    evidence=Path(tempfile.mkdtemp(prefix='real_continual_'+model_id+'_'+args.device.replace(':','')+'_',dir=root/'test-output'))
    target=evidence/'workspace/models'/model_id
    shutil.copytree(source,target)
    app=App(evidence/'workspace');app.activate(model_id)
    sample=root/'dist/Synthetic_Pattern_One_WithSuggestedLabels.zip'
    description='네 방향으로 펼쳐진 은행잎과 둥근 곡선이 반복되며 가운데 꽃 모양을 이루는 흑백 선 문양이다.'
    if fusion:
        # Same synthetic input and suggested labels; add a visual description only.
        with zipfile.ZipFile(sample) as archive:
            csv_name=next(n for n in archive.namelist() if n.endswith('labels.csv'))
            reader=csv.DictReader(io.StringIO(archive.read(csv_name).decode('utf-8-sig')));rows=list(reader)
            fields=list(reader.fieldnames)
            if 'description' not in fields:fields.append('description')
            for row in rows:row['description']=description
            buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            sample=evidence/'input_with_description.zip'
            with zipfile.ZipFile(sample,'w') as dest:
                for name in archive.namelist():dest.writestr(name,buf.getvalue().encode('utf-8-sig') if name==csv_name else archive.read(name))
    dataset=app.import_data(sample,'합성 문양 · AI 임시 라벨 · 실행 검증 전용')
    image=root/'dist/Synthetic_Pattern_One/images/SYNTH_GINKGO_20261102_001.png'
    def run(kind,payload):
        started=time.monotonic();run=app.launch(kind,payload)
        print(json.dumps(dict(event='started',kind=kind,run=run['id'])),flush=True)
        code=app.process.wait(timeout=600)
        folder=app.store.item('runs',run['id']);status=read(folder/'status.json')
        if code or status['status']!='complete': raise RuntimeError(str(folder)+': '+str(status))
        return folder,read(folder/'result.json'),time.monotonic()-started
    folder,first,t1=run('train',dict(dataset_ids=[dataset['id']],model_id=model_id,device=args.device,name='실제 가중치 한 장 실행 검증'))
    assert first['checks']['head_weights_changed']
    assert first['selected_epoch']==10 and first['metrics']['dev']['f1_at_5'] is None and first['metrics']['eval']['f1_at_5'] is None
    assert app.snapshot()['active']['model_id']==model_id
    app.activate(first['model_id']); app=App(evidence/'workspace')
    assert app.snapshot()['active']['model_id']==first['model_id']
    infer_folder,infer,t2=run('infer',dict(image_path=str(image),model_id=first['model_id'],device=args.device,**({'description':description} if fusion else {})))
    saved=json.loads((folder/'predictions.jsonl').read_text().splitlines()[0])
    torch.testing.assert_close(torch.tensor(infer['scores']),torch.tensor(saved['scores']),rtol=1e-5,atol=1e-6)
    assert [p['label'] for p in infer['top5']]==saved['top5'] and not infer['GT_input']
    active=app.snapshot()['active']['model_id']
    second_folder,second,t3=run('train',dict(dataset_ids=[dataset['id']],model_id=active,device=args.device,epochs=2,batch=1,name='새 버전 이어서 학습 검증'))
    parent_head=load_file(str(app.store.item('models',first['model_id'])/'head.safetensors'))
    assert read(second_folder/'head_initialization.json')['state_sha256']==head_state_hash(parent_head)
    assert second['checks']['head_weights_changed']
    assert read(app.store.item('models',second['model_id'])/'manifest.json')['parent_model']==first['model_id']
    final_folder,final,t4=run('infer',dict(image_path=str(image),model_id=second['model_id'],device=args.device,**({'description':description} if fusion else {})))
    saved2=json.loads((second_folder/'predictions.jsonl').read_text().splitlines()[0])
    torch.testing.assert_close(torch.tensor(final['scores']),torch.tensor(saved2['scores']),rtol=1e-5,atol=1e-6)
    assert [p['label'] for p in final['top5']]==saved2['top5']
    assert before=={rel:sha(source/rel) for rel in before}
    runtimes=[r['runtime'] for r in (first,infer,second,final)]
    if args.device.startswith('cuda'):assert all(r['device'].startswith('cuda') and r['peak_allocated_bytes']>0 for r in runtimes)
    report=dict(status='passed',device=args.device,base_model=model_id,runtimes=runtimes,synthetic_input=True,labels='AI suggested, not human GT',research_performance_claim=False,
        first_model=first['model_id'],second_model=second['model_id'],first_train_seconds=round(t1,2),first_inference_seconds=round(t2,2),
        second_train_seconds=round(t3,2),second_inference_seconds=round(t4,2),source_weights_unchanged=True,
        exact_parent_head_continuation=True,fresh_process_inference_matches=True,automatic_activation=False,
        first_train_metrics=first['metrics']['train'],evaluation_metrics=first['metrics']['eval'])
    write(evidence/'VERIFICATION.json',report)
    print(json.dumps(dict(evidence=str(evidence),report=report),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
