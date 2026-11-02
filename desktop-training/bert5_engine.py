"""Original five image/text fusion models; local assets only, no caption generation."""
from pathlib import Path
from core import LABELS, read, local_only

ENGINE='bert5_fusion_v1'

def fusion_head(text_dim,image_dim,hidden_dim,num_heads):
    import torch
    class FusionHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_proj=torch.nn.Linear(text_dim,hidden_dim)
            self.image_proj=torch.nn.Linear(image_dim,hidden_dim)
            self.attn=torch.nn.MultiheadAttention(hidden_dim,num_heads,dropout=.2,batch_first=True)
            self.ln=torch.nn.LayerNorm(hidden_dim);self.drop=torch.nn.Dropout(.2)
            self.head=torch.nn.Linear(hidden_dim*2,len(LABELS))
        def forward(self,text,image):
            seq=torch.cat([self.text_proj(text).unsqueeze(1),self.image_proj(image).unsqueeze(1)],dim=1)
            attention,_=self.attn(seq,seq,seq)
            return self.head(self.drop(self.ln(seq+attention).flatten(1)))
    return FusionHead()

class Bert5Engine:
    def __init__(self,store,model_id,device,spec=None):
        import os,torch
        from worker import verify_files
        self.store=store;self.package=store.item('models',model_id)
        self.spec=spec if spec is not None else read(self.package/'manifest.json')
        if self.spec.get('engine')!=ENGINE or self.spec.get('labels')!=LABELS:raise ValueError('5종 BERT 구성 또는 라벨 순서가 다릅니다.')
        verify_files(self.package,self.spec)
        self.base_id=self.spec.get('base_model',model_id);self.base_package=store.item('models',self.base_id)
        if self.base_id!=model_id:
            base=read(self.base_package/'manifest.json')
            if base.get('engine')!=ENGINE or base.get('base_model'):raise ValueError('5종 BERT 원본 참조 오류')
            verify_files(self.base_package,base)
        self.config=read(local_only(self.base_package/'fusion_config.json'))
        if self.config['label_vocab']!=LABELS or len(self.config['models'])!=5:raise ValueError('5종 BERT 설정 오류')
        self.device=torch.device(device);self.torch=torch
        if self.device.type=='cuda' and not torch.cuda.is_available():raise ValueError('NVIDIA GPU를 사용할 수 없습니다.')
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        self.tags=[s['tag'] for s in self.config['models']]
        if set(self.tags)!={'klue','xlmr','kcbert','mbert','kobigbird'}:raise ValueError('5개 모델 구성이 다릅니다.')

    def features(self,images,descriptions):
        import gc,torch
        from transformers import AutoConfig,AutoModel,AutoImageProcessor,AutoTokenizer
        from safetensors.torch import load_file
        from worker import load_checkpoint
        if len(images)!=len(descriptions) or any(not isinstance(t,str) or not t.strip() for t in descriptions):
            raise ValueError('5종 BERT 모델은 이미지마다 설명문이 필요합니다. 이미지 전용 모델을 사용하거나 설명문을 입력하세요.')
        backbone=local_only(self.base_package/'backbone')
        processor=AutoImageProcessor.from_pretrained(str(backbone),local_files_only=True)
        dino=AutoModel.from_pretrained(str(backbone),local_files_only=True,trust_remote_code=False,use_safetensors=True).float().to(self.device).eval()
        with torch.no_grad():
            pixel=processor(images=images,return_tensors='pt')['pixel_values'].to(self.device)
            hid=dino(pixel_values=pixel).last_hidden_state;prefix=1+getattr(dino.config,'num_register_tokens',0)
            image=torch.cat([hid[:,0],hid[:,prefix:].mean(1)],dim=1).float().cpu()
        if image.shape[1]!=self.config['image_dim']:raise ValueError('이미지 특징 크기가 다릅니다.')
        del dino,hid,pixel;gc.collect()
        if self.device.type=='cuda':torch.cuda.empty_cache()
        pieces=[image];heads={};self.text_dims=[]
        override=load_file(str(local_only(self.package/'head.safetensors'))) if self.base_id!=self.package.name else None
        for entry in self.config['models']:
            tag=entry['tag'];text_path=local_only(self.base_package/'text'/tag)
            config=AutoConfig.from_pretrained(str(text_path),local_files_only=True,trust_remote_code=False)
            text=AutoModel.from_config(config,trust_remote_code=False).float()
            state=load_checkpoint(self.base_package/'weights'/(tag+'.pt'))
            text.load_state_dict({k[len('text_backbone.'):]:v for k,v in state.items() if k.startswith('text_backbone.')},strict=True)
            head=fusion_head(config.hidden_size,self.config['image_dim'],self.config['hidden_dim'],self.config['num_heads'])
            head.load_state_dict({k:v for k,v in state.items() if not k.startswith('text_backbone.')},strict=True)
            if override is not None:head.load_state_dict({k[len(tag)+1:]:v for k,v in override.items() if k.startswith(tag+'.')},strict=True)
            del state
            text.to(self.device).eval()
            for p in text.parameters():p.requires_grad_(False)
            tokenizer=AutoTokenizer.from_pretrained(str(text_path),local_files_only=True,trust_remote_code=False)
            tokens=tokenizer(descriptions,padding='max_length',truncation=True,max_length=entry['max_length'],return_tensors='pt').to(self.device)
            with torch.no_grad():
                hidden=text(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask']).last_hidden_state
                mask=tokens['attention_mask'].unsqueeze(-1).float()
                pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
                pieces.append(pooled.float().cpu())
            heads[tag]=head.to(self.device);self.text_dims.append(config.hidden_size)
            del text,tokens,hidden,pooled,mask;gc.collect()
            if self.device.type=='cuda':torch.cuda.empty_cache()
        self.head=torch.nn.ModuleDict(heads).float().to(self.device).eval()
        return torch.cat(pieces,dim=1)

    def training_logits(self,features):
        x=features.to(self.device);offset=self.config['image_dim'];image=x[:,:offset];logits=[]
        for tag,dim in zip(self.tags,self.text_dims):
            logits.append(self.head[tag](x[:,offset:offset+dim],image));offset+=dim
        return logits

    def training_loss(self,features,target):
        return sum(spec['ensemble_weight']*self.torch.nn.functional.binary_cross_entropy_with_logits(logit,target)
                   for spec,logit in zip(self.config['models'],self.training_logits(features)))

    def scores(self,features,batch=64):
        self.head.eval();output=[]
        with self.torch.no_grad():
            for x in features.split(batch):
                scores=sum(spec['ensemble_weight']*logit.sigmoid() for spec,logit in zip(self.config['models'],self.training_logits(x)))
                output.extend(scores.cpu().tolist())
        return output
