import torch
from torch.utils.data import DataLoader,IterableDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset

class TokenizedDataset(IterableDataset):
    def __init__(self,hf_dataset,tokenizer,max_length,stride):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
    def __iter__(self):
        token_buffer = []
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None :
            data_stream = self.hf_dataset.shard(
                num_shards=worker_info.num_workers,
                index = worker_info.id)
        else:
            data_stream = self.hf_dataset
            
        for sample in data_stream:
            tokens = self.tokenizer.encode(sample["text"], add_special_tokens = False )
            
            if bos_id is not None:
                tokens = [bos_id]+tokens
            if eos_id is not None:
                tokens = tokens + [eos_id] 
        
            token_buffer.extend(tokens)
            
            while len(token_buffer) >= self.max_length + 1 :
                in_ids =token_buffer[:self.max_length]
                tar_ids = token_buffer[1:self.max_length+1]
                yield  (torch.tensor(in_ids,dtype=torch.long),torch.tensor(tar_ids,dtype=torch.long))
                token_buffer = token_buffer[self.stride:]
            
        
def dataloader_stream(b_sz,max_l,stride, tokenizer_dir = "./tokenizer"):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    dataset = TokenizedDataset(ds,tokenizer, max_l ,stride )
    dataloader = DataLoader(dataset,batch_size= b_sz, drop_last =True, num_workers=0,pin_memory=True)
    
    
    return dataloader, len(tokenizer)


        

