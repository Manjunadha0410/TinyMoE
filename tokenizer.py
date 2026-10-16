from datasets import load_dataset 
from tokenizers import Tokenizer,models,trainers, pre_tokenizers,decoders,normalizers
from transformers import PreTrainedTokenizerFast
import os

def train_and_save_tokenizer(save_dir="./tokenizer", vocab_size=8192):
    dataset = load_dataset('roneneldan/TinyStories',split='train')

    def b_iter(ds,batch_size=10000):
        for i in range(0,len(ds),batch_size):
            yield ds[i:i+batch_size]['text']
            
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) 
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size ,special_tokens=["<unk>","<pad>" , "<bos>", "<eos>"],min_frequency=3)

    print("Training Tokenizer")
    tokenizer.train_from_iterator(b_iter(dataset,10000),trainer= trainer, length= len(dataset))
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(save_dir, "tokenizer.json"),
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    fast_tokenizer.save_pretrained(save_dir)
    return fast_tokenizer

def load_tokenizer(save_dir="./tokenizer"):
    if not os.path.exists(save_dir):
        return train_and_save_tokenizer(save_dir=save_dir)
    return PreTrainedTokenizerFast.from_pretrained(save_dir)