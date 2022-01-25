import torch, pandas as pd
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
from torch.utils.data import Dataset



def generate(context='#3000000', device=None, path='../saved_model'):
    tokenizer = BertTokenizer.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path)
    ppl = TextGenerationPipeline(model, tokenizer, device=device)
    model.to(torch.device(f'cuda:{device}' if device is not None else 'cpu'))
    
    gen_text = ppl(context, max_length=200, do_sample=True)
    return gen_text

class TextData(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data.iloc[index]['content']
