import torch, random, os, argparse, json
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_ as clip_grad
from transformers import BertTokenizer, GPT2LMHeadModel
from torch.optim import AdamW

import pandas as pd, numpy as np
from utils import TextData, generate
from torch.utils.data import DataLoader

ARG = argparse.ArgumentParser()

ARG.add_argument('--epoch', type=int, default=3, help='Epoch num.')
ARG.add_argument('--seed', type=int, default=98765, help='Random seed.')
ARG.add_argument('--cuda', type=str, default=None, help='Cuda ID.')
ARG.add_argument('--batch', type=int, default=2, help='Training batch size.')
ARG.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
ARG.add_argument('--data', type=str, default='', help='Dataset directory.')
ARG.add_argument('--pretrain_path', type=str, default='../cache_dir', help='Pretrain model path.')
ARG.add_argument('--pretrain_name', type=str, default='uer/gpt2-chinese-cluecorpussmall', help='Pretrain model.')

ARG = ARG.parse_args()


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(_loader, arg):
    device = torch.device('cpu' if arg.cuda is None else 'cuda:' + arg.cuda)

    tokenizer = BertTokenizer.from_pretrained(arg.pretrain_name, cache_dir=arg.pretrain_path)
    model = GPT2LMHeadModel.from_pretrained(arg.pretrain_name, cache_dir=arg.pretrain_path).to(device)
    opt = AdamW(model.parameters(), lr=arg.lr)
    print('Model loaded, start training.')

    process_bar = tqdm(range(len(_loader) * arg.epoch))
    for _ in range(arg.epoch):
        for seqs in _loader:
            input_dict = tokenizer(seqs, padding=True, truncation=True, return_tensors='pt')
            outputs = model(input_dict['input_ids'].to(device), labels=input_dict['input_ids'].to(device))
            _, loss = outputs.logits, outputs.loss

            opt.zero_grad()
            loss.backward()
            clip_grad(model.parameters(), 2.)
            opt.step()
            process_bar.update(1)
        
    model.save_pretrained('../saved_model')
    tokenizer.save_pretrained('../saved_model')
    predict(arg)

def predict(arg):
    res = generate(device=int(arg.cuda))
    with open('res.json', 'w') as f:
        json.dump(res, f)

if __name__ == '__main__':
    seed_torch(ARG.seed)

    trn_data = pd.read_csv('../processed_data/' + ARG.data + 'train.csv').dropna()
    tst_data = pd.read_csv('../processed_data/' + ARG.data + 'test.csv').dropna()
    val_data = pd.read_csv('../processed_data/' + ARG.data + 'val.csv').dropna()

    loader = DataLoader(TextData(trn_data), batch_size=ARG.batch, shuffle=True)
    train(loader, ARG)
