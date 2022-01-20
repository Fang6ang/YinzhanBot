import torch
from torch.nn.utils import clip_grad_norm_ as clip_grad
from transformers import BertTokenizer, GPT2LMHeadModel, AdamW
import pandas as pd
from utils import TextData
from torch.utils.data import DataLoader
import argparse


ARG = argparse.ArgumentParser()

ARG.add_argument('--epoch', type=int, default=140, help='Epoch num.')
ARG.add_argument('--seed', type=int, default=98765, help='Random seed.')
ARG.add_argument('--cuda', type=str, default=None, help='Cuda ID.')
ARG.add_argument('--batch', type=int, default=1024, help='Training batch size.')
ARG.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
ARG.add_argument('--data', type=str, default='', help='Dataset directory.')
ARG.add_argument('--pretrain_path', type=str, default='../cache_dir', help='Pretrain model path.')
ARG.add_argument('--pretrain_name', type=str, default='uer/gpt2-chinese-cluecorpussmall', help='Pretrain model.')

ARG = ARG.parse_args()


def train(_loader, arg):
    device = torch.device('cpu' if arg.cuda is None else 'cuda:' + arg.cuda)

    tokenizer = BertTokenizer.from_pretrained(arg.pretrain_name, cache_dir=arg.pretrain_path)
    model = GPT2LMHeadModel.from_pretrained(arg.pretrain_name, cache_dir=arg.pretrain_path).to(device)
    opt = AdamW(model.parameters(), lr=arg.lr)
    print('Model loaded, start training.')

    for seqs in _loader:
        input_dict = tokenizer(seqs, padding=True, truncation=True, return_tensors='pt')
        outputs = model(input_dict['input_ids'], labels=input_dict['input_ids'])
        _, loss = outputs.logits, outputs.loss

        opt.zero_grad()
        loss.backward()
        clip_grad(model.parameters(), 2.)
        opt.step()
        print(loss.item())
        break


if __name__ == '__main__':
    trn_data = pd.read_csv('../processed_data/' + ARG.data + 'train.csv').dropna()
    tst_data = pd.read_csv('../processed_data/' + ARG.data + 'test.csv').dropna()
    val_data = pd.read_csv('../processed_data/' + ARG.data + 'val.csv').dropna()

    loader = DataLoader(TextData(trn_data), batch_size=ARG.batch, shuffle=True)
    train(loader, ARG)
