from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
import pandas as pd
from utils import TextData
from torch.utils.data import DataLoader
import argparse


ARG = argparse.ArgumentParser()

ARG.add_argument('--epoch', type=int, default=140, help='Epoch num.')
ARG.add_argument('--seed', type=int, default=98765, help='Random seed.')
ARG.add_argument('--batch', type=int, default=1024, help='Training batch size.')
ARG.add_argument('--data', type=str, default='', help='Training batch size.')

ARG = ARG.parse_args()


def train(model_path, _loader):
    tokenizer = BertTokenizer.from_pretrained('uer/gpt2-chinese-cluecorpussmall', cache_dir='../cache_dir')
    model = GPT2LMHeadModel.from_pretrained('uer/gpt2-chinese-cluecorpussmall', cache_dir='../cache_dir')
    print('Model loaded, start training.')
    for seqs in _loader:
        input_dict = tokenizer(seqs, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**input_dict)
        break


if __name__ == '__main__':
    trn_data = pd.read_csv('../processed_data/' + ARG.data + 'train.csv').dropna()
    tst_data = pd.read_csv('../processed_data/' + ARG.data + 'test.csv').dropna()
    val_data = pd.read_csv('../processed_data/' + ARG.data + 'val.csv').dropna()

    loader = DataLoader(TextData(trn_data), batch_size=4, shuffle=True)
    train(0, loader)
