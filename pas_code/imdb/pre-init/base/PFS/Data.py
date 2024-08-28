from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import os
import argparse

class Parallel(Dataset):
    def __init__(self, directory=None, split=None, tokenizer=None, in_list=False, L=None):
        super(Parallel, self).__init__()
        if in_list:
            self.data = L
        else:
            with open(os.path.join(directory, 'simple_' + split+'.txt'), 'r', encoding='utf-8') as f:
                L = f.readlines()
                L = [line.strip().split('\t') for line in L]
            self.data = [(tokenizer.encode(line[1], max_length=200, padding='max_length', truncation=True), int(line[0])) for line in L]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        x, y = self.data[item]
        return torch.tensor(x), int(y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home1/wangyh/sst/')
    parser.add_argument('--root', type=bool, default=True)
    parser.add_argument('--binary', type=bool, default=False)
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_data = Mono(args.data_path, 'mono', tokenizer)
    print(train_data[0])