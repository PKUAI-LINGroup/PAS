from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import os
import argparse
import time

class Parallel(Dataset):
    def __init__(self, directory=None, split=None, tokenizer=None, in_list=False, L=None):
        super(Parallel, self).__init__()
        if in_list:
            self.data = L
        else:
            with open(os.path.join(directory, 'cleaned_' + split + '.txt'), 'r', encoding='utf-8') as f:
                L = f.readlines()
            with open(os.path.join(directory, 'cleaned_' + split + '_labels.txt'), 'r', encoding='utf-8') as f:
                labels = f.readlines()
            self.data = [(tokenizer.encode(L[i], max_length=200, padding='max_length', truncation=True), int(labels[i])-1) for i in range(len(L))]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        x, y = self.data[item]
        return torch.tensor(x), int(y)

class Mono(Dataset):
    def __init__(self, directory, split, tokenizer):
        super(Mono, self).__init__()
        with open(os.path.join(directory, 'cleaned_' + split + '.txt'), 'r', encoding='utf-8') as f:
            L = f.readlines()
        self.data = [tokenizer.encode(line, max_length=200, padding='max_length', truncation=True) for line in L]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        x = self.data[item]
        return torch.tensor(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/mnt/lustrefs/home/wangyh/agnews/')
    args = parser.parse_args()

    start = time.time()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_data = Parallel(args.data_path, 'train', tokenizer)
    unlabel_data = Mono(args.data_path, 'unlabel', tokenizer)
    finetune_data = Parallel(args.data_path, 'finetune', tokenizer)
    print(len(train_data))
    print(train_data[0])
    print(len(unlabel_data))
    print(unlabel_data[0])
    print(len(finetune_data))
    print(finetune_data[0])
    print(f'Loading data takes {time.time()-start} seconds')