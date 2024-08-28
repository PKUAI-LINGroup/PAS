from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import os
import csv
import argparse

class Parallel(Dataset):
    def __init__(self, directory=None, split=None, tokenizer=None, in_list=False, L=None):
        super(Parallel, self).__init__()
        if in_list:
            self.data = L
        else:
            with open(os.path.join(directory, 'simple_' + split+'.tsv'), 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                lines = []
                for line in reader:
                    if line[2] == 'contradiction':
                        label = 0
                    elif line[2] == 'neutral':
                        label = 1
                    elif line[2] == 'entailment':
                        label = 2
                    lines.append([line[0] + '[SEP]' + line[1], label])
            self.data = [(tokenizer.encode(line[0], max_length=70, padding='max_length', truncation=True), line[1]) for line in lines]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        x, y = self.data[item]
        return torch.tensor(x), int(y)

def main(args):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_set = Parallel(directory=args.data_path, split='train', tokenizer=tokenizer)
    len_list = [len(line[0]) for line in train_set]
    len_list.sort()
    print(train_set[0])
    print(train_set[1])
    print(train_set[2])
    print(len_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home1/wangyh/SNLI/')
    args = parser.parse_args()
    main(args)