from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch
import os
import argparse
def rpad(array, n=70):
    """Right padding."""
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)


'''def get_binary_label(label):
    if label < 2:
        return 0
    if label > 2:
        return 1
    raise ValueError("Invalid label")

class SST(Dataset):
    def __init__(self, directory, split, tokenizer, root=True, binary=True):
        super().__init__()
        assert split in ['train', 'dev', 'test']
        self.sst = pytreebank.load_sst(directory)[split]
        if root and binary:
            self.data = [(rpad(tokenizer.encode(tree.to_lines()[0]), n=100), get_binary_label(tree.label)) for tree in self.sst if tree.label != 2]
        elif root and not binary:
            self.data = [
                (rpad(tokenizer.encode(tree.to_lines()[0]), n=100), tree.label) for
                tree in self.sst]
        elif not root and not binary:
            self.data = [
                (rpad(tokenizer.encode(line), n=100), label)
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
            ]
        else:
            self.data = [
                (
                    rpad(tokenizer.encode(line), n=100),
                    get_binary_label(label),
                )
                for tree in self.sst
                for label, line in tree.to_labeled_lines()
                if label != 2
            ]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x, y = self.data[item]
        x = torch.tensor(x)
        return x, y'''

'''class Parallel(Dataset):
    def __init__(self, directory, split, tokenizer):
        super(Parallel, self).__init__()
        with open(os.path.join(directory, split+'.csv'), 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            l = [line for line in r]
        self.data = [(rpad(tokenizer.encode(line[0]), n=100), line[1]) for line in l]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        x, y = self.data[item]
        return torch.tensor(x), int(y)

class Mono(Dataset):
    def __init__(self, directory, split, tokenizer):
        super(Mono, self).__init__()
        with open(os.path.join(directory, split+'.csv'), 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            l = [line for line in r]
        self.data = [rpad(tokenizer.encode(line[0]), n=100) for line in l]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        x = self.data[item]
        return torch.tensor(x)'''

class Parallel(Dataset):
    def __init__(self, directory=None, split=None, tokenizer=None, in_list=False, L=None):
        super(Parallel, self).__init__()
        if in_list:
            self.data = L
        else:
            with open(os.path.join(directory, 'simple_' + split+'.txt'), 'r', encoding='utf-8') as f:
                L = f.readlines()
                L = [line.strip().split('\t') for line in L]
            self.data = [(tokenizer.encode(line[1], max_length=512, padding='max_length', truncation=True), int(line[0])-1) for line in L]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        x, y = self.data[item]
        return torch.tensor(x), int(y)

class Mono(Dataset):
    def __init__(self, directory, split, tokenizer):
        super(Mono, self).__init__()
        with open(os.path.join(directory, 'simple_' + split+'.txt'), 'r', encoding='utf-8') as f:
            L = f.readlines()
            L = [line.strip().split('\t') for line in L]
        self.data = [tokenizer.encode(line[1], max_length=512, padding='max_length', truncation=True) for line in L]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        x = self.data[item]
        return torch.tensor(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home1/wangyh/sst/')
    parser.add_argument('--root', type=bool, default=True)
    parser.add_argument('--binary', type=bool, default=False)
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_data = Mono(args.data_path, 'mono', tokenizer)
    print(train_data[0])