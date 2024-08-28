'''import re
split_pattern = re.compile(r'([.,!?":;)(|$#\\])')
with open('D:/repository/ssl_text_classification/data/elec/train.txt', 'r', encoding='utf-8') as f:
    L = f.readlines()
    L1 = [split_pattern.sub('', sentence) for sentence in L]
with open('D:/repository/ssl_text_classification/data/elec/simple_train.txt', 'w', encoding='utf-8') as f1:
    f1.writelines(L1)

with open('D:/repository/ssl_text_classification/data/elec/test.txt', 'r', encoding='utf-8') as f:
    L = f.readlines()
    L1 = [split_pattern.sub('', sentence) for sentence in L]
with open('D:/repository/ssl_text_classification/data/elec/simple_test.txt', 'w', encoding='utf-8') as f1:
    f1.writelines(L1)

with open('D:/repository/ssl_text_classification/data/elec/unlabel.txt', 'r', encoding='utf-8') as f:
    L = f.readlines()
    L1 = [split_pattern.sub('', sentence) for sentence in L]
with open('D:/repository/ssl_text_classification/data/elec/simple_unlabel.txt', 'w', encoding='utf-8') as f1:
    f1.writelines(L1)'''

from transformers import BertTokenizer
from Data import Parallel, Mono
from tqdm import tqdm
import argparse
def main(args):
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-medium')
    #trainset = Parallel(args.data_dir, "train", tokenizer)
    #devset = Parallel(args.data_dir, "test", tokenizer)
    restset = Parallel(args.data_dir, "unlabel", tokenizer)
    train_max_len = 0
    count200 = 0
    count300 = 0
    count400 = 0
    count500 = 0
    '''for item in tqdm(trainset):
        if len(item[0]) > train_max_len:
            train_max_len = len(item[0])
        if len(item[0]) > 200:
            count200 += 1
        if len(item[0]) > 300:
            count300 += 1
        if len(item[0]) > 400:
            count400 += 1
        if len(item[0]) > 500:
            count500 += 1
    print(f'train_max_len: {train_max_len}')
    print(f'count200: {count200} count300: {count300} count400: {count400} count500: {count500}')'''
    '''dev_max_len = 0
    for item in tqdm(devset):
        if len(item[0]) > dev_max_len:
            dev_max_len = len(item[0])
        if len(item[0]) > 200:
            count200 += 1
        if len(item[0]) > 300:
            count300 += 1
        if len(item[0]) > 400:
            count400 += 1
        if len(item[0]) > 500:
            count500 += 1
    print(f'dev_max_len: {dev_max_len}')
    print(f'count200: {count200} count300: {count300} count400: {count400} count500: {count500}')'''
    rest_max_len = 0
    for item in tqdm(restset):
        if len(item[0]) > rest_max_len:
            dev_max_len = len(item[0])
        if len(item[0]) > 200:
            count200 += 1
        if len(item[0]) > 300:
            count300 += 1
        if len(item[0]) > 400:
            count400 += 1
        if len(item[0]) > 500:
            count500 += 1
    print(f'rest_max_len: {rest_max_len}')
    print(f'count200: {count200} count300: {count300} count400: {count400} count500: {count500}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home1/wangyh/elec/')
    args = parser.parse_args()
    main(args=args)