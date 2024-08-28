import csv
import json

with open('D:/repository/ssl_text_classification/data/SNLI/train.jsonl', 'r') as f:
    L0 = []
    L1 = []
    train_num = 10000
    unlabel_num = 50000
    for line in f.readlines():
        data = json.loads(line)
        if train_num > 0:
            train_num -= 1
            L0.append([data['sentence1'], data['sentence2'], data['gold_label']])
        elif unlabel_num > 0:
            unlabel_num -= 1
            L1.append([data['sentence1'], data['sentence2'], data['gold_label']])
        else:
            break
with open('D:/repository/ssl_text_classification/data/SNLI/simple_train.tsv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(L0)
with open('D:/repository/ssl_text_classification/data/SNLI/simple_unlabel.tsv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(L1)
with open('D:/repository/ssl_text_classification/data/SNLI/dev.jsonl', 'r') as f:
    L = []
    for line in f.readlines():
        data = json.loads(line)
        L.append([data['sentence1'], data['sentence2'], data['gold_label']])
with open('D:/repository/ssl_text_classification/data/SNLI/simple_dev.tsv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(L)
with open('D:/repository/ssl_text_classification/data/SNLI/test.jsonl', 'r') as f:
    L = []
    for line in f.readlines():
        data = json.loads(line)
        L.append([data['sentence1'], data['sentence2'], data['gold_label']])
with open('D:/repository/ssl_text_classification/data/SNLI/simple_test.tsv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(L)
'''with open('D:/repository/ssl_text_classification/data/SNLI/train.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    skip_num = 1
    train_num = 10000
    unlabel_num = 50000
    L1 = []
    L2 = []
    for line in reader:
        if skip_num > 0:
            skip_num -= 1
        elif train_num > 0:
            train_num -= 1
            L1.append(line[1:])
        elif unlabel_num > 0:
            unlabel_num -= 1
            L2.append(line[1:])
        else:
            break
with open('D:/repository/ssl_text_classification/data/SNLI/simple_train.tsv', 'w', encoding='utf-8', newline='') as f1:
    writer = csv.writer(f1, delimiter='\t')
    writer.writerows(L1)
with open('D:/repository/ssl_text_classification/data/SNLI/simple_unlabel.tsv', 'w', encoding='utf-8', newline='') as f2:
    writer = csv.writer(f2, delimiter='\t')
    writer.writerows(L2)
print(len(L1), len(L2))'''