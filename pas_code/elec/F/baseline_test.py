import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, BertModel
from torch import nn
import torch.nn.functional as F
import numpy as np
import argparse
from Data import Parallel, Mono
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, f1_score

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def test(model, test_loader):
    model.eval()
    Labels = []
    Predicted = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.cuda()
            labels = labels.cuda()
            logits = model(data)['logits']
            predicted = torch.max(logits, 1)[1]
            Predicted += predicted.tolist()
            Labels += labels.tolist()
    print(confusion_matrix(Labels, Predicted))
    print(classification_report(Labels, Predicted, labels=[0, 1], digits=4))
    print(f1_score(Labels, Predicted, average='micro'))


def main(args):
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-medium')
    devset = Parallel(args.data_dir, "test", tokenizer)

    dev_loader = torch.utils.data.DataLoader(devset, batch_size=args.lbsz, shuffle=False)

    config = BertConfig.from_pretrained('prajjwal1/bert-medium')
    config.num_labels = 2
    model = BertForSequenceClassification(config=config).cuda()
    model.load_state_dict(torch.load(args.checkpoints_dir+f'supervised_weights'))
    test(model, dev_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home1/wangyh/elec/')
    parser.add_argument('--checkpoints_dir', type=str, default='/home1/wangyh/elec/baseline/checkpoints/')
    parser.add_argument('--lbsz', type=int, default=64)
    parser.add_argument('--ubsz', type=int, default=64)
    args = parser.parse_args()
    main(args)