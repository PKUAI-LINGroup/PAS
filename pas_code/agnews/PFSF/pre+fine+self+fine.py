import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer, BertModel
from torch import nn
import torch.nn.functional as F
import numpy as np
import argparse
from Data import Parallel
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix, f1_score

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from tqdm import tqdm

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    loss = 0
    total_num = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            data = data.cuda()
            labels = labels.cuda()
            logits = model(data)['logits']
            err = loss_fn(logits, labels)
            loss += err.item()
            predicted = torch.max(logits, 1)[1]
            correct += (predicted == labels.cuda()).sum()
            total_num += data.shape[0]

    return (float(correct)/total_num) *100, (loss/len(test_loader))


def train_supervised(model, train_loader, test_loader, args, writer):
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = args.supervised_epoch
    best_epoch = 0
    best_acc = 0.0
    model.train()
    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch}')
        total_num = 0
        running_loss = 0
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            output = model(X_batch, labels=y_batch)
            #labeled_loss = output['loss']
            logits = output['logits']
            total_num += X_batch.shape[0]
            labeled_loss = criterion(logits.view(-1, model.module.num_labels), y_batch.view(-1))

            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
            running_loss += labeled_loss.item()

        test_acc, test_loss = evaluate(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.checkpoints_dir + 'finetune_weights')
            best_epoch = epoch
        writer.add_scalar(f'labeled_train_loss', running_loss/len(train_loader), epoch)
        writer.add_scalar(f'labeled_dev_acc', test_acc, epoch)
        writer.add_scalar(f'labeled_dev_loss', test_loss, epoch)
        writer.add_scalar(f'labeled_best_epoch', best_epoch, epoch)
        model.train()

def predict(model, unlabeled_loader):
    torch.cuda.empty_cache()
    x_data = []
    y_data = []
    pseudo_labels = []
    confidence = []
    for x_unlabeled, y_unlabeled in tqdm(unlabeled_loader):
        # Forward Pass to get the pseudo labels.
        x_unlabeled = x_unlabeled.cuda()
        model.eval()
        unlabeled_logits = model(x_unlabeled)['logits']
        softmax = nn.Softmax(dim=-1)
        unlabeled_logits = softmax(unlabeled_logits)
        value, pseudo_labeled = torch.max(unlabeled_logits, 1)
        x_data = x_data + x_unlabeled.tolist()
        y_data = y_data + y_unlabeled.tolist()
        pseudo_labels = pseudo_labels + pseudo_labeled.tolist()
        confidence = confidence + value.tolist()
    if len(confidence) >= 10000:
        pivot = sorted(confidence, reverse=True)[9999]
    else:
        pivot = sorted(confidence, reverse=True)[-1]
    pseudo_labeled_data = []
    rest_data = []
    for i in range(len(confidence)):
        if confidence[i] >= pivot:
            pseudo_labeled_data.append((x_data[i], pseudo_labels[i]))
        else:
            rest_data.append((x_data[i], y_data[i]))
    return pseudo_labeled_data, rest_data

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
    print(classification_report(Labels, Predicted, labels=[0, 1, 2, 3], digits=4))
    print(f1_score(Labels, Predicted, average='micro'))

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
 
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-medium')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    trainset = Parallel(args.data_dir, "train", tokenizer)
    devset = Parallel(args.data_dir, "test", tokenizer)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.lbsz, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(devset, batch_size=args.lbsz, shuffle=False)
    writer = SummaryWriter()
    
    config = BertConfig.from_pretrained('prajjwal1/bert-medium')
    config.num_labels = 4
    

    model = BertForSequenceClassification(config=config).cuda()
    model = nn.DataParallel(model)
    Iteration = 10
    model.load_state_dict(torch.load('/mnt/lustrefs/home/wangyh/home1/agnews/pre+fine+self/CL_checkpoints/' + f'iteration{Iteration}_supervised_weights'))
    train_supervised(model, train_loader, dev_loader, args, writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/lustrefs/home/wangyh/home1/agnews/')
    parser.add_argument('--supervised_epoch', type=int, default=10)
    parser.add_argument('--checkpoints_dir', type=str, default='/mnt/lustrefs/home/wangyh/home1/agnews/pre+fine+self+fine/checkpoints/')
    parser.add_argument('--save_dir', type=str, default='/mnt/lustrefs/home/wangyh/home1/agnews/pre+fine+self+fine/iter_data/')
    parser.add_argument('--lbsz', type=int, default=32)
    parser.add_argument('--ubsz', type=int, default=32)
    args = parser.parse_args()
    main(args)