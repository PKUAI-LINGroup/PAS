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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
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
            labeled_loss = criterion(logits.view(-1, model.num_labels), y_batch.view(-1))

            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
            running_loss += labeled_loss.item()

        test_acc, test_loss = evaluate(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.checkpoints_dir + f'supervised_weights')
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
    if len(confidence) >= 5000:
        pivot = sorted(confidence, reverse=True)[4999]
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
    print(classification_report(Labels, Predicted, labels=[0, 1], digits=4))
    print(f1_score(Labels, Predicted, average='micro'))

class Classifier(nn.Module):
    def __init__(self, num_labels):
        super(Classifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.liner = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids):
        output = self.bert(input_ids)
        output = self.dropout(output.get('pooler_output'))
        output = self.liner(output)

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    '''num_train_samples = 1000
    samples_per_class = int(num_train_samples / 9)

    x = pd.read_csv(args.data_dir+'train.csv')
    y = x['label']
    x.drop(['label'], inplace=True, axis=1)

    x_test = pd.read_csv(args.data_dir+'dev.csv')
    y_test = x_test['label']
    x_test.drop(['label'], inplace=True, axis=1)

    x_train, x_unlabeled = x[y.values == 0].values[:samples_per_class], x[y.values == 0].values[samples_per_class:]
    y_train = y[y.values == 0].values[:samples_per_class]

    for i in range(1, 10):
        x_train = np.concatenate([x_train, x[y.values == i].values[:samples_per_class]], axis=0)
        y_train = np.concatenate([y_train, y[y.values == i].values[:samples_per_class]], axis=0)

        x_unlabeled = np.concatenate([x_unlabeled, x[y.values == i].values[samples_per_class:]], axis=0)
    normalizer = Normalizer()
    x_train = normalizer.fit_transform(x_train)
    x_unlabeled = normalizer.transform(x_unlabeled)
    x_test = normalizer.transform(x_test.values)

    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)

    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test.values).type(torch.LongTensor)

    train = torch.utils.data.TensorDataset(x_train, y_train)
    test = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train, batch_size=TRAIN_BS, shuffle=True, num_workers=8)

    unlabeled_train = torch.from_numpy(x_unlabeled).type(torch.FloatTensor)

    unlabeled = torch.utils.data.TensorDataset(unlabeled_train)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled, batch_size=UNLABELED_BS, shuffle=True, num_workers=8)

    test_loader = torch.utils.data.DataLoader(test, batch_size=TEST_BS, shuffle=True, num_workers=8)'''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    trainset = Parallel(args.data_dir, "train", tokenizer)
    devset = Parallel(args.data_dir, "test", tokenizer)
    restset = Parallel(args.data_dir, "unlabel", tokenizer)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.lbsz, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(devset, batch_size=args.lbsz, shuffle=False)
    rest_loader = torch.utils.data.DataLoader(restset, batch_size=args.ubsz, shuffle=True)
    writer = SummaryWriter()
    #config = BertConfig.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('prajjwal1/bert-medium')
    config.num_labels = 2
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config).cuda()
    #model = BertForSequenceClassification(config=config).cuda()

    #config = BertConfig.from_pretrained('bert-base-uncased')
    #config.num_labels = 4
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', config=config).cuda()
    #train_supervised(model, train_loader, dev_loader, args, writer, iteration=100)
    #train_supervised(model, train_loader, dev_loader, args, writer)
    #model.load_state_dict(torch.load(args.save_dir+'10000_supervised_weights'))
    #pseudo_labeled_data, rest_data = predict(model, rest_loader)
    train_data = [(item[0].tolist(), item[1]) for item in trainset]
    rest_data = [(item[0].tolist(), item[1]) for item in restset]
    print(len(train_data), len(rest_data))
    model = BertForSequenceClassification(config=config).cuda()
    train_supervised(model, train_loader, dev_loader, args, writer)
    #Iteration = 1
    #model.load_state_dict(torch.load(args.save_dir + f'iteration{Iteration - 1}_supervised_weights'))
    #test(model, dev_loader)
    #Iteration=10
    #model.load_state_dict(torch.load(args.checkpoints_dir+f'iteration9_supervised_weights'))
    #test(model, dev_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='AGNews')
    parser.add_argument('--data_dir', type=str, default='/home1/wangyh/elec/')
    parser.add_argument('--supervised_epoch', type=int, default=30)
    parser.add_argument('--checkpoints_dir', type=str, default='/home1/wangyh/elec/baseline/checkpoints/')
    parser.add_argument('--save_dir', type=str, default='/home1/wangyh/elec/baseline/iter_data/')
    parser.add_argument('--lbsz', type=int, default=16)
    parser.add_argument('--ubsz', type=int, default=16)
    args = parser.parse_args()
    main(args)