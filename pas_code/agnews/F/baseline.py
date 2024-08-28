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

def train_supervised(model, train_loader, test_loader, args, writer, num_labels):
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    EPOCHS = args.supervised_epoch
    best_epoch = 0
    best_acc = 0.0
    model.train()
    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch}')
        running_loss = 0
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            output = model(X_batch, labels=y_batch)
            logits = output['logits']
            labeled_loss = criterion(logits.view(-1, num_labels), y_batch.view(-1))

            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
            running_loss += labeled_loss.item()

        test_acc, test_loss = evaluate(model, test_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.checkpoints_dir + 'supervised_weights')
            best_epoch = epoch
        writer.add_scalar(f'labeled_train_loss', running_loss/len(train_loader), epoch)
        writer.add_scalar(f'labeled_dev_acc', test_acc, epoch)
        writer.add_scalar(f'labeled_dev_loss', test_loss, epoch)
        writer.add_scalar(f'labeled_best_epoch', best_epoch, epoch)
        model.train()

def studentLossFn(teacher_pred, student_pred, y, T, alpha):
		"""
		Loss function for student network: Loss = alpha * (distillation loss with soft-target) + (1 - alpha) * (cross-entropy loss with true label)
		Return: loss
		"""
		if (alpha > 0):
			loss = F.kl_div(F.log_softmax(student_pred / T, dim=1), F.softmax(teacher_pred / T, dim=1), reduction='batchmean') * (T ** 2) * alpha + F.cross_entropy(student_pred, y) * (1 - alpha)
		else:
			loss = F.cross_entropy(student_pred, y)
		return loss

def soft_train_supervised(teacher_model, student_model, train_loader, dev_loader, args, writer):
    torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
    EPOCHS = args.supervised_epoch
    best_epoch = 0
    best_acc = 0.0
    student_model.train()
    for epoch in range(EPOCHS):
        print(f'EPOCH: {epoch}')
        running_loss = 0
        for X_batch, y_batch in tqdm(train_loader):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            student_logits = student_model(X_batch, labels=y_batch)['logits']
            with torch.no_grad():
                teacher_logits = teacher_model(X_batch, labels=y_batch)['logits']

            labeled_loss = studentLossFn(teacher_logits, student_logits, y_batch, T=5, alpha=0.5)

            optimizer.zero_grad()
            labeled_loss.backward()
            optimizer.step()
            running_loss += labeled_loss.item()

        test_acc, test_loss = evaluate(student_model, dev_loader)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student_model.state_dict(), args.checkpoints_dir + f'supervised_weights')
            best_epoch = epoch
        writer.add_scalar(f'labeled_train_loss', running_loss/len(train_loader), epoch)
        writer.add_scalar(f'labeled_dev_acc', test_acc, epoch)
        writer.add_scalar(f'labeled_dev_loss', test_loss, epoch)
        writer.add_scalar(f'labeled_best_epoch', best_epoch, epoch)
        student_model.train()

def generate_soft_labels(model, train_loader):
    model.eval()
    Predicted = []
    with torch.no_grad():
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.cuda()
            logits = model(x_batch)['logits']
            Predicted += logits.tolist()
    return Predicted

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)

    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-medium')
    trainset = Parallel(args.data_dir, "train", tokenizer)
    devset = Parallel(args.data_dir, "test", tokenizer)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.lbsz, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(devset, batch_size=args.lbsz, shuffle=False)
    writer = SummaryWriter()
    config = BertConfig.from_pretrained('prajjwal1/bert-medium')
    config.num_labels = 4

    model = BertForSequenceClassification(config=config).cuda()
    train_supervised(model, train_loader, dev_loader, args, writer, 4)



    #teacher_model = BertForSequenceClassification(config=config).cuda()
    #student_model = BertForSequenceClassification(config=config).cuda()
    #teacher_model.load_state_dict(torch.load('/home1/wangyh/agnews/self/checkpoints/iteration0_supervised_weights'))
    #soft_train_data = np.load(args.save_dir+'train_data.npy', allow_pickle=True).tolist()
    #soft_trainset = Parallel(in_list=True, L=soft_train_data, soft=True)
    #soft_train_supervised(teacher_model, student_model, train_loader, dev_loader, args, writer)








    # model.load_state_dict(torch.load('/home1/wangyh/agnews/self/checkpoints/iteration0_supervised_weights'))
    # soft_labels = generate_soft_labels(model, train_loader)
    # train_data = [(trainset[i][0].tolist(), soft_labels[i]) for i in range(len(trainset))]
    # np.save(args.save_dir + f'train_data', train_data)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='AGNews')
    parser.add_argument('--data_dir', type=str, default='/home1/wangyh/agnews/')
    parser.add_argument('--supervised_epoch', type=int, default=10)
    parser.add_argument('--checkpoints_dir', type=str, default='/home1/wangyh/agnews/baseline/checkpoints/')
    parser.add_argument('--save_dir', type=str, default='/home1/wangyh/agnews/baseline/iter_data/')
    parser.add_argument('--lbsz', type=int, default=32)
    parser.add_argument('--ubsz', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)