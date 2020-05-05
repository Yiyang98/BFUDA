from __future__ import print_function
import random
import os
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import model
import numpy as np
from data_loader import *
import argparse, time, os
import torch.nn.functional as F
import ResNet as models

seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def train(args, model, train_loader, train_loader1, optimizer, epoch):
    model.train()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source
    else:
        num_iter = len_target
    
    for batch_idx in range(num_iter):
        if batch_idx % len_source == 0:
            iter_source = iter(train_loader)    
        if batch_idx % len_target == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target, label_target = iter_target.next()
        data_target = data_target.cuda()
        optimizer.zero_grad()
        src_pred, mmd_loss = model(data_source, data_target)
        loss = nn.CrossEntropyLoss()(src_pred, label_source)
        p = float(batch_idx + (epoch-1) * num_iter) / args.epochs / num_iter
        lamda = 2. / (1. + np.exp(-10 * p)) - 1
        loss += lamda * mmd_loss
        loss.backward()
        optimizer.step()
        if (batch_idx+epoch*num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*args.batch_size, num_iter*args.batch_size,
                100. * batch_idx / num_iter, loss.item()))

def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output, mmd_loss = model(data, data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.data.cpu().max(1, keepdim=True)[1]
            correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DAN USPS MNIST')
    parser.add_argument('--task', default='USPS2MNIST', help='task to perform')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='cuda device id')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.task == 'USPS2MNIST':
        source_list, target_list, test_list= data_loader(task = 'U2M')
    elif args.task == 'MNIST2USPS':
        source_list, target_list, test_list= data_loader(task = 'M2U')
    else:
        raise Exception('task cannot be recognized!')

    train_loader = torch.utils.data.DataLoader(dataset= source_list, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader1 = torch.utils.data.DataLoader(dataset= target_list, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset= test_list, batch_size=args.test_batch_size, shuffle=True)

    model = models.NUMNet()
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    save_table = np.zeros(shape=(args.epochs, 2))
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, train_loader1, optimizer, epoch)
        acc = test(args, model, test_loader)
        save_table[epoch-1, :] = epoch, acc
        np.savetxt(args.task+'_50m_128_0.005.txt', save_table, delimiter=',', fmt='%1.3f')
    np.savetxt(args.task+'_50m_128_0.005.txt', save_table, delimiter=',', fmt='%1.3f')

if __name__ == '__main__':
    main()


