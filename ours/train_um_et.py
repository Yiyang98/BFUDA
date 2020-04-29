import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
from torch.autograd import Variable
import loss as loss_func
import numpy as np
import random
import network
from data_loader import *
from loss_com import *

seed=1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

def Tsharpen(input):
    lamda=0.5
    sharp=torch.pow(input,1/lamda)
    sharp2=torch.sum(sharp,1)
    result=torch.div(sharp,sharp2.view(-1, 1))
    return result


def train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch, start_epoch, method, ccp):
    cl_method = 'ga'   #choices=['ga', 'nn', 'free', 'pc', 'forward']
    meta_method = 'free' if cl_method =='ga' else cl_method
    K=10

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
        optimizer_ad.zero_grad()
        feature, output = model(torch.cat((data_source, data_target), 0))
        #err_s_label, loss_vector = non_negative_loss (f=output.narrow(0, 0, data_source.size(0)), K=10, labels=label_source, ccp=ccp,beta=0)
        loss, loss_vector = chosen_loss_c(f=output.narrow(0, 0, data_source.size(0)), K=K, labels=label_source, ccp=ccp, meta_method=meta_method)
        #loss = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        softmax_output = nn.Softmax(dim=1)(output)
        if cl_method == 'ga':
            if torch.min(loss_vector).item() < 0:
                loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(K, requires_grad=True).view(-1,1).to(device)), 1)
                min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
                loss = torch.sum(min_loss_vector)
                loss.backward(retain_graph=True)
                for group in optimizer.param_groups:
                    for p in group['params']:
                        p.grad = -1*p.grad
            else:
                loss.backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        if epoch > start_epoch:
            if method == 'CDAN-E':
                softmax_output = Tsharpen(softmax_output)
                entropy = loss_func.Entropy(softmax_output)
                loss2 = loss_func.CDAN([feature, softmax_output], ad_net, entropy, network.calc_coeff(num_iter*(epoch-start_epoch)+batch_idx), random_layer)
            elif method == 'CDAN':
                loss2 = loss_func.CDAN([feature, softmax_output], ad_net, None, None, random_layer)
            elif method == 'DANN':
                loss2 = loss_func.DANN(feature, ad_net)
            else:
                raise ValueError('Method cannot be recognized.')
        if epoch > start_epoch:
            loss2.backward()
            optimizer.step()
            optimizer_ad.step()
        if (batch_idx+epoch*num_iter) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}'.format(
                epoch, batch_idx*args.batch_size, num_iter*args.batch_size,
                100. * batch_idx / num_iter, loss.item()))

def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            feature, output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.data.cpu().max(1, keepdim=True)[1]
            correct += pred.eq(target.data.cpu().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def main(i):
    # Training settings
    parser = argparse.ArgumentParser(description='CDAN USPS MNIST')
    parser.add_argument('--method', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])
    parser.add_argument('--task', default='USPS2MNIST', help='task to perform')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=550, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr2', type=float, default=0.005, metavar='LR2',
                        help='learning rate2 (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='cuda device id')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--random', type=bool, default=False,
                        help='whether to use random')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.task == 'USPS2MNIST':
        source_list, ordinary_train_dataset, target_list, test_list, ccp= data_loader(task = 'U2M')
        start_epoch = 50
        decay_epoch = 600
    elif args.task == 'MNIST2USPS':
        source_list, ordinary_train_dataset, target_list, test_list, ccp= data_loader(task = 'M2U')
        start_epoch = 50
        decay_epoch = 600
    else:
        raise Exception('task cannot be recognized!')

    train_loader = torch.utils.data.DataLoader(dataset= source_list, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    train_loader1 = torch.utils.data.DataLoader(dataset= target_list, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    o_train_loader = torch.utils.data.DataLoader(dataset= ordinary_train_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset= test_list, batch_size=args.test_batch_size, shuffle=True, num_workers=8)

    model = network.LeNet()
    model = model.cuda()
    class_num = 10

    if args.random:
        random_layer = network.RandomLayer([model.output_num(), class_num], 500)
        ad_net = network.AdversarialNetwork(500, 500)
        random_layer.cuda()
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(model.output_num() * class_num, 500)
    ad_net = ad_net.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    optimizer_ad = optim.SGD(ad_net.parameters(), lr=args.lr2, weight_decay=0.0005, momentum=0.9)

    '''
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_ad = optim.Adam(ad_net.parameters(), lr=args.lr2, weight_decay=1e-4)
    '''

    save_table = np.zeros(shape=(args.epochs, 3))
    for epoch in range(1, args.epochs + 1):
        if epoch % decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5
        train(args, model, ad_net, random_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch, start_epoch, args.method, ccp)
        acc1 = test(args, model, o_train_loader)
        acc2 = test(args, model, test_loader)
        save_table[epoch-1, :] = epoch-50, acc1, acc2
        np.savetxt(str(i)+args.task + '_.txt', save_table, delimiter=',', fmt='%1.3f')
    np.savetxt(str(i)+args.task + '_.txt', save_table, delimiter=',', fmt='%1.3f')

if __name__ == '__main__':
    for i in range(0,5):
        main(i)

    #main()
