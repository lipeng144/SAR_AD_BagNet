import torch
from SAR_BagNet import SAR_BagNet
from torch import nn
import time
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import argparse
parser = argparse.ArgumentParser(description='Pytorch MSTAR Training')
parser.add_argument('--mean',type=float,default=0.184,help='mean of dataset')
parser.add_argument('--std',type=float,default=0.119,help='standard deviation of dataset')
parser.add_argument('--epochs',type=int,default=200,help='number of total epochs to run')
parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--test_batch_size', type=int, default=30, help='input batch size for testing ')
parser.add_argument('--if_al',type=bool,default=True,help='if the adversarial learning is used in this learning,you can set False to pre-train a DNN for further training')
parser.add_argument('--attack_strength',type=int,default=8,help='attack strength of PGD attack')
parser.add_argument('--attack_step_size',type=int,default=2,help='attack step size of PGD attack')
parser.add_argument('--attack_iter',type=int,default=10,help='attack iter of PGD attack')
global args
args = parser.parse_args()
def clip_by_tensor(t,t_min,t_max):
    m1=t>t_min
    m2=~m1
    t_mew=m1*t+m2*t_min
    n1=t<t_max
    n2=~n1
    t_mew=n1*t_mew+n2*t_max
    return t_mew
def train(train_iter, test_iter, net, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for i,(X, y) in enumerate(train_iter):
            X=X.to(device)
            y=y.to(device)
            if if_al:
                X.requires_grad = True
                X.requires_grad_()
                pertubation = torch.zeros(X.shape).type_as(X).cuda()
                min, max = X - attack_strength / 255 / std, X + attack_strength / 255 / std

                for _ in range(attack_iter):
                    y_hat=net(X)
                    l=loss(y_hat,y)
                    grad_outputs = None
                    grads = torch.autograd.grad(l, X, grad_outputs=grad_outputs, only_inputs=True, retain_graph=False)[0]
                    pertubation = step_size / 255 / std * torch.sign(grads)
                    X = clip_by_tensor(X + pertubation, min, max)
                    X = clip_by_tensor(X, -mean / std, (1 - mean) / std)
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch {}, loss {:.4f}, train acc {:.4f}, test acc{:.4f}, time {:.2f} sec'
              .format(epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n
if __name__ == '__main__':
    mean = args.mean
    std = args.std
    img_size=100
    train_dir = '.../train'
    test_dir = '.../val'
    train_batch_size=args.batch_size
    test_batch_size=args.test_batch_size
    normalize = transforms.Normalize((mean,),(std,))
    train_dataset = datasets.ImageFolder(
                    train_dir,
                    transforms.Compose([
                    transforms.Resize(size=(img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    normalize,
                    ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=0, pin_memory=False)
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=3),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=0, pin_memory=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_dir = '.../saved_model'
    lr = 0.001
    num_epochs=args.epochs
    attack_strength = args.attack_strength
    step_size = args.attack_step_size
    attack_iter = args.attack_iter
    net = SAR_BagNet(pretrained=False)
    if_al=args.if_al
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(train_loader,test_loader,net,optimizer,device,num_epochs)
    torch.save(obj=net, f=os.path.join(mode_dir, ('SAR_BagNet' + '{0:.3f}.pth').format(18)))