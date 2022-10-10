from __future__ import print_function
import os
import argparse
import save_1
from torchvision import datasets, transforms
from SAR_BagNet import SAR_BagNet
from trades import trades_loss
import time
parser = argparse.ArgumentParser(description='PyTorch TRADES Adversarial Training')
parser.add_argument('--batch_size', type=int, default=30, metavar='N',
                    help='batch size')
parser.add_argument('--test_batch_size', type=int, default=30,
                    help='input batch size for testing ')
parser.add_argument('--epochs', type=int, default=200 ,
                    help='number of epochs to train')
parser.add_argument('--attack_strength', default=8/255,
                    help='attack strength of PGD attack')
parser.add_argument('--attack_iter', default=10,
                    help='attack iter of PGD attack')
parser.add_argument('--attack_step_size', default=2/255,
                    help='attack step size of PGD attack')
parser.add_argument('--beta', default=6.0,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--mean',type=float,default=0.184,help='mean of dataset')
parser.add_argument('--std',type=float,default=0.119,help='standard  of dataset')
parser.add_argument('--distance',type=float,default=0.119,help='standard  of dataset')
global args
args = parser.parse_args()
def train(train_iter, test_iter, net, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for i,(X, y) in enumerate(train_iter):
            X=X.to(device)
            y=y.to(device)
            optimizer.zero_grad()
            l = trades_loss(model=net,
                               x_natural=X,
                               y=y,
                               optimizer=optimizer,
                               step_size=args.attack_step_size,
                               epsilon=args.attack_strength,
                               perturb_steps=args.attack_iter,
                               beta=args.beta,
                               distance='l_inf')

            l.backward()
            optimizer.step()
            y_hat = net(X)
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch {}, loss {:.4f}, train acc {:.4f}, test acc{:.4f}, time {:.2f} sec'
              .format(epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        save_1.save_model_w_condition(model=net,
                                    model_dir='.../saved_models/',
                                    model_name=str(epoch+1) + 'bagnet_TR_8_2', accu=test_acc,
                                    target_accu=0.98)
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
# settings
if __name__ == '__main__':
    mean = args.mean
    std = args.std
    img_size=100
    train_dir = '.../train'
    test_dir = '.../val'
    train_batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=0, pin_memory=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_dir = '.../saved_models/'
    lr = 0.001
    num_epochs=args.epochs
    net = SAR_BagNet.SAR_BagNet(pretrained=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(train_loader,test_loader,net,optimizer,device,num_epochs)
    torch.save(obj=net, f=os.path.join(mode_dir, 'SAR_BagNet_trades ' + '.pth'))
