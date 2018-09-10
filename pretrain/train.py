import sys
import torch
import torch.nn
import torch.nn.functional
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

sys.path.append('.')
import pretrain.mobilenetv2
import pretrain.darknet
import pretrain.shufflenetv2


class Model_classifi(torch.nn.Module):
    def __init__(self):
        super(Model_classifi, self).__init__()
        self.features = pretrain.darknet.darknet21(None)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(1024, 100)

    def forward(self, x):
        _, _, x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, 1)


# parameters
writer = SummaryWriter(log_dir='pretrain/log')
frequency = 100
# load data
root = '/home/wxrui/DATA/cifar100/'
train_set = torchvision.datasets.CIFAR100(root=root, train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.RandomResizedCrop(416),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor()]),
                                          )
eval_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True,
                                         transform=transforms.Compose([
                                             transforms.Resize(416),
                                             transforms.ToTensor()]),
                                         )
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=16,
                                           pin_memory=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_set, batch_size=1, shuffle=True, num_workers=16,
                                          pin_memory=True)

# model
model = Model_classifi()
model = torch.nn.DataParallel(model.cuda())
model.load_state_dict(torch.load('pretrain/output/pre_train29.pth'))

# criterion and optimizer
multistep = [60]
epochs = multistep[-1]
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=multistep, gamma=0.1)

# train the network
for epoch in range(epochs):
    # adjust learning_rate
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])

    # train
    running_loss = 0
    correct = 0
    total = 0
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += torch.sum(predicted == target).item()
        if i % frequency == frequency - 1:
            print('train {}/{}: {}/{}: loss={:.3f}'.format(epoch, epochs, i, len(train_loader),
                                                           running_loss / frequency))
            print('train {}/{}: {}/{}: accuracy={:.3f}%'.format(epoch, epochs, i, len(train_loader),
                                                                correct / total * 100))
            writer.add_scalar('Train/loss', running_loss / frequency, epoch * len(train_loader) + i)
            writer.add_scalar('Train/accu', correct / total, epoch * len(train_loader) + i)
            running_loss = 0
            correct = 0
            total = 0

    # eval
    model.eval()
    correct = 0
    total = 0
    for i, (input, target) in enumerate(eval_loader):
        input, target = input.cuda(), target.cuda()
        output = model(input)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print('the accuracy of {}/{} epoch is {:.3f}%'.format(epoch, epochs, correct / total * 100))
    writer.add_scalar('Eval/accu', correct / total, epoch)

    # save
    torch.save(model.state_dict(), 'pretrain/output/pre_train%d.pth' % epoch)
writer.close()

# save backbone
torch.save(model.module.features.state_dict(), 'darknet21.pth')
