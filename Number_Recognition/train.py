# import torch
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from model import LeNet5
# from number_dataset import Mnist


# train_set = Mnist(root = './data/MNIST/raw', train = True, 
#                   transform = transforms.Compose([transforms.ToTensor()]))
# train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)

# net = LeNet5().to(device)
# loss_func = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# loss_list = [] # save loss value for plotting

# for epoch in range(100):
#     running_loss = 0.0
#     for batch_idx, data in enumerate(train_loader):
#         images, labels = data 
#         images = images.to(device)
#         labels = labels.to(device)
#         optimizer.zero_grad()
#         loss = loss_func(net(images), labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#         if batch_idx % 300 == 299:
#             print('epoch:{} batch_idx:{} loss:{}'
#                   .format(epoch+1, batch_idx+1, running_loss/300))
#             loss_list.append(running_loss/300)
#             running_loss = 0.0

# torch.save(net.state_dict(),"Linear.pth")
# print('Finished Training')

# # Plot loss value
# import matplotlib.pyplot as plt
# plt.plot(loss_list)
# plt.xlabel('batch_idx')
# plt.ylabel('loss')
# plt.show()


import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from model import LeNet5
from number_dataset import Mnist
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, color-coded for accuracy.
    '''
    net.eval()
    with torch.no_grad():
        output = net(images)
        _, preds = torch.max(output, 1)

    fig = plt.figure(figsize=(12, 48))

    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].cpu().numpy().squeeze(), cmap='gray')
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            preds[idx],
            torch.nn.functional.softmax(output, dim=1)[idx][preds[idx]] * 100.0,
            labels[idx]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


# 创建一个 SummaryWriter 对象
writer = SummaryWriter('runs/experiment_1')

def visualize_features(module, input, output):
    # 将特征图移动到 CPU，并移除计算图
    output = output.detach().cpu()
    # print(output.shape)
    # 选择第一个通道的所有图像
    output = output[:, 0:1, :, :]
    # 复制到3个通道
    output = output.repeat(1, 3, 1, 1)
    # 将特征图添加到 TensorBoard
    writer.add_images('features', output)


train_set = Mnist(root = './data/MNIST/raw', train = True, 
                  transform = transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size = 32, shuffle = True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

net = LeNet5().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

# 在第一层卷积层上注册钩子
net.conv1.register_forward_hook(visualize_features)

loss_list = [] # save loss value for plotting

for epoch in range(20):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        images, labels = data 
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print('epoch:{} batch_idx:{} loss:{}'
                  .format(epoch+1, batch_idx+1, running_loss/300))
            loss_list.append(running_loss/300)
            running_loss = 0.0

            # 添加特征图
            writer.add_images('images', images)
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(net, images, labels),
                              global_step=epoch * len(train_loader) + batch_idx)

torch.save(net.state_dict(),"Linear.pth")
print('Finished Training')

# Plot loss value
plt.plot(loss_list)
plt.xlabel('batch_idx')

# 关闭 SummaryWriter
writer.close()