from model import LeNet5
import torch
from torch.utils.data import DataLoader
from number_dataset import Mnist
from torchvision import transforms
import matplotlib.pyplot as plt

net = LeNet5()
net.load_state_dict(torch.load("Linear.pth"))
test_set = Mnist(root = './data/MNIST/raw', train =False, transform=transforms.Compose([transforms.ToTensor()]))

test_loader = DataLoader(test_set, batch_size = 32, shuffle = True)

for data in test_loader:
    images, labels = data

# print(images.type(), 'images type')

test_output = net(images[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(labels[:10].numpy(), 'real number')

def plt_image(images):#定义一个函数，将需要预测的手写数字图画出来
    n = 10
    plt.figure(figsize=(10,4))
    for i in range(n):
        ax = plt.subplot(2,5,i+1)
        plt.imshow(images[i].reshape(28,28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
plt_image(images)
