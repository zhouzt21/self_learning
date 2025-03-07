from torch.utils.data import Dataset
import os
import gzip
import numpy as np

class Mnist(Dataset):
    def __init__(self,root , train = True, transform = None):
        self.file_pre = 'train' if train == 'True' else 't10k'
        self.transform = transform
        self.label_path = os.path.join(root, self.file_pre + '-labels-idx1-ubyte.gz')
        print(self.label_path, 'label_path')
        self.image_path = os.path.join(root, self.file_pre + '-images-idx3-ubyte.gz')
        print(self.image_path, 'image_path')
        self.images, self.labels = self.__read_data(self.image_path, self.label_path)

    def __read_data(self, image_path, label_path):
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(image_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)
        
        return images, labels
    
    def __getitem__(self, index):  #迭代使用？使用Minist()会调用_getitem_
        image, label = self.images[index], int(self.labels[index])

        # 如果需要转成 tensor，（RGB,HWC）张量， 则使用 tansform
        if self.transform is not None:
            image = self.transform(np.array(image))  # 此处需要用 np.array(image)，转化为数组
        return image, label

    def __len__(self):#获取元素个数
        return len(self.labels)
