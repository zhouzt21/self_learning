from torchvision import datasets

train_data = datasets.MNIST(root = './data', train = True, download = True)
test_data = datasets.MNIST(root = './data', train = False, download = True)


