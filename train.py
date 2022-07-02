import torch
import torchvision
import torch.nn as nn
from Model import VGG, build_vgg
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from torch.utils.data import DataLoader

import sys
from tqdm import tqdm

#import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter

def main():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #图像预处理
        # data_transform = {
        #     "train_set": transforms.Compose([
        #         torchvision.transforms.ToTensor()
        #         ])
        # }
        data_transform = transforms.Compose([
                torchvision.transforms.ToTensor()
        ])


        cifarData = torchvision.datasets.CIFAR10(root="./TrainData", train=True, transform=data_transform, download=True)
        cifarTestData = torchvision.datasets.CIFAR10(root="./TrainData", train=False, transform=data_transform, download=True)

        trainloader = torch.utils.data.DataLoader(cifarData, batch_size=36, shuffle=True, num_workers=4, drop_last=True) #batch size 一次打包几个数据 shuffle是否将数据打乱
        testloader = torch.utils.data.DataLoader(cifarTestData, batch_size=64, shuffle=False, num_workers=4)#num workers线程

        test_num = len(cifarTestData)

        #转换成迭代器
        # test_data_iter = iter(testloader)
        # test_image, test_label = test_data_iter.next()

        # img, target = cifarTestData[0]
        # print(img.shape)
        # print(target)

        # writer = SummaryWriter("dataloader")
        # step = 0
        # for data in testloader:
        #     imgs, targets = data
        #     writer.add_images("test_data", imgs, step)
        #     step += 1
        # writer.close

        #显示绘制数据集图片
        # def imshow(img):
        #     #反标准化
        #     img = img / 2 + 0.5
        #     npimg = img.numpy()
        #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #     plt.show

        #调换图片和标签的顺序
        classier_list = cifarData.class_to_idx
        classier_dict = dict((val, key) for key, val in classier_list.items())

        model_name = "vgg_16"
        vgg_net = build_vgg(model_name=model_name, num_classes = 10)
        vgg_net.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(vgg_net.parameters(), lr=0.0001)

        epochs = 30
        best_ac = 0.0
        save_path = './{}VggNet.pth'.format(model_name)
        train_steps = len(trainloader)

        for epoch in range(epochs):
                vgg_net.train()
                running_loss = 0.0
                train_bar = tqdm(trainloader, file=sys.stdout)
                for step, data in enumerate(train_bar):
                        images, labels = data
                        optimizer.zero_grad()#清空之前的信息
                        outputs = vgg_net(images.to(device))
                        loss = loss_function(outputs, labels.to(device))
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()

                        train_bar.desc = "train epoch[{}/{}] loss:{:/3f}".format(epoch +1, epochs, loss)

                vgg_net.eval()
                acc = 0.0
                with torch.no_grad():
                        test_bar = tqdm(testloader, file=sys.stdout)
                        for test_data in test_bar:
                                test_images, test_labels = test_data
                                outputs = vgg_net(test_images.to(device))
                                predict_y = torch.max(outputs, dim=1)[1]
                                acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
                test_acc = acc/ test_num
                print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' %(epoch + 1, running_loss/train_steps, test_acc))

                if test_acc > best_ac:
                        best_ac = test_acc
                        torch.save(net.stat_dict(), save_path)

        print('FInished Traning')

if __name__== '__main__':
        main()


