from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    #num_classes:定义最后的分类数量
    def __init__(self, features, num_classes: int = 1000, init_weights=False):
        super().__init__()
        self.features = features
        #nn.Sequential定义最后三层全连接层：分类层
        self.classfier = nn.Sequential(
            #Conv.512*7*7 --> FC.4096
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        #如果初始化权重为True,执行初始化参数
        if init_weights:
            self.init_weights()

    #正向传播
    def forward(self, pic_info: torch.Tensor):
        pic_info = self.features(pic_info)
        #pic_info = self.avgpool(pic_info)
        #输出展评，后面的参数为batch维度转换的起始位置
        pic_info = torch.flatten(pic_info, 1)
        #输入到分类网络结构
        pic_info = self.classifier(pic_info)
        return(pic_info)

    def init_weights(self):
    #遍历每一个模块，也就是遍历每一层
        for m in self.modules():
            #如果这个模块是一个卷基层
            if isinstance(m, nn.Conv2d):
                #Pytorch使用了kaiming正态分布初始化参数
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                #如果设置了bias，使bias为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    
def make_layers_features(cfg: List[Union[str, int]]):#Union 函数解释字符，解释[]内的东西“既可能是又可能是”
    conv_layers: List[nn.Module] = []
    #定义输入图片的通道，RGB为3
    input_channel = 3
    for c in cfg:
        if c == "M":
            conv_layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)] #添加pooling层
        else:
            conv2d= nn.Conv2d(input_channel, c, kernel_size = 3, padding = 1)
            conv_layers += [conv2d, nn.ReLU(True)] #添加卷积层和ReLU激活层1
            input_channel = c
    # *--收集参数，最后放入元组类型列表，非关键字传入
    return nn.Sequential(*conv_layers)

#卷积层网络结构
cfgs = {
    'vgg_16' : [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    'vgg_19' : [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]    
}

# **定义的是按关键字输入
def build_vgg(model_name, **rest):
    cfg = cfgs[model_name]
    model = VGG(make_layers_features(cfg))
    return model
