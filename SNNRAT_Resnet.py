import torch
import torch.nn as nn

#from models import layers     # for main call
import layers     # for local test

#TODO
#use general builder to generate all resnet models

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 conv with padding"""
    return layers.SeqToANNContainer(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation))

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 conv"""
    return layers.SeqToANNContainer(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False))

class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, T, inplanes, planes, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        
        self.T = T
        self.merge = layers.MergeTemporalDim()
        self.expand = layers.ExpandTemporalDim(self.T) 
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = layers.SeqToANNContainer(nn.BatchNorm2d(planes))
        self.neuron1 = layers.LIF()
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = layers.SeqToANNContainer(nn.BatchNorm2d(planes))
        self.neuron2 = layers.LIF()
        
        self.downsample = downsample
        

    def forward(self, x):     # x.dims() = 5
        residual = x    
        x01 = self.conv1(x)     # 5 -> 4 
        x02 = self.bn1(x01)     # 4-> 4
        x03 = self.neuron1(x02)     # 4 -> 5 -> 4     
        
        x04 = self.conv2(x03)     # 4->4
        x05 = self.bn2(x04)     # 4 -> 5
        
        if self.downsample is not None:
            residual = self.downsample(residual)     # 5 -> 4(conv + bn) -> 5
        
        x06 = residual + x05
        x07 = self.neuron2(x06)     # 4 -> 5 -> 4

        feature = [
            x01, x02, x03,
            x04, x05, x06,
            x07
                   ]    

        return feature


class Resnet(nn.Module):    
    def __init__(self, T, num_classes=10, **kwargs):
        super(Resnet, self).__init__()
        self.T = T
        self.expand = layers.ExpandTemporalDim(self.T)     
        self.merge = layers.MergeTemporalDim()

        # [notice] this operation is problematic because it makes dificult to observe the data flow between Conv nad Bn, try to fix this
        # [warning] 这个 preconv 和标准 resnet 好像不太一样，标准 resnet 包括 cbap, 这里只有 cb+a 
        self.pre_conv = layers.SeqToANNContainer(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(64))
        self.neuron1 = layers.LIF()    
        
        self.layer1 = self.make_layer(64, 64, 2, stride=2)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256, 512, 2, stride=2)
        
        self.pool = layers.SeqToANNContainer(nn.AdaptiveAvgPool2d(1))
 
        self.fc1 = layers.SeqToANNContainer(nn.Linear(512, num_classes))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
 
        downsample = layers.SeqToANNContainer(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False), nn.BatchNorm2d(out_ch))
        layers_ = []
        layers_.append(BasicBlock(self.T, in_ch, out_ch, stride, downsample))
        for i in range(1, block_num):
            layers_.append(BasicBlock(self.T, out_ch, out_ch))
        return nn.Sequential(*layers_)

    def forward(self, input):

        if self.T > 0:
            input = layers.add_dimention(input, self.T)     # 4 -> 5
            
        x01_ = self.pre_conv(input)     # 5-> 4
        x02_ = self.neuron1(x01_)     # 4 -> 5 
        
        x_list = self.layer1(x02_)     # 5 -> 5  
        x_list = self.layer2(x_list)     
        x_list = self.layer3(x_list)
        x_list = self.layer4(x_list)
        
        x03_ = self.pool(x_list[-1])     # 5 -> 4 -> 5

        x04_ = torch.flatten(x03_, 2)     # 5 -> 3
        
        x05_ = self.fc1(x04_)     # 3 -> 3

        feature_ = [x01_, x02_, *x_list, x03_, x05_]
    
        return feature_
    
    def set_simulation_time(self, mode='bptt', input_decay=False, tau=1.):
        for module in self.modules():
            if isinstance(module, layers.LIF) or isinstance(module, layers.LIF):
                module.mode = mode
                module.input_decay = input_decay
                module.tau = tau

if __name__ == '__main__':
    model = Resnet(T=4)
    print(model)
    x = torch.rand(2, 3, 32, 32)
    x = model(x)
    for idx, l in enumerate(x):     # [warning] 上述代码改动出现了问题，期望目标是提升训练速度，但这点并没有显著改善；同时带来的副作用是某些层输出的shape是
                                    # [T*B, C, H, W], 这导致在后续的分析时需要进行额外分析，去将 feature 重新分割成 [T, B, C, H, W] 的形状，由于期望的提升
                                    # 并没有获得，同时导致了额外的问题，将代码回归到标准形式，同时也更为简洁 [working...]
        print(idx, l.shape)