import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class FPN(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(FPN, self).__init__()
        
        self.l_conv1 = nn.Conv2d(in_channels[0], out_channel, kernel_size=1, bias=False)
        self.l_bn1 = nn.BatchNorm2d(out_channel)

        self.fpn_conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.fpn_bn1 = nn.BatchNorm2d(out_channel)

        self.l_conv2 = nn.Conv2d(in_channels[1], out_channel, kernel_size=1, bias=False)
        self.l_bn2 = nn.BatchNorm2d(out_channel)

        self.fpn_conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.fpn_bn2 = nn.BatchNorm2d(out_channel)

        self.l_conv3 = nn.Conv2d(in_channels[2], out_channel, kernel_size=1, bias=False)
        self.l_bn3 = nn.BatchNorm2d(out_channel)

        self.fpn_conv3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.fpn_bn3 = nn.BatchNorm2d(out_channel)

        self.l_conv4 = nn.Conv2d(in_channels[3], out_channel, kernel_size=1, bias=False)
        self.l_bn4 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, inputs):
        x4 = self.l_conv4(inputs[3])
        x4 = self.l_bn4(x4)
        x4 = self.relu(x4)
        
       # x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        
#        x3 = self.l_conv3(inputs[2])
#        x3 = self.l_bn3(x3)
#        x3 = self.relu(x3)

#        x3 += x4 
#        
#        x3 = self.fpn_conv3(x3)
#        x3 = self.fpn_bn3(x3)
#        x3 = self.relu(x3)
        
#        x3 = F.interpolate(x3, scale_factor=2, mode='nearest')

#        x2 = self.l_conv2(inputs[1])
#        x2 = self.l_bn2(x2)
#        x2 = self.relu(x2)
#    
#        x2 += x3
#        
#        x2 = self.fpn_conv2(x2)
#        x2 = self.fpn_bn2(x2)
#        x2 = self.relu(x2)
#        
#        x2 = F.interpolate(x2, scale_factor=2, mode='nearest')

#        x1 = self.l_conv1(inputs[0])
#        x1 = self.l_bn1(x1)
#        x1 = self.relu(x1)
#        
#        x1 += x2
#        
#        x1 = self.fpn_conv1(x1)
#        x1 = self.fpn_bn2(x1)
#        x1 = self.relu(x1)

        return x4

