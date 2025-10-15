"""
This is the implementation of DeepLabv2 without multi-scale inputs. This implementation uses ResNet-101 by default.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import math
import numpy as np
affine_par = True
from torch.autograd import Function

class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lamda
        return output, None	
		
		

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

			
def grad_reverse(x, lambd=1):
    return GradReverse(lambd)(x)
	
	
def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out


	
class Classifier_Module_GRL(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x, reverse=True, eta=0.1):
        if reverse:
            x = grad_reverse(x, eta)		
		
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out			
			
			
			
class Classifier_Module_1024(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module_1024, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
            return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

    
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i
            


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}] 


def Res_Deeplab(num_classes=21):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model
	
	
	
class Multi_Split_ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Multi_Split_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 
        x = self.layer1(x) # 256
        x = self.layer2(x) #512
        x = self.layer3(x)  # 1024
        x1 = self.layer4(x)  # 2048
        x2 = self.layer5(x1)

        return x, x1, x2

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

    
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i
            


    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}] 


def Multi_Split_Res_Deeplab(num_classes=21):
    model = Multi_Split_ResNet(Bottleneck,[3, 4, 23, 3], num_classes)
    return model
	

	

	
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    #c= prob.shape[1]
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)    
	
	
def interp(prob, input_size1, input_size2):	
    return  F.upsample(prob, size=(input_size1, input_size2), mode='bilinear', align_corners=True)
	
	
	#nn.Upsample(size=(input_size1, input_size2), mode='bilinear', align_corners=True)	
	
class FeatureDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=512, num_classes=2):
        super(FeatureDiscriminator, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(ndf//2, ndf//4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.5)
			)
        self.cls1 = nn.Conv2d(ndf//4, num_classes, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(ndf//4, num_classes, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1) 	
        self.fc = nn.Linear(ndf//4, 1)

    def forward(self, x, size=None):
        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        #maps = self.avgpool(out)
        #conv4_maps = maps 
        #D_out = self.fc( maps.view(maps.size(0), -1) )	
        out = torch.cat((src_out, tgt_out), dim=1)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out

	
class Split_feature_discriminator_0(nn.Module):

    def __init__(self, layer_name, ndf = 64):
        super(Split_feature_discriminator_0, self).__init__()
		
        if layer_name == 'layer1':
            self.conv1 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40  #layer 1
        elif layer_name == 'layer2':
            self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 2
        elif layer_name == 'layer3':
            self.conv1 = nn.Conv2d(ndf*4*4, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 3
        elif layer_name == 'layer4':
            self.conv1 = nn.Conv2d(ndf*4*8, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 4
			

        #self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40  #layer 1
        #self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 2	
        #self.conv1 = nn.Conv2d(ndf*4*4, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 3	
        #self.conv1 = nn.Conv2d(ndf*4*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 4	
		
        self.conv2 = nn.Conv2d(  ndf*4, ndf*3, kernel_size=4, stride=2, padding=1) # 20 x 20
        self.conv3 = nn.Conv2d(ndf*3, ndf*2, kernel_size=4, stride=2, padding=1) # 10 x 10
        self.conv4 = nn.Conv2d(ndf*2, ndf*1, kernel_size=4, stride=1, padding=1) # 10 x 10
        self.avgpool = nn.AdaptiveAvgPool2d(1) 	
		
        self.fc = nn.Linear(ndf*1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
    # def forward(self, x, reverse=False, eta=1.0):
        # if reverse:
            # x = grad_reverse(x, eta) # GRL layer	
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps
		
		
class Split_feature_discriminator_2(nn.Module):

    def __init__(self, layer_name, ndf = 64):
        super(Split_feature_discriminator_2, self).__init__()
		
        if layer_name == 'layer1':
            self.conv1 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40  #layer 1
        elif layer_name == 'layer2':
            self.conv1 = nn.Conv2d(ndf*4*1, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 2
        elif layer_name == 'layer3':
            self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 3
        elif layer_name == 'layer4':
            self.conv1 = nn.Conv2d(ndf*4*4, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 4
			

        #self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40  #layer 1
        #self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 2	
        #self.conv1 = nn.Conv2d(ndf*4*4, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 3	
        #self.conv1 = nn.Conv2d(ndf*4*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 4	
		
        self.conv2 = nn.Conv2d(  ndf*4, ndf*3, kernel_size=4, stride=2, padding=1) # 20 x 20
        self.conv3 = nn.Conv2d(ndf*3, ndf*2, kernel_size=4, stride=1, padding=1) # 10 x 10
        self.conv4 = nn.Conv2d(ndf*2, ndf*1, kernel_size=4, stride=1, padding=1) # 10 x 10
        self.avgpool = nn.AdaptiveAvgPool2d(1) #nn.AvgPool2d((10, 10))	
		
        self.fc = nn.Linear(ndf*1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
    # def forward(self, x, reverse=False, eta=1.0):
        # if reverse:
            # x = grad_reverse(x, eta) # GRL layer	
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps	
		
class Split_feature_discriminator_4(nn.Module):

    def __init__(self, layer_name, ndf = 64):
        super(Split_feature_discriminator_4, self).__init__()
		
        if layer_name == 'layer1':
            self.conv1 = nn.Conv2d(ndf*1, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40  #layer 1
        elif layer_name == 'layer2':
            self.conv1 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 2
        elif layer_name == 'layer3':
            self.conv1 = nn.Conv2d(ndf*4*1, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 3
        elif layer_name == 'layer4':
            self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 4
			

        #self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40  #layer 1
        #self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 2	
        #self.conv1 = nn.Conv2d(ndf*4*4, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 3	
        #self.conv1 = nn.Conv2d(ndf*4*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 4	
		
        self.conv2 = nn.Conv2d(  ndf*4, ndf*3, kernel_size=4, stride=2, padding=1) # 20 x 20
        self.conv3 = nn.Conv2d(ndf*3, ndf*2, kernel_size=4, stride=2, padding=1) # 10 x 10
        self.conv4 = nn.Conv2d(ndf*2, ndf*1, kernel_size=4, stride=1, padding=1) # 10 x 10
        self.avgpool = nn.AdaptiveAvgPool2d(1) # nn.AvgPool2d((10, 10))	
		
        self.fc = nn.Linear(ndf*1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
    # def forward(self, x, reverse=False, eta=1.0):
        # if reverse:
            # x = grad_reverse(x, eta) # GRL layer	
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps		

		
class Split_feature_discriminator_8(nn.Module):

    def __init__(self, layer_name, ndf = 64):
        super(Split_feature_discriminator_8, self).__init__()
		
        if layer_name == 'layer1':
            self.conv1 = nn.Conv2d(32, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40  #layer 1
        elif layer_name == 'layer2':
            self.conv1 = nn.Conv2d(ndf*1, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 2
        elif layer_name == 'layer3':
            self.conv1 = nn.Conv2d(ndf*2*1, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 3
        elif layer_name == 'layer4':
            self.conv1 = nn.Conv2d(ndf*4*1, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 4
			

        #self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40  #layer 1
        #self.conv1 = nn.Conv2d(ndf*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 2	
        #self.conv1 = nn.Conv2d(ndf*4*4, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 3	
        #self.conv1 = nn.Conv2d(ndf*4*4*2, ndf*4, kernel_size=4, stride=2, padding=1) # 40 x 40	#layer 4	
		
        self.conv2 = nn.Conv2d(  ndf*4, ndf*3, kernel_size=4, stride=2, padding=1) # 20 x 20
        self.conv3 = nn.Conv2d(ndf*3, ndf*2, kernel_size=4, stride=2, padding=1) # 10 x 10
        self.conv4 = nn.Conv2d(ndf*2, ndf*1, kernel_size=4, stride=1, padding=1) # 10 x 10
        self.avgpool = nn.AdaptiveAvgPool2d(1) # nn.AvgPool2d((10, 10))	
		
        self.fc = nn.Linear(ndf*1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
    # def forward(self, x, reverse=False, eta=1.0):
        # if reverse:
            # x = grad_reverse(x, eta) # GRL layer	
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps			
				
class entropy_discriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(entropy_discriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
        self.drop = nn.Dropout2d(0.4)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
    # def forward(self, x, reverse=False, eta=1.0):
        # if reverse:
            # x = grad_reverse(x, eta)
	
        x = self.conv1(x)
        x = self.leaky_relu(x)
		
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)	
		
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
		
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.drop(x)			
		
        x = self.conv5(x)
       		
        return x
	
	
# baseline_model
class deeplab_maxmin(nn.Module):
    def __init__(self, block, layers, num_classes):# layer3_name, layer4_name):#, input_size1, input_size2):
        self.inplanes = 64
        super(deeplab_maxmin, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)
        # self.layer6 = self._disc_entropy_D1(num_classes)
        # self.layer7 = self._disc_feature(layer3_name)		
        # self.layer8 = self._disc_feature(layer4_name)
	
        # self.size1=input_size1
        # self.size2=input_size2			
		
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
               # for i in m.parameters():
                   # i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)
		
    # def _disc_entropy_D1(self,num_classes):
        # return entropy_discriminator(num_classes)		
		
    # def _disc_feature(self,layer_name):
        # return Split_feature_discriminator_0(layer_name)	
		

			
		

    def forward(self, x):
        size1 = x.size(2)
        size2 = x.size(3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 
        x = self.layer1(x) # 256
        x = self.layer2(x) #512
        x = self.layer3(x)  # 1024
        x4 = self.layer4(x)  # 2048
        x5 = self.layer5(x4)
		
        # #prob_2_entropy(F.softmax(pred_src_main, dim=1))
		# # GRL
        # x5 = interp(x5, size1, size2)	
        # reverse_x5_entropy=prob_2_entropy( F.softmax(x5, dim=1) ) #grad_reverse( prob_2_entropy( F.softmax(x5, dim=1) ) )
		
		
        # entropy_output= self.layer6(reverse_x5_entropy)

        # reverse_x3 =x3# grad_reverse(x3)	
        # reverse_x4 = x4 #grad_reverse(x4)
        # domian_x3_output = self.layer7(reverse_x3)
        # domian_x4_output = self.layer8(reverse_x4)

        return   x5, x, x4

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

    
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i    
            
    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}] 

								
				
				
				

def Deeplab_maxmin_model(num_classes=21):#, layer3_name='layer3', layer4_name='layer4'): #, input_size1=321, input_size2=321):
    model = deeplab_maxmin(Bottleneck,[3, 4, 23, 3], num_classes)#, layer3_name, layer4_name) #,321,321)
    return model
	

		

	
	
# baseline_model
class deeplab_maxmin_all(nn.Module):
    def __init__(self, block, layers, num_classes):# layer3_name, layer4_name):#, input_size1, input_size2):
        self.inplanes = 64
        super(deeplab_maxmin_all, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine = affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],num_classes)
        # self.layer6 = self._disc_entropy_D1(num_classes)
        # self.layer7 = self._disc_feature(layer3_name)		
        # self.layer8 = self._disc_feature(layer4_name)
	
        # self.size1=input_size1
        # self.size2=input_size2			
		
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
               # for i in m.parameters():
                   # i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self,block, dilation_series, padding_series,num_classes):
        return block(dilation_series,padding_series,num_classes)
		
    # def _disc_entropy_D1(self,num_classes):
        # return entropy_discriminator(num_classes)		
		
    # def _disc_feature(self,layer_name):
        # return Split_feature_discriminator_0(layer_name)	
		

			
		

    def forward(self, x):
        size1 = x.size(2)
        size2 = x.size(3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 
        x1 = self.layer1(x) # 256
        x2 = self.layer2(x1) #512
        x3 = self.layer3(x2)  # 1024
        x4 = self.layer4(x3)  # 2048
        x5 = self.layer5(x4)

        return   x5, x1, x2, x3, x4

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for 
        the last classification layer. Note that for each batchnorm layer, 
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

    
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj+=1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i    
            
    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10*args.learning_rate}] 

								
				
				
				

def Deeplab_maxmin_model_all(num_classes=21):#, layer3_name='layer3', layer4_name='layer4'): #, input_size1=321, input_size2=321):
    model = deeplab_maxmin_all(Bottleneck,[3, 4, 23, 3], num_classes)#, layer3_name, layer4_name) #,321,321)
    return model
	
		