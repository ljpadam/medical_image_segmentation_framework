import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class LUConv(nn.Module):
    def __init__(self, inChans,  outChans, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, outChans)
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(outChans)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, outplanes=None):
        '''inplanes: the num of the input and output channels,
           planes: the num of the bottlenectk channels'''
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=stride, stride = stride, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        if outplanes == None:
            outplanes = inplanes
        self.conv3 = nn.Conv3d(planes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if self.stride != 1:
            self.downsample =  nn.Sequential(
                nn.Conv3d(inplanes, outplanes, kernel_size=stride, stride=stride, bias=False),
                nn.BatchNorm3d(outplanes)
                )


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

        if self.stride != 1:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ShrinkChannels(nn.Module):

    def __init__(self, inplanes, outplanes):
        '''inplanes: the num of the input  channels,
           outplanes: the num of the output channels'''
        super(ShrinkChannels, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(outplanes)
    
    def forward(self, x):
        return self.bn1(self.conv1(x))


def _make_nConv(inChans, depth):
    layers = []
    for _ in range(depth):
        layers.append(Bottleneck(inChans, inChans//4))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm3d(32)
        self.relu1 = ELUCons(elu, 32)

    def forward(self, x):

        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        self.down_sample = Bottleneck(inChans, inChans//4, stride = 2, outplanes=outChans) 
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.down_sample(x)
        out = self.ops(down)
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, skipxChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = nn.ConvTranspose3d(inChans, outChans//2, kernel_size=2, stride=2)
        self.bn1 = nn.InstanceNorm3d(outChans//2)
        self.relu1 = ELUCons(elu, outChans//2)

        self.skip = ShrinkChannels(skipxChans, outChans//2)

        self.ops1 = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):

        out1 = self.relu1(self.bn1(self.up_conv(x)))
        skipdata = self.skip(skipx)
        xcat = torch.cat((out1, skipdata), 1)
        out = self.ops1(xcat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        outputChannel = 2
        self.conv1 = nn.Conv3d(inChans, outputChannel, kernel_size=1, padding=0)
        if nll:
            self.softmax = F.log_softmax
        else:
            self.softmax = F.softmax

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv1(x)
        outputChannel = 2
        # make channels the last axis
        out = out.permute(0, 2, 3, 4, 1).contiguous()
        # flatten
        out = out.view(out.numel() // outputChannel, outputChannel)
        out = self.softmax(out)
        # treat channel 0 as the predicted output
        return out


class VNet2(nn.Module):
    # the deeper and thinner version of vnet
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=True):
        super(VNet2, self).__init__()
        self.in_tr = InputTransition(32, elu)
        self.down_tr64 = DownTransition(32, 64, 3, elu)
        self.down_tr128 = DownTransition(64, 128, 3, elu)
        self.down_tr256 = DownTransition(128, 256, 3, elu)
        self.up_tr128 = UpTransition(256, 16, 128,  2, elu)
        self.up_tr64 = UpTransition(16, 16, 64, 2, elu)
        self.up_tr32 = UpTransition(16, 16, 32, 2, elu)
        self.out_tr = OutputTransition(16, elu, nll)

    def forward(self, x):
        out32 = self.in_tr(x)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        out = self.up_tr128(out256, out128)
        out = self.up_tr64(out, out64)
        out = self.up_tr32(out, out32)

        out = self.out_tr(out)
        return out
