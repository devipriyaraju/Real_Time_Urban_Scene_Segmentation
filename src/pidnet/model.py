"""PIDNet architecture used in this reproduction.

Implementation follows the architecture described by Xu, Xiong, and
Bhattacharyya (CVPR 2023) and the authors' public PIDNet implementation.
See NOTICE and README.md for attribution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 2. Model building blocks  (BasicBlock, Bottleneck, heads, PAPPM/DAPPM,
#    PagFM, Bag, Light_Bag) - faithful to the official PIDNet implementation
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

bn_mom = 0.1
algc = False  # align_corners for the model's internal interpolations

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.no_relu = no_relu
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out if self.no_relu else self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out if self.no_relu else self.relu(out)

class segmenthead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, 1, padding=0, bias=True)
        self.scale_factor = scale_factor
    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            h, w = x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=[h, w], mode='bilinear', align_corners=algc)
        return out

class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BN=nn.BatchNorm2d):
        super().__init__()
        def pool(k, s, p):
            return nn.Sequential(nn.AvgPool2d(k, s, p), BN(inplanes, momentum=bn_mom),
                                 nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale1, self.scale2, self.scale3 = pool(5,2,2), pool(9,4,4), pool(17,8,8)
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), BN(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale0 = nn.Sequential(BN(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        def proc():
            return nn.Sequential(BN(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True),
                                 nn.Conv2d(branch_planes, branch_planes, 3, padding=1, bias=False))
        self.process1, self.process2, self.process3, self.process4 = proc(), proc(), proc(), proc()
        self.compression = nn.Sequential(BN(branch_planes*5, momentum=bn_mom), nn.ReLU(inplace=True),
                                         nn.Conv2d(branch_planes*5, outplanes, 1, bias=False))
        self.shortcut = nn.Sequential(BN(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, outplanes, 1, bias=False))
    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        up = lambda t: F.interpolate(t, size=[h,w], mode='bilinear', align_corners=algc)
        xl = [self.scale0(x)]
        xl.append(self.process1(up(self.scale1(x)) + xl[0]))
        xl.append(self.process2(up(self.scale2(x)) + xl[1]))
        xl.append(self.process3(up(self.scale3(x)) + xl[2]))
        xl.append(self.process4(up(self.scale4(x)) + xl[3]))
        return self.compression(torch.cat(xl, 1)) + self.shortcut(x)

class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BN=nn.BatchNorm2d):
        super().__init__()
        def pool(k, s, p):
            return nn.Sequential(nn.AvgPool2d(k, s, p), BN(inplanes, momentum=bn_mom),
                                 nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale1, self.scale2, self.scale3 = pool(5,2,2), pool(9,4,4), pool(17,8,8)
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), BN(inplanes, momentum=bn_mom),
                                    nn.ReLU(inplace=True), nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale0 = nn.Sequential(BN(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, 1, bias=False))
        self.scale_process = nn.Sequential(BN(branch_planes*4, momentum=bn_mom), nn.ReLU(inplace=True),
                                nn.Conv2d(branch_planes*4, branch_planes*4, 3, padding=1, groups=4, bias=False))
        self.compression = nn.Sequential(BN(branch_planes*5, momentum=bn_mom), nn.ReLU(inplace=True),
                                          nn.Conv2d(branch_planes*5, outplanes, 1, bias=False))
        self.shortcut = nn.Sequential(BN(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
                                      nn.Conv2d(inplanes, outplanes, 1, bias=False))
    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]
        up = lambda t: F.interpolate(t, size=[h,w], mode='bilinear', align_corners=algc)
        x_ = self.scale0(x)
        sl = [up(self.scale1(x))+x_, up(self.scale2(x))+x_, up(self.scale3(x))+x_, up(self.scale4(x))+x_]
        scale_out = self.scale_process(torch.cat(sl, 1))
        return self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)

class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BN=nn.BatchNorm2d):
        super().__init__()
        self.with_channel, self.after_relu = with_channel, after_relu
        self.f_x = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False), BN(mid_channels))
        self.f_y = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1, bias=False), BN(mid_channels))
        if with_channel:
            self.up = nn.Sequential(nn.Conv2d(mid_channels, in_channels, 1, bias=False), BN(in_channels))
        if after_relu:
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x, y):
        sz = x.size()
        if self.after_relu:
            y, x = self.relu(y), self.relu(x)
        y_q = F.interpolate(self.f_y(y), size=[sz[2], sz[3]], mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        if self.with_channel:
            sim = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        y = F.interpolate(y, size=[sz[2], sz[3]], mode='bilinear', align_corners=False)
        return (1 - sim) * x + sim * y

class Light_Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BN=nn.BatchNorm2d):
        super().__init__()
        self.conv_p = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), BN(out_channels))
        self.conv_i = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), BN(out_channels))
    def forward(self, p, i, d):
        edge = torch.sigmoid(d)
        return self.conv_p((1 - edge) * i + p) + self.conv_i(i + edge * p)

class Bag(nn.Module):
    def __init__(self, in_channels, out_channels, BN=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Sequential(BN(in_channels), nn.ReLU(inplace=True),
                                  nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False))
    def forward(self, p, i, d):
        edge = torch.sigmoid(d)
        return self.conv(edge * p + (1 - edge) * i)


# =============================================================================
# 3. PIDNet network  (three branches: P=detail, I=context, D=boundary)
# =============================================================================
class PIDNet(nn.Module):
    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96,
                 head_planes=128, augment=True):
        super().__init__()
        self.augment = augment
        # I Branch (context)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, 2, 1), nn.BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, 2, 1), nn.BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes*2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes*2, planes*4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes*4, planes*8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes*8, planes*8, 2, stride=2)
        # P Branch (detail)
        self.compression3 = nn.Sequential(nn.Conv2d(planes*4, planes*2, 1, bias=False),
                                           nn.BatchNorm2d(planes*2, momentum=bn_mom))
        self.compression4 = nn.Sequential(nn.Conv2d(planes*8, planes*2, 1, bias=False),
                                           nn.BatchNorm2d(planes*2, momentum=bn_mom))
        self.pag3 = PagFM(planes*2, planes)
        self.pag4 = PagFM(planes*2, planes)
        self.layer3_ = self._make_layer(BasicBlock, planes*2, planes*2, m)
        self.layer4_ = self._make_layer(BasicBlock, planes*2, planes*2, m)
        self.layer5_ = self._make_layer(Bottleneck, planes*2, planes*2, 1)
        # D Branch (boundary)
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes*2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(nn.Conv2d(planes*4, planes, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(planes, momentum=bn_mom))
            self.diff4 = nn.Sequential(nn.Conv2d(planes*8, planes*2, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(planes*2, momentum=bn_mom))
            self.spp = PAPPM(planes*16, ppm_planes, planes*4)
            self.dfm = Light_Bag(planes*4, planes*4)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes*2, planes*2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes*2, planes*2)
            self.diff3 = nn.Sequential(nn.Conv2d(planes*4, planes*2, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(planes*2, momentum=bn_mom))
            self.diff4 = nn.Sequential(nn.Conv2d(planes*8, planes*2, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(planes*2, momentum=bn_mom))
            self.spp = DAPPM(planes*16, ppm_planes, planes*4)
            self.dfm = Bag(planes*4, planes*4)
        self.layer5_d = self._make_layer(Bottleneck, planes*2, planes*2, 1)
        # heads
        if self.augment:
            self.seghead_p = segmenthead(planes*2, head_planes, num_classes)
            self.seghead_d = segmenthead(planes*2, planes, 1)
        self.final_layer = segmenthead(planes*4, head_planes, num_classes)
        # init (this is the "from scratch" initialization)
        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1); nn.init.constant_(mod.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes*block.expansion, 1, stride, bias=False),
                                       nn.BatchNorm2d(planes*block.expansion, momentum=bn_mom))
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, stride=1, no_relu=(i == blocks-1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(inplanes, planes*block.expansion, 1, stride, bias=False),
                                       nn.BatchNorm2d(planes*block.expansion, momentum=bn_mom))
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def forward(self, x):
        w_out, h_out = x.shape[-1] // 8, x.shape[-2] // 8
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)
        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + F.interpolate(self.diff3(x), size=[h_out, w_out], mode='bilinear', align_corners=algc)
        if self.augment: temp_p = x_
        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))
        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + F.interpolate(self.diff4(x), size=[h_out, w_out], mode='bilinear', align_corners=algc)
        if self.augment: temp_d = x_d
        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(self.spp(self.layer5(x)), size=[h_out, w_out], mode='bilinear', align_corners=algc)
        x_ = self.final_layer(self.dfm(x_, x, x_d))
        if self.augment:
            return [self.seghead_p(temp_p), x_, self.seghead_d(temp_d)]
        return x_

def build_pidnet(size="s", num_classes=19, augment=True):
    if size == "s":
        return PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96,  head_planes=128, augment=augment)
    if size == "m":
        return PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96,  head_planes=128, augment=augment)
    return PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=augment)

