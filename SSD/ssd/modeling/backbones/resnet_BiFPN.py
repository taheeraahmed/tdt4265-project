import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import OrderedDict, Tuple, List
import torchvision.models as models
import torchvision.ops as ops


from torch.autograd import Variable
class Layer(nn.Sequential):
    def __init__(self,in_channels,out_channels):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0),
            nn.ReLU(),
        )


class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    From: https://github.com/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    From: https://github.com/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)


class ResNet(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    
    """
    def __init__(self,
            output_channels: List[int],
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        # Get pretrained Retina Network
        self.model = models.resnet34(pretrained=True)
        
        # Create two more layers
        self.layer5 = Layer(512, 256)
        self.layer6 = Layer(256, 256)
        
        
    def forward_first_layer(self, model, image):
        """Executing forward pass for the zeroth Retina Net layer"""
        x = model.conv1(image)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        return x

    def forward(self, x):
        """
        Performing forward pass for a layer at the time and saving every output in an array. 
        The forward functiom should output features with shape:
            [shape(-1, 256, 32, 256),
            shape(-1, 512, 16, 128),
            shape(-1, 1024, 8, 64),
            shape(-1, 2048, 4, 32),
            shape(-1, 2048, 2, 16),
            shape(-1, 2048, 1, 8)]
        When done, the array of outputs is passed into the FPN and the outputs from FPN is returned
        """
        out_features = []
        features_dict = OrderedDict()



        
        # Layer 0
        x = self.forward_first_layer(self.model,x)


        # Layer 1
        feat0 = self.model.layer1(x)
        
        # Layer 2
        feat1 = self.model.layer2(feat0)

        # Layer 3
        feat2 = self.model.layer3(feat1)

        # Layer 4
        feat3 = self.model.layer4(feat2)

        # Layer 5
        feat4 = self.layer5(feat3)
        
        # Layer 6
        feat5 = self.layer6(feat4)


        out_features = [feat0, feat1, feat2, feat3, feat4, feat5]
        

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)


class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    
    From: https://github.com/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        

        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        

        self.w1 = nn.Parameter(torch.normal(0, 1, size = (2, 4)))
        self.w1_relu = nn.ReLU(inplace=False)
        self.w2 = nn.Parameter(torch.normal(0, 1, size = (3, 4)))
        self.w2_relu = nn.ReLU(inplace=False)

    
    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x = inputs
        
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon) 
        w2 = self.w2_relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        

        # TODO: We forgot to add the last P 
        p7_td = p7_x  

        p6_td = self.p6_td(w1[0, 0] * p6_x + w1[1, 0] * F.interpolate(p7_td, scale_factor=2))       
        p5_td = self.p5_td(w1[0, 1] * p5_x + w1[1, 1] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * F.interpolate(p4_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))
        p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * nn.Upsample(scale_factor=0.5)(p6_out))


        return [p3_out, p4_out, p5_out, p6_out, p7_out]
    
class BiFPN(nn.Module):
  """
  From: https://github.com/tristandb/EfficientDet-PyTorch/blob/master/bifpn.py
  """
    
  def __init__(self, out_channels,output_feature_sizes, feature_size = 64, num_layers=3):
        super(BiFPN, self).__init__()

        self.output_feature_sizes = output_feature_sizes

        self.resnet = ResNet(out_channels, output_feature_sizes = self.output_feature_sizes )
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.kernel_size = 1


        self.p0 = nn.Conv2d(out_channels[0], self.feature_size, 1, 1, 0)
        self.p1 = nn.Conv2d(out_channels[1], self.feature_size, 1, 1, 0)
        self.p2 = nn.Conv2d(out_channels[2], self.feature_size, 1, 1, 0)
        self.p3 = nn.Conv2d(out_channels[3], self.feature_size, 1, 1, 0)
        self.p4 = nn.Conv2d(out_channels[4], self.feature_size, 1, 1, 0)
        self.p5 = nn.Conv2d(out_channels[5], self.feature_size, 1, 1, 0)

        self.out_channels = [self.feature_size] * 6

        bifpns = []
        for _ in range(self.num_layers):
            bifpns.append(BiFPNBlock(self.feature_size))
        self.bifpn = nn.Sequential(*bifpns)

    
  def forward(self, inputs):

        (x0, x1, x2, x3, x4, x5) = self.resnet.forward(inputs)
        
        x0 = self.p0(x0)
        x1 = self.p1(x1)
        x2 = self.p2(x2)
        x3 = self.p3(x3)
        x4 = self.p4(x4)
        x5 = self.p5(x5)

        features = [x1, x2, x3, x4, x5]

        out_features_BiFPN = self.bifpn(features)


        out_features = []
        for val in out_features_BiFPN:
            out_features.append(val)

        return out_features
