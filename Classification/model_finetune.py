from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision import models
import pretrainedmodels
import torch
import torch.nn.functional as F

def VGG(arch_name, num_classes, pretrained=True):
    assert "vgg" in arch_name
    backbone = models.__dict__[arch_name](pretrained)
    # 修改分类层
    # backbone.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
    # backbone.classifier[-1] =nn.Linear(
    #         backbone.classifier[-1].in_features, num_classes)

    backbone.classifier = nn.Sequential(
        nn.Linear(
            backbone.classifier[0].in_features, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(in_features=1024, out_features=num_classes)
    )

    return backbone


def ResNet(arch_name, num_classes, pretrained=True):
    assert "resnet" in arch_name

    backbone = models.__dict__[arch_name](pretrained)
    backbone.fc = nn.Sequential(
        # nn.Dropout(0.5),
        nn.Linear(
            backbone.fc.in_features, num_classes))
    return backbone


def Efficient(arch_name, num_classes, pretrained=True):
    assert "efficientnet-b" in arch_name
    # 'efficientnet-b{N}' for N=0,1,2,3,4,5,6,7
    backbone = EfficientNet.from_pretrained(arch_name) if pretrained else EfficientNet.from_name(arch_name)
    backbone._fc = nn.Linear(backbone._fc.in_features, num_classes)
    return backbone


def SENet(arch_name, n_classes, pretrained=True):
    # 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d'
    pretrained = "imagenet" if pretrained else None
    model = pretrainedmodels.__dict__[arch_name](pretrained=pretrained)
    model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)
    return model


def DenseNet(arch_name, num_classes, pretrained=True):
    net = models.__dict__[arch_name](pretrained)
    net.classifier = nn.Linear(net.classifier.in_features, num_classes)
    return net


def inception_resnet(n_classes, pretrained=True):
    # 299, 299
    pretrained = "imagenet" if pretrained else None
    net = pretrainedmodels.models.inceptionresnetv2(pretrained=pretrained)
    net.last_linear = nn.Linear(net.last_linear.in_features, n_classes)
    return net


if __name__ == '__main__':
    net = VGG(arch_name="vgg16", num_classes=30)
    # net = ResNet(arch_name="resnet34", num_classes=30)
    # net = Efficient("efficientnet-b0",num_classes=30)

    # net = SENet("se_resnext50_32x4d",30,False)

    # net = DenseNet("densenet121",num_classes=30)

    # net = inception_resnet(30,False)

    print(net)
    x = torch.zeros((1, 3, 299, 299))
    print(net(x).shape)
