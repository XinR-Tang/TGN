import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # 第一个卷积层，1*1用来改变通道维数的同时减少参数量
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 第二个卷积层, 3*3的卷积，主要用于提取特征
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 第三个卷积层，1*1用来扩大4被的channel
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # 残差连接
        self.shortcut = nn.Sequential()

        # 为了保证输入输出空间匹配，才可以残差连接
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    # block是用于构建ResNet的残差块类型，这里是Bottleneck, num_blocks是一个列表，表示每个残差层中块的数量
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64   # 初始通道数是64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        用于构建一个残差层。
        :param block: 块的类型
        :param planes: 输出通道数
        :param num_blocks: 块数量
        :param stride: 步长
        :return:
        """
        # strides是一个列表，包含每个块的步长
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

        out = self.layer1(out)
        out = self.layer2(out)
        # 提取最后卷积层的特征
        features = self.layer3(out)
        out = self.layer4(features)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        # 返回最后一个卷积层的特征和最终的预测标签
        return out, features

def load_pretrained_resnet50(model, freeze_layers=True):
    # 加载预训练的 ResNet-50 模型
    pretrained_model = models.resnet50(pretrained=True)

    # 将预训练的权重复制到自定义模型中（除了全连接层）
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_model.state_dict().items() if k in model_dict and 'fc' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if freeze_layers:
        # 冻结特定的层
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        # layer3, layer4 和 fc 层保持可训练状态

if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms

    image = Image.new('RGB', (224, 224))
    # 创建模型实例
    model = ResNet(Bottleneck, [3, 4, 6, 3])

    # load_pretrained_resnet50(model)
    # 定义预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 缩放图像，使其最小边为 256 像素
        transforms.CenterCrop(224),  # 从图像中心裁剪出 224x224 的大小
        transforms.ToTensor(),  # 将 PIL 图像转换成 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 应用预处理
    x = preprocess(image)
    # 添加一个批次维度，因为 PyTorch 的模型期望批次维度存在
    x = x.unsqueeze(0)
    # 输出x的维度
    print(x.shape)

    # 获取预测和最后一个卷积层的特征
    predictions, features = model(x)

    print(predictions.shape, features.shape)