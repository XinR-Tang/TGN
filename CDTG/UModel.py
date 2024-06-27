import os
import torch
import torch.nn as nn
from Datasets import FGSCDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from resnet50 import Bottleneck,ResNet,load_pretrained_resnet50
from Utils import adjust_learning_rate,test,save_models,test_specific_classes

os.environ['CUDA_VISIBLE_DEVICES']='0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU


"------------------------------------1.超参数定义-----------------------------------"
lr = 0.001
batchsize = 32
num_classes = 8
weight_decay=1e-4
num_epochs = 100
train_data_folder = r'/root/autodl-tmp/datasets/dior/train/head'
test_data_folder = r'/root/autodl-tmp/datasets/dior/test/head'
train_class_file = r'/root/autodl-tmp/datasets/dior/train/head/head_classes.txt'
test_class_file = r'/root/autodl-tmp/datasets/dior/test/head/head_classes.txt'
save_path = './save_model1/UModel.txt'
save_model_path = './save_model1/UModel/UModel'
result_file = './save_model1/UModel_all_test.txt'
specific_classes = ['13', '18', '15', '0', '16', '2', '11', '19']

"------------------------------------2.创建头部的训练dataloader和测试dataloader-------------------------"
# 示例：创建训练和测试数据集
train_dataset = FGSCDataset(data_folder=train_data_folder,
                            class_file=train_class_file, specific_classes=specific_classes, mode='train')
test_dataset = FGSCDataset(data_folder=test_data_folder,
                           class_file=test_class_file, specific_classes=specific_classes, mode='test')

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
"------------------------------------3.网络定义-------------------------------------"
cuda_avail = torch.cuda.is_available()
Umodel = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes).to(device)

# 加载预训练权重,并冻结相关的层
load_pretrained_resnet50(Umodel, freeze_layers=False)

# 优化器及损失函数
U_optimizer = torch.optim.Adam(Umodel.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()

if cuda_avail:
    Umodel.cuda()

"------------------------------------4.训练-------------------------------------"
def train(num_epochs, save_path):
    with open(save_path, 'a') as f:
        for epoch in range(num_epochs):
            Umodel.train()
            train_acc = 0.0
            train_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                if cuda_avail:
                    images = images.to(device)
                    labels = labels.to(device)

                U_optimizer.zero_grad()
                outputs, f1 = Umodel(images)

                loss = loss_fn(outputs, labels)
                loss.backward()
                U_optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, prediction = torch.max(outputs.data, 1)
                train_acc += torch.sum(prediction == labels.data)

            adjust_learning_rate(epoch, U_optimizer)

            train_acc /= len(train_loader.dataset)
            train_loss /= len(train_loader.dataset)

            Umodel.eval()  # 切换到测试模式
            test_loss, test_acc = test(Umodel, test_loader,loss_fn)

            if epoch % 10 == 0:
                save_models(epoch, Umodel, save_model_path)  # 确保所有必要参数都被传递

            f.write(f"Epoch:{epoch}, Train_loss:{train_loss}, Train_acc:{train_acc}, Test_loss:{test_loss}, Test_acc:{test_acc}\n")
            print(f"Epoch:{epoch}, Train_loss:{train_loss}, Train_acc:{train_acc}, Test_loss:{test_loss}, Test_acc:{test_acc}")


if __name__ == '__main__':
    train(num_epochs, save_path)
    test_specific_classes(Umodel, test_loader, result_file, specific_classes)
