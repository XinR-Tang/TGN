import os
import torch
import torch.nn as nn
from Datasets import FGSCDataset, DOTADataset, DIORDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from resnet50 import Bottleneck, ResNet, load_pretrained_resnet50
from Utils import adjust_learning_rate, test, save_models, test_specific_classes

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU

"------------------------------------1.超参数定义-----------------------------------"
lr = 0.001
alpha = 0.2
batchsize = 64
weight_decay = 1e-4
num_epochs = 100
distill_interval = 50

head_num_classes = 8
head_train_data_folder = r'/root/autodl-tmp/TGN/KMDN/datasets/dior/train/head'
head_test_data_folder = r'/root/autodl-tmp/TGN/KMDN/datasets/dior/test/head'
head_train_class_file = r'/root/autodl-tmp/TGN/KMDN/datasets/dior/train/head/head_classes.txt'
head_test_class_file = r'/root/autodl-tmp/TGN/KMDN/datasets/dior/test/head/head_classes.txt'
head_specific_classes = ['13', '18', '15', '0', '16', '2', '11', '19']

tail_num_classes = 12
tail_train_data_folder = r'/root/autodl-tmp/TGN/KMDN/datasets/dior/train/tail'
tail_test_data_folder = r'/root/autodl-tmp/TGN/KMDN/datasets/dior/test/tail'
tail_train_class_file = r'/root/autodl-tmp/TGN/KMDN/datasets/dior/train/tail/tail_classes.txt'
tail_test_class_file = r'/root/autodl-tmp/TGN/KMDN/datasets/dior/test/tail/tail_classes.txt'
tail_specific_classes = ['4', '3', '10', '12', '7', '5', '8', '14', '1', '9', '6', '17']

UModel_pth = '/root/autodl-tmp/TGN/KMDN/save_model_head/UModel/UModel_epoch_0.pth'
RModel_pth = '/root/autodl-tmp/TGN/KMDN/save_model_tail/RModel/RModel_epoch_0.pth'
save_model_path_head = './save_model_dill/UModel/UModel'
save_model_path_tail = './save_model_dill/RModel/RModel'
Head_result_file = './save_model_dill/DH.txt'
tail_result_file = './save_model_dill/DT.txt'

if not os.path.exists("save_model_dill"):
    os.makedirs("save_model_dill")

if not os.path.exists(save_model_path_head):
    os.makedirs(save_model_path_head)

if not os.path.exists(save_model_path_tail):
    os.makedirs(save_model_path_tail)
"------------------------------------2.创建头部的训练dataloader和测试dataloader-------------------------"
# 示例：创建训练和测试数据集
head_train_dataset = DIORDataset(data_folder=head_train_data_folder,
                                 class_file=head_train_class_file, specific_classes=head_specific_classes, mode='train')
head_test_dataset = DIORDataset(data_folder=head_test_data_folder,
                                class_file=head_test_class_file, specific_classes=head_specific_classes, mode='test')

# 创建数据加载器
head_train_loader = DataLoader(head_train_dataset, batch_size=batchsize, shuffle=True)
head_test_loader = DataLoader(head_test_dataset, batch_size=batchsize, shuffle=False)

"------------------------------------3.创建尾部的训练dataloader和测试dataloader-------------------------"
# 示例：创建训练和测试数据集
tail_train_dataset = DIORDataset(data_folder=tail_train_data_folder,
                                 class_file=tail_train_class_file, specific_classes=tail_specific_classes, mode='train')
tail_test_dataset = DIORDataset(data_folder=tail_test_data_folder,
                                class_file=tail_test_class_file, specific_classes=tail_specific_classes, mode='test')

# 创建数据加载器
tail_train_loader = DataLoader(tail_train_dataset, batch_size=batchsize, shuffle=True)
tail_test_loader = DataLoader(tail_test_dataset, batch_size=batchsize, shuffle=False)

"------------------------------------4.网络定义-------------------------------------"
cuda_avail = torch.cuda.is_available()
Umodel = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=head_num_classes).to(device)
Rmodel = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=tail_num_classes).to(device)

state_dict = torch.load(UModel_pth, map_location=torch.device('cpu'))
# 加载不涉及全连接层的权重
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
# 现在加载状态字典
Umodel.load_state_dict(state_dict, strict=False)

state_dict = torch.load(RModel_pth, map_location=torch.device('cpu'))
# 加载不涉及全连接层的权重
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
# 现在加载状态字典
Rmodel.load_state_dict(state_dict, strict=False)

# 优化器及损失函数
U_optimizer = torch.optim.Adam(Umodel.parameters(), lr=lr, weight_decay=weight_decay)
R_optimizer = torch.optim.Adam(Rmodel.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn1 = torch.nn.MSELoss(reduce=True, size_average=True)
loss_fn2 = nn.CrossEntropyLoss()

if cuda_avail:
    Umodel.cuda()

"------------------------------------5.训练----------------------------------------"


def train(num_epochs, distill_interval=distill_interval):
    # 创建两个不同的文件来存储训练结果
    head_file = open('./save_model_dill/Head_JieGuo.txt', 'a')
    tail_file = open('./save_model_dill/Tail_JieGuo.txt', 'a')

    for epoch in range(num_epochs):
        # 确定当前轮次的教师模型和学生模型，以及相应的数据加载器
        if epoch // distill_interval % 2 == 0:
            teacher_model, student_model = Rmodel, Umodel
            teacher_optimizer, student_optimizer = R_optimizer, U_optimizer
            teacher_loader, student_loader = tail_train_loader, head_train_loader
            current_file = head_file
            model = "Head"
            if epoch == 0:
                print("开始尾部向头部蒸馏！！")
        else:
            teacher_model, student_model = Umodel, Rmodel
            teacher_optimizer, student_optimizer = U_optimizer, R_optimizer
            teacher_loader, student_loader = head_train_loader, tail_train_loader
            current_file = tail_file
            model = "Tail"
            if epoch == 50:
                print("开始头部向尾部蒸馏！！")
        # 在每个 epoch 开始时调整学习率
        adjust_learning_rate(epoch, student_optimizer)
        teacher_model.eval()
        student_model.train()

        train_acc = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(student_loader):
            if cuda_avail:
                images = images.to(device)
                labels = labels.to(device)

            student_optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs, teacher_f = teacher_model(images)

            student_outputs, student_f = student_model(images)

            # 注意：这里假设您的模型返回的是最后的分类输出
            # 如果您的模型返回了其他类型的特征，您需要相应地调整这部分
            loss_ce = loss_fn2(student_outputs, labels)
            loss_distill = loss_fn1(student_f, teacher_f)
            loss = loss_ce + alpha * loss_distill
            loss.backward()
            student_optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, prediction = torch.max(student_outputs.data, 1)
            train_acc += torch.sum(prediction == labels.data).item()

        train_acc /= len(student_loader.dataset)
        train_loss /= len(student_loader.dataset)

        # 这里假设test函数适用于您的模型
        test_loss, test_acc = test(student_model, head_test_loader if model == "Head" else tail_test_loader, loss_fn2)
        if epoch <= 50:
            save_models(epoch, Umodel, save_model_path_head)  # 确保所有必要参数都被传递
        else:
            save_models(epoch, Rmodel, save_model_path_tail)
        # 将结果写入当前使用的采样策略对应的文件
        current_file.write(
            f"Epoch:{epoch}, Train_loss:{train_loss}, Train_acc:{train_acc}, Test_loss:{test_loss}, Test_acc:{test_acc}\n")
        print(
            f"Epoch:{epoch}, Train_loss:{train_loss}, Train_acc:{train_acc}, Test_loss:{test_loss}, Test_acc:{test_acc}\n")

    head_file.close()
    tail_file.close()


if __name__ == '__main__':
    train(num_epochs)
    test_specific_classes(Umodel, head_test_loader, Head_result_file, head_specific_classes)
    test_specific_classes(Rmodel, tail_test_loader, tail_result_file, tail_specific_classes)