import os
import torch
import torch.nn as nn
from Datasets import FGSCDataset, DIORDataset, DOTADataset
from albumentations.pytorch import ToTensorV2
import model_finetune
import cv2
import albumentations as alb
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from Utils import adjust_learning_rate, test, save_models, test_specific_classes, calculate_f1_score, log_f1_scores

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU

"------------------------------------1.超参数定义-----------------------------------"
lr = 0.001
batchsize = 32
num_classes = 20
weight_decay = 1e-4
num_epochs = 50
train_data_folder = r'/root/autodl-tmp/TGN/dior/train'
test_data_folder = r'/root/autodl-tmp/TGN/dior/test'
train_class_file = r'/root/autodl-tmp/TGN/dior/anno/DIOR_train.txt'
test_class_file = r'/root/autodl-tmp/TGN/dior/anno/DIOR_test.txt'
save_path = './save_model/Model.txt'
save_pathf1 = './save_model/Modelf1.txt'
save_model_path = './save_model/Model/UModel'
result_file = './save_model/Model_all_test.txt'

specific_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                    '18', '19']

"------------------------------------2.创建测试dataloader-------------------------"
test_dataset = DIORDataset(data_folder=test_data_folder,
                           class_file=test_class_file, specific_classes=specific_classes, mode='test')

test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

"------------------------------------3.网络定义-------------------------------------"
# 检查CUDA是否可用
cuda_avail = torch.cuda.is_available()

# 加载模型权重
weights = torch.load(r"/root/autodl-tmp/TGN/Classification/save_model/result.pth")

model = model_finetune.ResNet("resnet50", num_classes)

model.load_state_dict(weights)

U_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()

if cuda_avail:
    model.cuda()


def test(model, test_loader, loss_fn, num_classes):
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, target in test_loader:
            if cuda_avail:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy().flatten())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    f1_scores, overall_f1 = calculate_f1_score(all_labels, all_preds, num_classes)

    return test_loss, accuracy, f1_scores, overall_f1


def calculate_f1_score(labels, predictions, num_classes):
    f1_scores = f1_score(labels, predictions, average=None, labels=range(num_classes))
    overall_f1 = f1_score(labels, predictions, average='macro')
    return f1_scores, overall_f1


if __name__ == "__main__":
    print("start testing")
    test_loss, test_accuracy, f1_scores, overall_f1 = test(model, test_loader, loss_fn, num_classes)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test F1 Scores: {f1_scores}, Overall F1 Score: {overall_f1:.4f}")
