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
save_model_path = './save_model/Model/Model'
result_file = './save_model/Model_all_test.txt'

specific_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']

if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

"------------------------------------2.创建头部的训练dataloader和测试dataloader-------------------------"
train_dataset = DIORDataset(data_folder=train_data_folder,
                            class_file=train_class_file, specific_classes=specific_classes, mode='train')
test_dataset = DIORDataset(data_folder=test_data_folder,
                           class_file=test_class_file, specific_classes=specific_classes, mode='test')

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

"------------------------------------3.网络定义-------------------------------------"
# 检查CUDA是否可用
cuda_avail = torch.cuda.is_available()

# 创建模型实例
model = model_finetune.ResNet("resnet50", num_classes)

U_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()

if cuda_avail:
    model.cuda()

"------------------------------------4.训练-------------------------------------"
def train(num_epochs, save_path):
    print("start training...")
    with open(save_path, 'a') as f:
        for epoch in range(num_epochs):
            model.train()
            train_acc = 0.0
            train_loss = 0.0
            all_labels = []
            all_predictions = []

            for i, (images, labels) in enumerate(train_loader):
                if cuda_avail:
                    images = images.to(device)
                    labels = labels.to(device)

                U_optimizer.zero_grad()
                outputs = model(images)

                loss = loss_fn(outputs, labels)
                loss.backward()
                U_optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, prediction = torch.max(outputs.data, 1)
                train_acc += torch.sum(prediction == labels.data)

                all_labels.extend(labels.cpu())
                all_predictions.extend(prediction.cpu())

            adjust_learning_rate(epoch, U_optimizer)

            train_acc /= len(train_loader.dataset)
            train_loss /= len(train_loader.dataset)

            all_labels = torch.tensor(all_labels)
            all_predictions = torch.tensor(all_predictions)

            f1_scores, overall_f1 = calculate_f1_score(all_labels, all_predictions, num_classes)
            log_f1_scores(epoch, f1_scores, overall_f1, save_path)

            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {overall_f1:.4f}")
            f.write(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, F1 Score: {overall_f1:.4f}\n")

            test_loss, test_accuracy = test(model, test_loader, loss_fn)
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")
            f.write(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}\n")

            save_models(epoch, model, save_model_path)

if __name__ == "__main__":
    train(num_epochs, save_path)

    test_loss, test_accuracy = test(model, test_loader, loss_fn)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")