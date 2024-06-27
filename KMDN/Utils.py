import os
import torch
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置当前使用的GPU设备仅为0号设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 让torch判断是否使用GPU

cuda_avail = torch.cuda.is_available()

# 动态更新学习率
def adjust_learning_rate(epoch, optimizer):
    lr = 0.001
    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test(model, test_loader, loss_fn):
    model.eval()  # 确保模型处于评估模式
    test_loss = 0.0
    test_accuracy = 0.0
    total = 0

    with torch.no_grad():  # 在测试阶段不计算梯度
        for images, labels in test_loader:
            if cuda_avail:
                images = images.to(device)
                labels = labels.to(device)

            outputs, _ = model(images)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item() * images.size(0)

            _, prediction = torch.max(outputs.data, 1)
            test_accuracy += torch.sum(prediction == labels.data).item()
            total += labels.size(0)

    test_loss /= total
    test_accuracy /= total
    return test_loss, test_accuracy

def save_models(epoch, model, filename):
    torch.save(model.state_dict(), f"{filename}_epoch_{epoch}.pth")
    print(f"Checkpoint saved at epoch {epoch}")

def test_specific_classes(model, test_loader, result_file, specific_classes):
    model.eval()  # 确保模型处于评估模式

    class_map = {cls: idx for idx, cls in enumerate(specific_classes)}
    reverse_class_map = {idx: cls for cls, idx in class_map.items()}

    class_correct = {cls: 0 for cls in specific_classes}
    class_total = {cls: 0 for cls in specific_classes}

    with torch.no_grad():
        for images, labels in test_loader:
            if cuda_avail:
                images = images.to(device)
                labels = labels.to(device)

            outputs, _ = model(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                original_label = reverse_class_map[label.item()]
                original_prediction = reverse_class_map[prediction.item()]

                if original_label in specific_classes:
                    class_total[original_label] += 1
                    if original_label == original_prediction:
                        class_correct[original_label] += 1

    with open(result_file, 'a') as file:
        for cls in specific_classes:
            accuracy = 100.0 * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            file.write(f"Class {cls} Accuracy: {accuracy:.2f}%\n")

        overall_accuracy = 100.0 * sum(class_correct.values()) / sum(class_total.values()) if sum(class_total.values()) > 0 else 0
        file.write(f"\nOverall Accuracy (Specific Classes): {overall_accuracy:.2f}%\n")

def calculate_f1_score(labels, predictions, num_classes):
    f1_scores = f1_score(labels.cpu(), predictions.cpu(), average=None, labels=range(num_classes))
    overall_f1 = f1_score(labels.cpu(), predictions.cpu(), average='macro')
    return f1_scores, overall_f1

def log_f1_scores(epoch, f1_scores, overall_f1, file_path):
    with open(file_path, 'a') as f:
        f.write(f"Epoch {epoch}:\n")
        for idx, score in enumerate(f1_scores):
            f.write(f"Class {idx} F1 Score: {score:.4f}\n")
        f.write(f"Overall F1 Score: {overall_f1:.4f}\n\n")

def plot_confusion_matrix(labels, predictions, classes, file_path):
    cm = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy())
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) * 100  # Convert to percentages

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.close()
