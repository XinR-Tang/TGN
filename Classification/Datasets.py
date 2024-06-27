import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FGSCDataset(Dataset):
    def __init__(self, data_folder, class_file, specific_classes, mode='train'):
        self.data_folder = data_folder
        self.mode = mode
        self.specific_classes = specific_classes
        # 读取类别文件并存储路径及类别
        self.img_labels = []
        with open(class_file, 'r') as file:
            for line in file:
                image_path, label = line.strip().split(' ')
                full_path = os.path.join(self.data_folder, image_path)
                self.img_labels.append((full_path, int(label)))
        self.class_map = {cls: idx for idx, cls in enumerate(self.specific_classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.img_labels[idx][0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx][1]
        label = self.class_map[str(label)]
        # 应用不同的变换
        if self.mode == "train":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([.5], [.5])
            ])
        elif self.mode == "test":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([.5], [.5])
            ])

        image = transform(image)
        return image, label


class DIORDataset(Dataset):
    def __init__(self, data_folder, class_file, specific_classes, mode='train'):
        self.data_folder = data_folder
        self.mode = mode
        self.specific_classes = specific_classes
        # 读取类别文件并存储路径及类别
        self.img_labels = []
        with open(class_file, 'r') as file:
            for line in file:
                image_path, label = line.strip().split(' ')
                full_path = os.path.join(self.data_folder, image_path)
                self.img_labels.append((full_path, int(label)))
        self.class_map = {cls: idx for idx, cls in enumerate(self.specific_classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.img_labels[idx][0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx][1]
        label = self.class_map[str(label)]
        # 应用不同的变换
        if self.mode == "train":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([.5], [.5])
            ])
        elif self.mode == "test":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([.5], [.5])
            ])

        image = transform(image)
        return image, label

class DOTADataset(Dataset):
    def __init__(self, data_folder, class_file, specific_classes, mode='train'):
        self.data_folder = data_folder
        self.mode = mode
        self.specific_classes = specific_classes
        # 读取类别文件并存储路径及类别
        self.img_labels = []
        with open(class_file, 'r') as file:
            for line in file:
                image_path, label = line.strip().split(' ')
                full_path = os.path.join(self.data_folder, image_path)
                self.img_labels.append((full_path, int(label)))
        self.class_map = {cls: idx for idx, cls in enumerate(self.specific_classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_folder, self.img_labels[idx][0])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx][1]
        label = self.class_map[str(label)]
        # 应用不同的变换
        if self.mode == "train":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([.5], [.5])
            ])
        elif self.mode == "test":
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([.5], [.5])
            ])

        image = transform(image)
        return image, label

if __name__ == '__main__':
    train_data_folder = r'/root/autodl-tmp/datasets/FGSC/train/head'
    test_data_folder = r'/root/autodl-tmp/datasets/FGSC/test/head'
    train_class_file = r'/root/autodl-tmp/datasets/FGSC/train/head/head_classes.txt'
    test_class_file = r'/root/autodl-tmp/datasets/FGSC/test/head/head_classes.txt'
    specific_classes = ['2', '0', '17', '4', '6', '10', '13']
    # 创建数据集实例时也要按照新的参数顺序传递参数
    train_dataset = FGSCDataset(data_folder=train_data_folder,
                                class_file=train_class_file, specific_classes=specific_classes, mode='train')
    test_dataset = FGSCDataset(data_folder=test_data_folder,
                               class_file=test_class_file, specific_classes=specific_classes, mode='test')

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    images, label = next(iter(train_loader))
    print(images.shape)
    print(label.shape)
