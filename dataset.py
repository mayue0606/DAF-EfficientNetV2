from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

image_size = (384,384)  # 这里可以替换成你自己的大小

# 数据增强 、还有数据张量化  在线数据增强 数据增强管道通过多种随机变换
transform_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=20,
        translate=(0.2, 0.2),
        scale=(0.8, 1.2),
        shear=0,
        fill=0
    ),
    # 保证数值稳定性
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

transform_val = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

class TrainDataset(Dataset):
    def __init__(self, root_dir, transform=transform_train):
        self.root_dir = root_dir  # 记住数据位置
        self.transform = transform


        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        # print(self.classes)
        # 创建类别到索引的映射
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        # 记录每张图片的已知类型
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)
    # 准备好数据
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class ValDataset(Dataset):
    def __init__(self, root_dir, transform=transform_val):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])   # 通过目录获取类
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}


        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    batch_size = 16
    num_workers = 0
    train_dataset = TrainDataset(root_dir='data/train')

    for img,label in train_dataset:
        print(img)
        print(label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    # b(batch_size) c(channels) h(height) w(width)