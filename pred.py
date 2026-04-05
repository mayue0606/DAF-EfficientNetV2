import os
import torch
from PIL import Image
from torchvision import transforms
from effnetv2 import effnetv2_m
from dataset import ValDataset   # 复用类名获取标签顺序（保证与训练一致）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理 (与验证集一致)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])


def load_model(weight_path, num_classes):
    model = effnetv2_m(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def predict_image(model, img_path, class_names):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        pred_class = class_names[pred.item()]
    return pred_class


def predict_folder(model, folder_path, class_names):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder_path, fname)
            pred = predict_image(model, img_path, class_names)
            print(f"{fname} --> {pred}")


if __name__ == "__main__":
    # ---------------------
    # 修改为你训练生成的 best.pth 路径
    # ---------------------
    weight_path = r"/runs/train/exp18/weights/best.pth"

    dataset = ValDataset("../data/val")
    class_names = dataset.classes
    num_classes = len(class_names)

    model = load_model(weight_path, num_classes)

    img_path = r"/data/val\multiple_diseases\Train_245.jpg"
    if os.path.exists(img_path):
        pred = predict_image(model, img_path, class_names)
        print(f"预测结果：{pred}")

    # ---------------------
    # ② 批量预测图片文件夹（可选）
    # ---------------------
    # predict_folder(model, "test_images", class_names)
