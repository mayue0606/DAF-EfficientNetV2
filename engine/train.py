from tqdm import tqdm
from utils import confusion_matrix, precision_recall_f1, plot_confusion_matrix
import torch


def train_one_epoch(model, device, num_epochs, epoch, train_loader, optimizer, criterion):
    model.train()  # 验证模式
    running_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Train", ncols=100)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # 此处的image 是一整个batch的image，输入的图片<=batch_size  33:total  1:32 2:1
        loss = criterion(outputs, labels)  # 计算预测与真实标签的差异

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=running_loss / total, acc=100. * correct / total)

    return running_loss / total, 100. * correct / total


def evaluate(model, device, test_loader, criterion):
    model.eval()  # 评估模式
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 自定义指标计算
    num_cls = model.classifier.out_features
    cm = confusion_matrix(all_labels, all_preds, num_cls)
    precision_val, recall_val, f1_val = precision_recall_f1(cm)

    return running_loss / total, 100. * correct / total, precision_val, recall_val, f1_val, all_labels, all_preds
