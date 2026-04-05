import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         precisions, recalls, f1s, save_dir):
    """
    绘制训练过程的损失、准确率、P/R/F1曲线
    """
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()

    # Precision / Recall / F1
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, precisions, label='Precision')
    plt.plot(epochs_range, recalls, label='Recall')
    plt.plot(epochs_range, f1s, label='F1-score')
    plt.title('P/R/F1 Curve')
    plt.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Training curves saved to {save_path}")


def plot_confusion_matrix(cm, class_names, save_path=None, title="Confusion Matrix"):
    """
    绘制混淆矩阵
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
        save_path: 保存路径
        title: 图像标题
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # 显示每个格子的数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None,
                         precisions=None, recalls=None, f1s=None, save_dir="runs/train/exp"):
    """
    绘制训练过程曲线，包括 Loss、Accuracy、Precision、Recall、F1。
    参数:
        train_losses, val_losses: list of float
        train_accs, val_accs: list of float (可选)
        precisions, recalls, f1s: list of float (可选)
        save_dir: 保存图片的目录
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    # 绘制 Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # 绘制 Accuracy
    if train_accs is not None and val_accs is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_accs, 'b-', label='Train Acc')
        plt.plot(epochs, val_accs, 'r-', label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
        plt.close()

    # 绘制 Precision/Recall/F1
    if precisions is not None and recalls is not None and f1s is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, precisions, 'g-', label='Precision')
        plt.plot(epochs, recalls, 'm-', label='Recall')
        plt.plot(epochs, f1s, 'c-', label='F1')
        plt.title('Precision/Recall/F1 Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "prf1_curve.png"))
        plt.close()