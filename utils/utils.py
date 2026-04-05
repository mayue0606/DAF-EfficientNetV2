import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(y_true, y_pred, num_classes):
    """
    手写混淆矩阵实现
    Args:
        y_true: list or np.array, 真实标签
        y_pred: list or np.array, 预测标签
        num_classes: 类别数
    Returns:
        cm: (num_classes x num_classes) 矩阵
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def precision_recall_f1(cm):
    """
    根据混淆矩阵计算 macro 平均的 precision / recall / f1
    Args:
        cm: 混淆矩阵
    Returns:
        precision, recall, f1
    """
    eps = 1e-12  # 防止除零
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP

    precision_per_class = TP / (TP + FP + eps)
    recall_per_class = TP / (TP + FN + eps)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + eps)

    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1 = np.mean(f1_per_class)

    return float(precision), float(recall), float(f1)


# def plot_confusion_matrix(cm, class_names, save_path=None, title="Confusion Matrix"):
#     """
#     绘制混淆矩阵
#     Args:
#         cm: 混淆矩阵
#         class_names: 类别名称列表
#         save_path: 保存路径
#         title: 图像标题
#     """
#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     ax.figure.colorbar(im, ax=ax)
#     ax.set(xticks=np.arange(len(class_names)),
#            yticks=np.arange(len(class_names)),
#            xticklabels=class_names,
#            yticklabels=class_names,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # 显示每个格子的数值
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], 'd'),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#
#     fig.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#         plt.close()
#     else:
#         plt.show()
