import torch
import numpy as np

def eval_metrics(output, target, num_classes,conf_matrix):
    _, predict = output.max(1)

    c_predict=predict.cpu().numpy().flatten()
    c_target=target.cpu().numpy().flatten()
    for i in range(len(c_predict)):
        conf_matrix[c_predict[i],c_target[i]] += 1

    predict = predict.long() + 1
    target = target.long() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), num_classes, 1, num_classes)
    area_pred = torch.histc(predict.float(), num_classes, 1, num_classes)
    area_lab = torch.histc(target.float(), num_classes, 1, num_classes)
    area_union = area_pred + area_lab - area_inter


    correct = np.round(pixel_correct.cpu().numpy(), 5)
    labeld = np.round(pixel_labeled.cpu().numpy(), 5)
    inter = np.round(area_inter.cpu().numpy(), 5)
    union = np.round(area_union.cpu().numpy(), 5)

    #pixacc = 1.0 * correct / (np.spacing(1) + labeld)
    #mIoU = (1.0 * inter / (np.spacing(1) + union)).mean()
    return correct, labeld, inter, union,conf_matrix