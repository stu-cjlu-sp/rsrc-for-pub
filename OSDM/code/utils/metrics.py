import numpy as np
import torch


def calc_acc(cate_pred, cate_true):
    assert cate_pred.shape == cate_true.shape, "预测值和真实值的形状不一致"
    correct = cate_pred == cate_true
    acc_per_label  = torch.mean(correct.to(torch.float32), dim=0)
    avg_acc  = torch.mean(acc_per_label )
    return avg_acc


def calc_iou(pred, true):
    s_p, e_p = pred
    s_t, e_t = true
    inter_s = max(s_p, s_t)
    inter_e = min(e_p, e_t)
    inter = max(0, inter_e - inter_s)
    union_s = min(s_p, s_t)
    union_e = max(e_p, e_t)
    union = union_e - union_s
    iou = inter / union if union > 0 else 0
    return iou


def calc_pd(det_pred, det_true, iou_thresh=0.7):
    tp = 0
    ap = 0
    for pred, true in zip(det_pred, det_true):
        for p, t in zip(pred, true):
            if np.any(t != 0):
                ap += 1
                iou = calc_iou(p, t)
                if iou >= iou_thresh:
                    tp += 1
    pd = tp / ap if ap > 0 else 0
    return pd


def calc_pfa(det_pred, det_true):
    fp = 0
    an = 0
    for det_pred, det_true in zip(det_pred, det_true):
        for p, t in zip(det_pred, det_true):
            if np.all(t == 0):
                an += 1
                if np.any(p != 0):
                    fp += 1
    pfa = fp / an if an > 0 else 0
    return pfa