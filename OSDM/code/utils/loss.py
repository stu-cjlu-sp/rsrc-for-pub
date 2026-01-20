import torch


def calc_loss(cate_true, det_true, cls_pred, det_pred, cls_loss_fn, reg_loss_fn):
    scale = 1
    len = cate_true.size(1)
    cls_pred = cls_pred.transpose(1, 2)
    cls_loss = cls_loss_fn(cls_pred, cate_true).sum() / len
    det_loss = reg_loss_fn(det_pred * scale, det_true * scale).sum() / len
    return cls_loss, det_loss