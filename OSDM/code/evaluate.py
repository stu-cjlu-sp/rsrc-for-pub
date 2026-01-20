import torch
import numpy as np
from utils import calc_pd, calc_pfa, calc_acc


def evaluate(val_data, vocab_size, max_len, model, device):
    all_det_pred = []
    all_cate_pred = []
    all_pd = []
    all_pfa = []
    all_cate_logits = []

    for x7, det_true, _ in val_data:
        x7 = x7.to(device)
        det_true = det_true.to(device)
        cate_pred, det_pred, cate_logits = model(x7, training=False)
        all_cate_logits.append(cate_logits.detach().cpu().numpy())

        det_pred = torch.round(det_pred * vocab_size).detach().cpu().numpy()
        cate_pred = torch.argmax(cate_pred, dim=-1).to(torch.int32).detach().cpu().numpy()
        
        for i in range(det_pred.shape[0]):
            for j in range(1, det_pred.shape[1]):
                if det_pred[i, j, 0] < det_pred[i, j - 1, 0] or det_pred[i, j, 1] < det_pred[i, j - 1, 1]:
                    det_pred[i, j] = [0, 0]

        pd = calc_pd(det_pred, det_true.cpu().numpy())
        pfa = calc_pfa(det_pred, det_true.cpu().numpy())

        all_det_pred.extend(det_pred)
        all_cate_pred.extend(cate_pred)
        all_pd.append(pd)
        all_pfa.append(pfa)

    all_det_pred = np.array(all_det_pred)
    all_cate_pred = np.array(all_cate_pred)
    avg_pd = np.mean(all_pd)
    avg_pfa = np.mean(all_pfa)
    all_cate_logits = np.concatenate(all_cate_logits, axis=0)

    return all_det_pred, all_cate_pred, avg_pd, avg_pfa, all_cate_logits