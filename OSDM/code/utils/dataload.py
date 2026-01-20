import scipy.io as scio
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_train_data(signal_path, det_path, cate_path, batch_size=256):
    mat_data = [scio.loadmat(p) for p in [signal_path, det_path, cate_path]]
    signals = mat_data[0]['data']
    det_labels = mat_data[1]['label_box']
    cls_labels = mat_data[2]['label_cate']
    
    idx = np.random.permutation(len(signals))
    signals, det_labels, cls_labels = signals[idx], det_labels[idx], cls_labels[idx]
    
    inp_vocab_size = int(np.max(signals) + 1)
    det_vocab_size = int(np.max(det_labels))
    num_cls = int(np.max(cls_labels) + 1)

    tensors = [
        torch.tensor(signals, dtype=torch.float32),
        torch.tensor(det_labels / det_vocab_size, dtype=torch.float32),
        torch.tensor(cls_labels.astype(np.int16), dtype=torch.int64)
    ]

    train_ds = TensorDataset(*tensors)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return train_loader, inp_vocab_size, det_vocab_size, num_cls


def load_val_data(signal_path, det_path, cate_path):
    signals = torch.tensor(scio.loadmat(signal_path)['data'], dtype=torch.float32)
    det_labels = torch.tensor(scio.loadmat(det_path)['label_box'], dtype=torch.float32)
    cls_labels = torch.tensor(scio.loadmat(cate_path)['label_cate'].astype(np.int16), dtype=torch.int64)

    val_ds = TensorDataset(signals, det_labels, cls_labels)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    return val_loader, det_labels, cls_labels