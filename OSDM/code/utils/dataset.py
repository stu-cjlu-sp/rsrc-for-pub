import scipy.io as scio
import numpy as np
import os

def split_dataset(data_path,cate_path,det_path, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    mat_data = scio.loadmat(data_path)
    mat_label_cate = scio.loadmat(cate_path)
    mat_label_det = scio.loadmat(det_path)

    
    signals = mat_data['data']
    cate_labels = mat_label_cate['label_cate']
    det_labels = mat_label_det['label_box']
    
    num_samples = signals.shape[0]
    indices = np.random.permutation(num_samples)
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_signals = signals[train_indices]
    train_det = det_labels[train_indices]
    train_cate = cate_labels[train_indices]
    
    val_signals = signals[val_indices]
    val_det = det_labels[val_indices]
    val_cate = cate_labels[val_indices]
    
    test_signals = signals[test_indices]
    test_det = det_labels[test_indices]
    test_cate = cate_labels[test_indices]
    

    scio.savemat(os.path.join(output_dir, 'train', 'signal.mat'), {'data': train_signals})
    scio.savemat(os.path.join(output_dir, 'train', 'label_box.mat'), {'label_box': train_det})
    scio.savemat(os.path.join(output_dir, 'train', 'label_cate.mat'), {'label_cate': train_cate})
    
    scio.savemat(os.path.join(output_dir, 'val', 'signal.mat'), {'data': val_signals})
    scio.savemat(os.path.join(output_dir, 'val', 'label_box.mat'), {'label_box': val_det})
    scio.savemat(os.path.join(output_dir, 'val', 'label_cate.mat'), {'label_cate': val_cate})
    
    scio.savemat(os.path.join(output_dir, 'test', 'signal.mat'), {'data': test_signals})
    scio.savemat(os.path.join(output_dir, 'test', 'label_box.mat'), {'label_box': test_det})
    scio.savemat(os.path.join(output_dir, 'test', 'label_cate.mat'), {'label_cate': test_cate})


if __name__ == '__main__':
    
    data_path ='./'
    cate_path ='./'
    det_path ='./'

    output_directory = './'
    split_dataset(data_path,cate_path,det_path, output_directory)