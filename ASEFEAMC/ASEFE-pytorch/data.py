import os
import pickle

import h5py
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas as pd
import torch
from scipy.signal import stft, get_window
from PIL import Image
import numpy as np
from torchvision import transforms

def process_signals_to_spectrograms_STFT(inputs, fs=128, frame_length=40, overlap_ratio=0.95, output_size=(128, 128)):
    """
    Converts a batch of complex signals into normalized spectrogram images.

    Args:
        inputs (np.ndarray): Input signals of shape [batch_size, 2, 128] where
                             inputs[:, 0, :] is the I channel and inputs[:, 1, :] is the Q channel.
        fs (int): Sampling frequency.
        frame_length (int): Frame length for STFT.
        overlap_ratio (float): Overlap ratio for STFT.
        output_size (tuple): Desired output size of the spectrogram images.

    Returns:
        np.ndarray: Array of spectrogram images of shape [batch_size, output_size[0], output_size[1], 3].
    """
    batch_size = inputs.shape[0]
    num_channels = inputs.shape[1]  # Should be 2 (I and Q)

    # Create an array to hold the output spectrogram images
    spectrogram_images = np.zeros((batch_size, output_size[0], output_size[1], 3), dtype=np.float32)

    # STFT parameters
    overlap = int(overlap_ratio * frame_length)  # Calculate overlap samples
    window = get_window('hamming', frame_length)  # Hamming window

    for i in range(batch_size):
        # Construct the complex signal for the current sample
        I = inputs[i, 0, :]
        Q = inputs[i, 1, :]
        signal = I + 1j * Q

        # Compute STFT
        f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=frame_length, noverlap=overlap)

        # Convert to magnitude spectrogram
        magnitude_spectrogram = np.abs(Zxx)

        # Normalize the spectrogram
        magnitude_spectrogram = (magnitude_spectrogram - np.min(magnitude_spectrogram)) / (
                np.max(magnitude_spectrogram) - np.min(magnitude_spectrogram))

        # Resize to the desired output size
        spectrogram_image = Image.fromarray((magnitude_spectrogram).astype(np.float32)).resize(output_size)

        # Create a 3-channel image
        spectrogram_image = np.stack([spectrogram_image] * 3, axis=-1)

        # Store the spectrogram image in the output array
        spectrogram_images[i] = spectrogram_image

    return torch.tensor(spectrogram_images)

# Define the transformation for loading grayscale images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts to [0, 1] range
])


class STFTDataset(Dataset):
    def __init__(self, snr=0):
        # 加载数据
        self.save_folder = 'snr_data/output_data_snr_' + str(snr)
        self.img_files = os.listdir(self.save_folder)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.save_folder, self.img_files[index])
        img = transform(Image.open(img_path))
        label = int(img_path[img_path.find("label_") + len("label_"):img_path.find(".png")])
        return img, label


def stft_imgs(input_vectors, labels, save_dir, fs=128, frame_length=40, overlap_ratio=0.95):
    """
        Converts a batch of complex signals into normalized grayscale spectrogram images and saves them.

        Args:
            inputs (np.ndarray): Input signals of shape [batch_size, 2, 128].
            save_dir (str): Directory to save the spectrogram images.
            fs (int): Sampling frequency.
            frame_length (int): Frame length for STFT.
            overlap_ratio (float): Overlap ratio for STFT.
        """

    overlap = int(overlap_ratio * frame_length)
    window = get_window('hamming', frame_length)

    for i in range(len(input_vectors)):
        I = input_vectors[i, 0, :]
        Q = input_vectors[i, 1, :]
        signal = I + 1j * Q

        # Compute STFT
        f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=frame_length, noverlap=overlap)

        # Convert to magnitude spectrogram
        magnitude_spectrogram = np.abs(Zxx)

        # Normalize the spectrogram to range [0, 1]
        magnitude_spectrogram = (magnitude_spectrogram - np.min(magnitude_spectrogram)) / (
                    np.max(magnitude_spectrogram) - np.min(magnitude_spectrogram))

        # Convert to grayscale image (0-1 range)
        spectrogram_image = Image.fromarray((magnitude_spectrogram * 255).astype(np.uint8), mode='L')

        # Save the spectrogram as a grayscale image
        label = labels[i]
        spectrogram_image.save(os.path.join(save_dir, f'spectrogram_{i}_label_{label}.png'))


def load_signals_radio2016a(file_name):
    # 加载数据
    # file_name = 'snr_data/output_data_snr_0.csv'
    csv_file_path = file_name
    data_frame = pd.read_csv(csv_file_path)

    # 提取前256列数据并转换为张量
    vectors = torch.tensor(data_frame.iloc[:, :256].values, dtype=torch.float32)

    # 将256维向量转换为2x128的矩阵形式
    vectors = vectors.view(-1, 2, 128)

    mean_I = vectors[:, 0, :].mean(dim=0, keepdim=True)
    std_I = vectors[:, 0, :].std(dim=0, keepdim=True)

    mean_Q = vectors[:, 1, :].mean(dim=0, keepdim=True)
    std_Q = vectors[:, 1, :].std(dim=0, keepdim=True)

    # 归一化整个数据集
    vectors[:, 0, :] = (vectors[:, 0, :] - mean_I) / std_I
    vectors[:, 1, :] = (vectors[:, 1, :] - mean_Q) / std_Q

    # 提取Mod_Type列并转换为数值标签
    mod_types = data_frame['Mod_Type'].astype('category').cat.codes.values
    labels = torch.tensor(mod_types, dtype=torch.long)
    return vectors, labels

class Radioml_18_Dataset(Dataset):
    def __init__(self, dataset_path):
        super(Radioml_18_Dataset, self).__init__()

        h5_file = h5py.File(dataset_path, 'r')
        self.data = h5_file['X']
        self.mod = np.argmax(h5_file['Y'], axis=1)  # comes in one-hot encoding
        self.snr = h5_file['Z'][:, 0]
        self.len = self.data.shape[0]

        self.mod_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                            '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
                            'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
        self.snr_classes = np.arange(-20., 32., 2)  # -20dB to 30dB

        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
        train_indices = []
        val_indices = []
        test_indices = []
        for mod in range(0, 24):  # all modulations (0 to 23)
            for snr_idx in range(0, 26):  # all SNRs (0 to 25 = -20dB to +30dB)
                # 'X' holds frames strictly ordered by modulation and SNR
                start_idx = 26 * 4096 * mod + 4096 * snr_idx
                indices_subclass = list(range(start_idx, start_idx + 4096))

                # 60%/20%/20% training/validation/test split, applied evenly for each mod-SNR pair
                train_ratio = 0.6
                val_ratio = 0.2
                test_ratio = 0.2

                split_train = int(train_ratio * 4096)  # 2457
                split_validation = int(val_ratio * 4096)  # 819
                split_test = 4096 - split_train - split_validation  # 820

                np.random.shuffle(indices_subclass)
                train_indices_subclass = indices_subclass[:split_train]
                val_incices_subclass = indices_subclass[split_train:(split_train + split_validation)]
                test_indices_subclass = indices_subclass[-split_test:]

                # you could train on a subset of the data, e.g. based on the SNR
                # here we use all available training samples
                if snr_idx >= 0:
                    train_indices.extend(train_indices_subclass)
                val_indices.extend(val_incices_subclass)
                test_indices.extend(test_indices_subclass)

        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

class Radioml_16_Dataset(Dataset):
    def __init__(self, dataset_path):
        super(Radioml_16_Dataset, self).__init__()
        with open(dataset_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()

            snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], p.keys())))), [1, 0])

            X = []
            lbl = []
            snr_label = []

            for mod in mods:
                for snr in snrs:
                    samples = p[(mod, snr)]
                    X.append(samples)
                    for _ in range(samples.shape[0]):
                        lbl.append((mod, snr))
                        snr_label.append(snr)

            self.X = np.vstack(X)
            self.lbl = np.vstack(lbl)
            self.snr_label = np.vstack(snr_label)

            n_examples = int(self.X.shape[0])
            self.labels = list(map(lambda x: mods.index(lbl[x][0]), range(0, n_examples)))

            # 准备training, validation, test 的数据
            np.random.seed(2016)
            n_examples = int(self.X.shape[0])
            train_ratio = 0.6
            val_ratio = 0.2
            test_ratio = 0.2
            n_train = int(n_examples * train_ratio)
            n_val = int(n_examples * val_ratio)
            n_test = n_examples - n_train - n_val

            idx = np.random.permutation(n_examples)
            train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

            def to_tensor_data(indices):
                return (torch.tensor(self.X[indices], dtype=torch.float32),
                        torch.tensor([mods.index(self.lbl[i][0]) for i in indices], dtype=torch.long),
                        torch.from_numpy(np.array(self.snr_label[indices])).to(torch.int))

            X_train, Y_train, _ = to_tensor_data(train_idx)
            X_val, Y_val, _ = to_tensor_data(val_idx)
            X_test, Y_test, SNR_test = to_tensor_data(test_idx)

            self.train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
            self.val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=64, shuffle=False)
            self.test_loader = DataLoader(TensorDataset(X_test, Y_test, SNR_test), batch_size=64, shuffle=False)

            print("Dataset 2016a, {} samples for training, {} for validation, {} for testing".format(
                len(Y_train),
                len(Y_val),
                len(Y_test)))

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def prepare_stft_imgs(snr=18):
    print("### Preparing SNR_[{}] data ###".format(snr))
    file_name = 'snr_data/output_data_snr_{}.csv'.format(snr)
    save_dir = file_name.replace(".csv","")
    vectors, labels = load_signals_radio2016a(file_name)
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
    stft_imgs(input_vectors=vectors, labels=labels, save_dir=save_dir)

def prepare_stft_imgs_of_SNRs(SNR_list):
    for snr in SNR_list:
        prepare_stft_imgs(snr)



if __name__ == '__main__':
    # prepare_stft_imgs()
    # SNR_list = [n for n in range(-20, 20, 2)]
    # print(SNR_list)
    # prepare_stft_imgs_of_SNRs(SNR_list)

    Radioml_16_Dataset("/home/song/Datasets/RML2016.10a_dict.pkl")
