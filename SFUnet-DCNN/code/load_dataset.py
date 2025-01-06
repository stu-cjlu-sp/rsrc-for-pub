import utils_paths
import numpy as np
from PIL import Image
from tensorflow import keras
import tensorflow as tf

def input_data_train_Unet():
    # load data
    print("------Begin load data------")
    data = []
    data_noise = []
     
    imagePaths1 = sorted(list(utils_paths.list_images(
        '/home/sp432cy/sp432cy/Dataset/train_dataset')))

    imagePaths2 = sorted(list(utils_paths.list_images(
        '/home/sp432cy/sp432cy/Dataset/train_noise_dataset')))    
    
    for imagePath in imagePaths1:
        
        image = Image.open(imagePath)

        image = np.array(image)

        data.append(image)

    for imagePath in imagePaths2:

        image= Image.open(imagePath)

        image = np.array(image)

        data_noise.append(image)

    data = np.array(data, dtype="float32")/255
    data = data [:, :, :, np.newaxis]

    data_noise = np.array(data_noise, dtype="float32")/255
    data_noise = data_noise [:, :, :, np.newaxis]

    return data, data_noise

def input_data_validation_Unet():
    # load data
    print("------Begin load data------")
    data = []
    data_noise = []
     
    imagePaths1 = sorted(list(utils_paths.list_images(
        '/home/sp432cy/sp432cy/Dataset/validation_dataset')))

    imagePaths2 = sorted(list(utils_paths.list_images(
        '/home/sp432cy/sp432cy/Dataset/validation_noise_dataset')))    
    
    for imagePath in imagePaths1:
        
        image = Image.open(imagePath)

        image = np.array(image)

        data.append(image)

    # c = 0
    for imagePath in imagePaths2:

        image= Image.open(imagePath)

        image = np.array(image)

        data_noise.append(image)

    data = np.array(data, dtype="float32")/255
    data = data [:, :, :, np.newaxis]

    data_noise = np.array(data_noise, dtype="float32")/255
    data_noise = data_noise [:, :, :, np.newaxis]

    return data, data_noise


def input_data_train_DCNN():
    # load data
    print("------Begin load data------")
    data = []
    label = []
    dict=['BPSK','Costas','Frank','LFM','P1','P2','P3','P4','T1','T2','T3','T4']
    for snr in range(-14,12,2):

        i = -1

        for _ in range(12):
            
            imagePaths1 = sorted(list(utils_paths.list_images(
                f'/home/sp432cy/sp432cy/Dataset/train_noise_dataset/{snr}db/{dict[_]}')))
            
            i = i+1
    
            for imagePath in imagePaths1:
                
                image = Image.open(imagePath)

                image = np.array(image)

                data.append(image)
                label.append(i)

    data = np.array(data, dtype="float32")/255
    data = data [:, :, :, np.newaxis]
    label = (np.array(label, dtype="float32"))

    return data, label


def input_data_validation_DCNN():
    # load data
    print("------Begin load data------")
    data = []
    label = []
    dict=['BPSK','Costas','Frank','LFM','P1','P2','P3','P4','T1','T2','T3','T4']
    i = -1

    for _ in range(12):

        imagePaths1 = sorted(list(utils_paths.list_images(
            f'/home/sp432cy/sp432cy/Dataset/validation_noise_dataset/-10db/{dict[_]}')))
        
        i = i+1
    
        for imagePath in imagePaths1:
            
            image = Image.open(imagePath)

            # image = image.resize((256,256))

            image = np.array(image)

            data.append(image)
            label.append(i)

    data = np.array(data, dtype="float32") /255
    data = data [:, :, :, np.newaxis]
    label = (np.array(label, dtype="float32"))

    return data, label


def input_data_test_DCNN(snr=None):
    # load data
    print("------Begin load data------")
    data = []
    label = []
    dict=['BPSK','Costas','Frank','LFM','P1','P2','P3','P4','T1','T2','T3','T4']

    i = -1

    for _ in range(12):
        
        imagePaths1 = sorted(list(utils_paths.list_images(
                f'/home/sp432cy/sp432cy/Dataset/test_noise_dataset/{snr}db/{dict[_]}')))
        
        i = i+1

        for imagePath in imagePaths1:
            
            image = Image.open(imagePath)

            image = np.array(image)

            data.append(image)
            label.append(i)

    data = np.array(data, dtype="float32") /255
    data = data [:, :, :, np.newaxis]
    label = (np.array(label, dtype="float32"))

    return data, label


def input_data_test_Parameters(snr=None):
    # load data
    print("------Begin load data------")
    data = []
    label = []
    dict=['BPSK','Costas','Frank','LFM','P1','P2','P3','P4','T1','T2','T3','T4']

    i = -1

    for _ in range(12):
        
        imagePaths1 = sorted(list(utils_paths.list_images(
                f'/home/sp432cy/sp432cy/Dataset/test_dataset_different_parameters/{snr}db/{dict[_]}')))
        
        i = i+1

        for imagePath in imagePaths1:
            
            image = Image.open(imagePath)

            data.append(image)
            label.append(i)

    data = np.array(data, dtype="float32") / 255
    data = data [:, :, :, np.newaxis]
    label = (np.array(label, dtype="float32"))
    # label = keras.utils.to_categorical(label, 12)

    return data, label


def input_data_test_Semi_Physical(snr=None):
    # load data
    print("------Begin load data------")
    data = []
    label = []
    dict=['BPSK','Costas','Frank','LFM','P1','P2','P3','P4','T1','T2','T3','T4']

    i = -1

    for _ in range(12):
        
        imagePaths1 = sorted(list(utils_paths.list_images(
                f'/home/sp432cy/sp432cy/Dataset/test_dataset_semi_physical/{snr}dB/{dict[_]}')))

        i = i+1

        for imagePath in imagePaths1:
            
            image = Image.open(imagePath)

            data.append(image)
            label.append(i)

    data = np.array(data, dtype="float32") / 255
    data = data [:, :, :, np.newaxis]
    label = (np.array(label, dtype="float32"))
    # label = keras.utils.to_categorical(label, 12)

    return data, label

