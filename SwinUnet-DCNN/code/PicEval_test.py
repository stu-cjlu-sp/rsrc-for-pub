import tensorflow as tf
import utils_paths
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image


def read_image(file_path):
    image_data = tf.io.read_file(file_path)
    return tf.image.decode_png(image_data, channels=1)  

def calculate_reconstructed_metrics(original_paths, reconstructed_folder):
    total_psnr_reconstructed = 0.0
    total_ssim_reconstructed = 0.0
    num_images = 0

    for i, OriginalTFI_path in enumerate(original_paths, start=1):
        OriginalTFI_img = read_image(OriginalTFI_path)

        ReconstructedTFI_path = os.path.join(reconstructed_folder, f'reconstructed_image{i}.jpg')
        ReconstructedTFI_img = read_image(ReconstructedTFI_path)

        psnr_reconstructed = tf.image.psnr(ReconstructedTFI_img, OriginalTFI_img, max_val=255.)
        ssim_reconstructed = tf.image.ssim(ReconstructedTFI_img, OriginalTFI_img, max_val=255.)

        total_psnr_reconstructed += psnr_reconstructed
        total_ssim_reconstructed += ssim_reconstructed
        num_images += 1

    avg_psnr_reconstructed = total_psnr_reconstructed / num_images
    avg_ssim_reconstructed = total_ssim_reconstructed / num_images

    print("Average PSNR for Reconstructed Images:", avg_psnr_reconstructed.numpy())
    print("Average SSIM for Reconstructed Images:", avg_ssim_reconstructed.numpy())

    return avg_psnr_reconstructed, avg_ssim_reconstructed

def calculate_noisy_metrics(original_paths, noisy_paths):
    total_psnr_noisy = 0.0
    total_ssim_noisy = 0.0
    num_images = 0

    for OriginalTFI_path, NoisyTFI_path in zip(original_paths, noisy_paths):
        OriginalTFI_img = read_image(OriginalTFI_path)
        NoisyTFI_img = read_image(NoisyTFI_path)

        psnr_noisy = tf.image.psnr(NoisyTFI_img, OriginalTFI_img, max_val=255.)
        ssim_noisy = tf.image.ssim(NoisyTFI_img, OriginalTFI_img, max_val=255.)

        total_psnr_noisy += psnr_noisy
        total_ssim_noisy += ssim_noisy
        num_images += 1

    avg_psnr_noisy = total_psnr_noisy / num_images
    avg_ssim_noisy = total_ssim_noisy / num_images

    print("Average PSNR for Noisy Images:", avg_psnr_noisy.numpy())
    print("Average SSIM for Noisy Images:", avg_ssim_noisy.numpy())

    return avg_psnr_noisy, avg_ssim_noisy

# Example usage:

for snr in range(-16, 12, 2):

    original_paths = sorted(list(utils_paths.list_images(f'/home/sp432cy/sp432cy/Dataset/test_dataset/{snr}db')))
    noisy_paths = sorted(list(utils_paths.list_images(f'/home/sp432cy/sp432cy/Dataset/test_noise_dataset/{snr}db')))
    reconstructed_folder =f'/home/sp432cy/sp432cy/Test_Result/SNR_{snr}db/'
    avg_psnr_reconstructed, avg_ssim_reconstructed = calculate_reconstructed_metrics(original_paths, reconstructed_folder)
    avg_psnr_noisy, avg_ssim_noisy = calculate_noisy_metrics(original_paths, noisy_paths)




