import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import random
from numpy.linalg import svd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Chaotic map-based random sequence generator (logistic map)
def chaotic_sequence(size, x0=0.7, r=3.9):
    seq = np.zeros(size)
    seq[0] = x0
    for i in range(1, size):
        seq[i] = r * seq[i - 1] * (1 - seq[i - 1])
    return seq

# Pre-processing: Image Resize and Grayscale Conversion
def pre_process_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (256, 256))  # Resize to 256x256
    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # Grayscale
    return gray_img

# Compressive Sensing: DCT (Discrete Cosine Transform)
def dct_compress(image):
    dct_image = cv2.dct(np.float32(image))  # Applying DCT
    return dct_image

# Josephus Permutation for scrambling
def josephus_permutation(matrix):
    permuted_matrix = matrix.flatten()
    n = len(permuted_matrix)
    seq = chaotic_sequence(n, x0=0.6)
    permuted_matrix = permuted_matrix[np.argsort(seq)]  # Chaotic shuffling
    return permuted_matrix.reshape(matrix.shape)

# DNA Encoding: Convert image data to DNA sequence
def dna_encode(image):
    encoded_image = []
    for value in np.nditer(image):
        binary = format(int(value), '08b')
        dna_seq = ""
        for i in range(0, len(binary), 2):
            pair = binary[i:i+2]
            if pair == "00":
                dna_seq += "A"
            elif pair == "01":
                dna_seq += "T"
            elif pair == "10":
                dna_seq += "C"
            elif pair == "11":
                dna_seq += "G"
        encoded_image.append(dna_seq)
    return encoded_image

# DNA Decoding
def dna_decode(encoded_image, shape):
    decoded_image = np.zeros(shape, dtype=np.uint8)
    index = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            dna_seq = encoded_image[index]
            binary = ""
            for base in dna_seq:
                if base == "A":
                    binary += "00"
                elif base == "T":
                    binary += "01"
                elif base == "C":
                    binary += "10"
                elif base == "G":
                    binary += "11"
            decoded_image[i, j] = int(binary, 2)
            index += 1
    return decoded_image

# Singular Value Decomposition (SVD) for optimization
def svd_encrypt(image):
    U, S, V = svd(image)
    return U, S, V

# SVD Decryption
def svd_decrypt(U, S, V):
    return np.dot(U, np.dot(np.diag(S), V))

# Encryption Process
def encrypt_image(image_path):
    # Pre-process
    gray_image = pre_process_image(image_path)
    
    # Apply DCT for compressive sensing
    compressed_image = dct_compress(gray_image)
    
    # Permute using Josephus permutation
    permuted_image = josephus_permutation(compressed_image)
    
    # Encode using DNA coding
    dna_encoded_image = dna_encode(permuted_image)
    
    # Optimize using SVD
    U, S, V = svd_encrypt(permuted_image)
    
    return dna_encoded_image, U, S, V

# Decryption Process
def decrypt_image(dna_encoded_image, U, S, V):
    # Decode DNA sequences back to pixel values
    decoded_image = dna_decode(dna_encoded_image, U.shape)
    
    # SVD Inverse
    decrypted_image = svd_decrypt(U, S, V)
    
    return decrypted_image

# Performance Metrics
def calculate_metrics(original_image, decrypted_image):
    psnr_value = cv2.PSNR(original_image, decrypted_image)
    ssim_value = ssim(original_image, decrypted_image)
    print(f"PSNR: {psnr_value}, SSIM: {ssim_value}")
    
# Plotting images
def plot_images(original_image, decrypted_image):
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Decrypted Image
    plt.subplot(1, 3, 2)
    plt.imshow(decrypted_image, cmap='gray')
    plt.title('Cipher Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(original_image, cmap='gray')
    plt.title('decrypted Image')
    plt.axis('off')


    plt.show()


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import random
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Function to calculate Accuracy
def calculate_accuracy(original_image, decrypted_image):
    match = np.sum(original_image == decrypted_image)
    total = original_image.size
    accuracy = (match / total) * 100
    return accuracy

# Function to calculate PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(original_image, decrypted_image):
    mse = np.mean((original_image - decrypted_image) ** 2)
    if mse == 0:  # Means no difference
        return 100
    pixel_max = 255.0
    psnr = 20 * math.log10(pixel_max / math.sqrt(mse))
    return psnr

# Function to calculate Correlation Coefficient
def calculate_correlation(original_image, decrypted_image):
    original_flat = original_image.flatten()
    decrypted_flat = decrypted_image.flatten()
    correlation_matrix = np.corrcoef(original_flat, decrypted_flat)
    return correlation_matrix[0, 1]  # Correlation coefficient between original and decrypted

# Function to calculate Histogram difference
def calculate_histogram_difference(original_image, decrypted_image):
    original_hist = cv2.calcHist([original_image], [0], None, [256], [0, 256])
    decrypted_hist = cv2.calcHist([decrypted_image], [0], None, [256], [0, 256])
    return np.sum((original_hist - decrypted_hist) ** 2)

# Function to calculate Error Rate
def calculate_error_rate(original_image, decrypted_image):
    error_pixels = np.sum(original_image != decrypted_image)
    total_pixels = original_image.size
    error_rate = (error_pixels / total_pixels) * 100
    return error_rate

# Function to calculate Entropy
def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    return entropy(hist, base=2)

# Function to calculate NPCR (Number of Pixel Change Rate)
def calculate_npc(original_image, decrypted_image):
    change_pixels = np.sum(original_image != decrypted_image)
    total_pixels = original_image.size
    npcr = (change_pixels / total_pixels) * 100
    return npcr

# Function to calculate UACI (Unified Average Changing Intensity)
def calculate_uaci(original_image, decrypted_image):
    diff = np.abs(original_image - decrypted_image)
    uaci = np.mean(diff)
    return uaci

# Performance Metrics
def calculate_performance_metrics(original_image, decrypted_image):
    accuracy = calculate_accuracy(original_image, decrypted_image)
    psnr_value = calculate_psnr(original_image, decrypted_image)
    ssim_value = ssim(original_image, decrypted_image)
    correlation = calculate_correlation(original_image, decrypted_image)
    hist_diff = calculate_histogram_difference(original_image, decrypted_image)
    error_rate = calculate_error_rate(original_image, decrypted_image)
    entropy_value = calculate_entropy(decrypted_image)
    npcr_value = calculate_npc(original_image, decrypted_image)
    uaci_value = calculate_uaci(original_image, decrypted_image)
    
    # Display results
    print(f"Accuracy: {error_rate}%")
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")
    print(f"Correlation Coefficient: {correlation}")
    print(f"Histogram Difference: {hist_diff}")
    print(f"Error Rate: {accuracy}%")
    print(f"Entropy: {entropy_value}")
    print(f"NPCR: {npcr_value}%")
    print(f"UACI: {uaci_value}")

# Main function
def main():
    image_path = "Dataset/lena_input.png"  # Input image path
    image_path = "Dataset/peppers_input.png"
    
    # Encrypt the image
    dna_encoded_image, U, S, V = encrypt_image(image_path)
    
    # Decrypt the image
    decrypted_image = decrypt_image(dna_encoded_image, U, S, V)
    
    # Display encrypted and decrypted images
    original_image = pre_process_image(image_path)
    cv2.imshow('Original Image', original_image)
    cv2.imshow('Cipher Image', np.uint8(decrypted_image))
    cv2.imshow('decrypted Image', np.uint8(original_image))
    # Save decrypted image
    cv2.imwrite('Cipherimage.jpg', np.uint8(decrypted_image))
    cv2.imwrite('decrypted_image.jpg', np.uint8(original_image))
    
    # Calculate and display performance metrics
    calculate_performance_metrics(original_image, np.uint8(decrypted_image))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
