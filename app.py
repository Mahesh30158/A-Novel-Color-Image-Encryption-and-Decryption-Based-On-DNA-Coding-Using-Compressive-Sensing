
import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
from numpy.linalg import svd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from PIL import Image
import base64
from gtts import gTTS
import subprocess
# Function to get base64 encoding of a file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set the background of the Streamlit app
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpeg;base64,%s");
    background-position: center;
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown('<style>h1 { color: Black ; }</style>', unsafe_allow_html=True)
    st.markdown('<style>p { color: Black; }</style>', unsafe_allow_html=True)
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background/1.png')

# Streamlit app title
# st.title("Project : Detection of Brain Diseases using Machine Learning and Medical Imaging")

# Helper Functions
def chaotic_sequence(size, x0=0.7, r=3.9):
    seq = np.zeros(size)
    seq[0] = x0
    for i in range(1, size):
        seq[i] = r * seq[i - 1] * (1 - seq[i - 1])
    return seq

def pre_process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray_image, (256, 256))

def dct_compress(image):
    return cv2.dct(np.float32(image))

def josephus_permutation(matrix):
    permuted_matrix = matrix.flatten()
    seq = chaotic_sequence(len(permuted_matrix), x0=0.6)
    permuted_matrix = permuted_matrix[np.argsort(seq)]
    return permuted_matrix.reshape(matrix.shape)

def dna_encode(image):
    encoded_image = []
    for value in np.nditer(image):
        binary = format(int(value), '08b')
        dna_seq = "".join("ATCG"[(int(binary[i:i+2], 2))] for i in range(0, len(binary), 2))
        encoded_image.append(dna_seq)
    return encoded_image

def dna_decode(encoded_image, shape):
    decoded_image = np.zeros(shape, dtype=np.uint8)
    index = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            dna_seq = encoded_image[index]
            binary = "".join(format("ATCG".index(base), '02b') for base in dna_seq)
            decoded_image[i, j] = int(binary, 2)
            index += 1
    return decoded_image

def svd_encrypt(image):
    U, S, V = svd(image)
    return U, S, V

def svd_decrypt(U, S, V):
    return np.dot(U, np.dot(np.diag(S), V))

def calculate_psnr(original_image, decrypted_image):
    mse = np.mean((original_image - decrypted_image) ** 2)
    return 20 * math.log10(255.0 / math.sqrt(mse)) if mse != 0 else 100

def calculate_accuracy(original_image, decrypted_image):
    return np.sum(original_image == decrypted_image) / original_image.size * 100

def calculate_correlation(original_image, decrypted_image):
    return np.corrcoef(original_image.flatten(), decrypted_image.flatten())[0, 1]

def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    return entropy(hist, base=2)

def calculate_performance_metrics(original_image, decrypted_image):
    accuracy = calculate_accuracy(original_image, decrypted_image)
    psnr_value = calculate_psnr(original_image, decrypted_image)
    ssim_value = ssim(original_image, decrypted_image)
    correlation = calculate_correlation(original_image, decrypted_image)
    entropy_value = calculate_entropy(decrypted_image)
    uaci_value = calculate_uaci(original_image, decrypted_image)
    
    st.write(f"Accuracy: {accuracy*1000}%")
    st.write(f"PSNR: {psnr_value}")
    st.write(f"SSIM: {ssim_value}")
    st.write(f"Correlation Coefficient: {correlation}")
    st.write(f"Entropy: {entropy_value}")
    st.write(f"UACI: {uaci_value}")

# Streamlit Interface
def main():
    st.title("A Novel Of Color Image Encryption Algorithm Based On Compressive Sensing And Block-Based Dna Coding")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image:
        image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)

        st.image(image, caption="Original Image", use_column_width=True)

        # Process Image (Encrypt and Decrypt)
        gray_image = pre_process_image(image)
        compressed_image = dct_compress(gray_image)
        permuted_image = josephus_permutation(compressed_image)
        dna_encoded_image = dna_encode(permuted_image)
        U, S, V = svd_encrypt(permuted_image)
        
        # Decrypt Image
        decrypted_image = svd_decrypt(U, S, V)
        decrypted_image = dna_decode(dna_encoded_image, U.shape)
        decrypted_image = np.uint8(decrypted_image)

        st.image(decrypted_image, caption="Decrypted Image", use_column_width=True)

        # Calculate and display performance metrics
        calculate_performance_metrics(gray_image, decrypted_image)

        # Save Decrypted Image
        if st.button("Save Decrypted Image"):
            save_path = "decrypted_image.png"
            cv2.imwrite(save_path, decrypted_image)
            st.success(f"Decrypted image saved as {save_path}")

if __name__ == "__main__":
    main()
