# A Novel Color Image Encryption And Decryption Block Based DNA Coding Using Compressive Sensing.
8th sem Major project on the topic "Color Image Encryption Using Compressive Sensing and DNA Coding"using python languge spyder,anaconda prompt.

Objective:
The project aims to develop a secure method for encrypting color images without losing image quality. It integrates multiple advanced techniques such as compressive sensing, DNA coding, DCT, Josephus permutation, and chaotic systems. The goal is to protect images against unauthorized access and tampering. This method is suitable for secure communication and digital storage. It ensures high encryption strength and efficient reconstruction.

Techniques Used:
The encryption uses Discrete Cosine Transform (DCT) for compressive sensing, reducing redundancy in image data. A chaotic map is used to introduce randomness and unpredictability. Josephus permutation further scrambles the image, making it harder to decrypt. DNA coding maps pixel values to DNA sequences for a biological-inspired encryption method. During decryption, SVD (Singular Value Decomposition) helps in high-quality image reconstruction.

Encryption Process:
First, the image undergoes preprocessing steps like resizing, denoising, and optional grayscale conversion. Then DCT is applied to compress the image and prepare it for encryption. A chaotic map generates random sequences to shuffle the pixels, followed by Josephus permutation for deeper scrambling. Next, block-by-block DNA coding is applied to encode pixel values. For decryption, SVD reconstructs the image and enhancement algorithms restore original quality.

1.Original Image:
<img src="https://github.com/Mahesh30158/A-Novel-Color-Image-Encryption-and-Decryption-Based-On-DNA-Coding-Using-Compressive-Sensing/blob/main/baboon_input.png"/>

2.Encryption Image:
<img src="https://github.com/Mahesh30158/A-Novel-Color-Image-Encryption-and-Decryption-Based-On-DNA-Coding-Using-Compressive-Sensing/blob/main/decrypted_image.png"/>

3.Decryption Image:
<img src="https://github.com/Mahesh30158/A-Novel-Color-Image-Encryption-and-Decryption-Based-On-DNA-Coding-Using-Compressive-Sensing/blob/main/baboon_input.png"/>

Performance Metrics:
Several evaluation methods are used to test encryption strength and image quality. SSIM and PSNR measure the similarity between the original and decrypted image. Entropy and histogram analysis check the randomness and pixel distribution. NPCR and UAC are used to assess sensitivity to small changes in the image. A low error rate confirms minimal loss in decrypted image quality.

Advantages:
1.The proposed method provides strong protection through layered encryption mechanisms. 
2.It combines biological concepts (DNA coding) with mathematical models (DCT, SVD) and chaos theory.
3.The system is robust against statistical and brute-force attacks. 
4.Image quality after decryption remains high, ensuring usability. 
5.This method is highly applicable in secure image transmission and digital forensics.

Keywords:
SSIM, PSNR, Entropy, Histogram, NPCR, UAC, DNA Coding, Chaotic System, DCT, SVD, Josephus Permutation
Images of working model:

