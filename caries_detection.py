import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.feature import graycomatrix, graycoprops

def process_and_evaluate(image_path):
    # ---------------------------------------------------------
    # 1. IMAGE ACQUISITION & PRE-PROCESSING
    # ---------------------------------------------------------
    # Load the image in grayscale (8-bit)
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # ---------------------------------------------------------
    # 2. GLOBAL HISTOGRAM EQUALIZATION (GHE)
    # ---------------------------------------------------------
    # OpenCV has a highly optimized built-in function for the CDF mapping we discussed
    equalized_img = cv2.equalizeHist(original_img)

    # Calculate histograms for plotting
    hist_orig = cv2.calcHist([original_img], [0], None, [256], [0, 256])
    hist_eq = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])

    # ---------------------------------------------------------
    # 3. QUANTITATIVE EVALUATION (SSIM & PSNR)
    # ---------------------------------------------------------
    # SSIM measures structural integrity (1.0 means identical structure)
    ssim_value = ssim(original_img, equalized_img)
    
    # PSNR measures noise amplification (higher is usually better, measured in dB)
    psnr_value = psnr(original_img, equalized_img)
    
    print("--- Evaluation Metrics ---")
    print(f"Structural Similarity Index (SSIM): {ssim_value:.4f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.2f} dB\n")

    # ---------------------------------------------------------
    # 4. FEATURE EXTRACTION (GLCM) FOR RISK PREDICTION
    # ---------------------------------------------------------
    print("--- Texture Feature Extraction (GLCM) ---")
    # We extract features from the ENHANCED image to show how GHE helps
    # distances=[1] means looking at adjacent pixels, angles=[0] is horizontal
    glcm = graycomatrix(equalized_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    # Extract specific mathematical textures related to demineralization (caries)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    print(f"Contrast (Roughness): {contrast:.4f}")
    print(f"Homogeneity: {homogeneity:.4f}")
    print(f"Energy: {energy:.4f}")
    print(f"Correlation: {correlation:.4f}\n")
    
    # Note: In a full pipeline, these 4 values would be passed into a 
    # model like LogisticRegression.predict([[contrast, homogeneity, energy, correlation]])

    # ---------------------------------------------------------
    # 5. VISUALIZATION (For your final report)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8))

    # Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Low-Contrast X-Ray')
    plt.axis('off')

    # Original Histogram
    plt.subplot(2, 2, 2)
    plt.plot(hist_orig, color='black')
    plt.title('Original Histogram (Clustered)')
    plt.xlim([0, 256])

    # Equalized Image
    plt.subplot(2, 2, 3)
    plt.imshow(equalized_img, cmap='gray')
    plt.title('Enhanced X-Ray (GHE)')
    plt.axis('off')

    # Equalized Histogram
    plt.subplot(2, 2, 4)
    plt.plot(hist_eq, color='black')
    plt.title('Equalized Histogram (Stretched)')
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

# ==========================================
# Run the pipeline
# ==========================================
if __name__ == "__main__":
    # Replace this with the path to a test dental x-ray on your computer
    test_image_path = "sample_dental_xray.jpg" 
    process_and_evaluate(test_image_path)