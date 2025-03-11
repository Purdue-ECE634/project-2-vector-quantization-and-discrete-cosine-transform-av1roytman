import numpy as np
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt


def dct2(block):
	"""Compute the 2D DCT of an 8x8 block"""
	return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
	"""Compute the 2D inverse DCT of an 8x8 block"""
	return idct(idct(block.T, norm='ortho').T, norm='ortho')


def zigzag_indices(n=8):
	"""Generate a matrix of zigzag indices for an n×n block"""
	indices = np.empty((n, n), dtype=int)
	index = 0
	for s in range(2 * n - 1):
		if s % 2 == 0:
			# Even sum index: iterate i from 0 to s
			for i in range(s + 1):
				j = s - i
				if i < n and j < n:
					indices[i, j] = index
					index += 1
		else:
			# Odd sum index: iterate j from 0 to s
			for j in range(s + 1):
				i = s - j
				if i < n and j < n:
					indices[i, j] = index
					index += 1
	return indices


def mask_coefficients(block, K, zigzag_order):
	"""Keep only the K coefficients (lowest indices in zigzag order)"""
	mask = (zigzag_order < K).astype(np.float32)
	return block * mask


def performDCT(image_path):
	# Load a grayscale image (make sure the image dimensions are multiples of 8)
	image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise ValueError("Image not found. Please check the path.")

	# Crop image if necessary so its dimensions are multiples of 8
	height, width = image.shape
	height = height - (height % 8)
	width = width - (width % 8)
	image = image[:height, :width]

	# Precompute the zigzag order for an 8x8 block
	zigzag_order = zigzag_indices(n=8)

	# Dictionary to store reconstructions for different K values
	reconstructed_images = {}

	# List of K values to test
	K_values = [2, 4, 8, 16, 32]

	# Process each K value
	for K in K_values:
		recon = np.zeros_like(image, dtype=np.float32)
		# Process the image block by block
		for i in range(0, height, 8):
			for j in range(0, width, 8):
				block = image[i:i + 8, j:j + 8].astype(np.float32)
				# Compute DCT of the block
				block_dct = dct2(block)
				# Keep only K coefficients (using the zigzag order)
				block_dct_masked = mask_coefficients(block_dct, K, zigzag_order)
				# Reconstruct the block using inverse DCT
				block_recon = idct2(block_dct_masked)
				recon[i:i + 8, j:j + 8] = block_recon

		# Clip values and convert back to uint8
		recon = np.clip(recon, 0, 255).astype(np.uint8)
		reconstructed_images[K] = recon
		# Compute PSNR for a quantitative measure
		psnr_val = cv2.PSNR(image, recon)
		print(f"K = {K}: PSNR = {psnr_val:.2f} dB")

	# Plotting all images in one figure using matplotlib
	plt.figure(figsize=(12, 8))
	for idx, K in enumerate(K_values):
		plt.subplot(2, 3, idx + 1)
		plt.imshow(reconstructed_images[K], cmap='gray')
		plt.title(f"K = {K}")
		plt.axis('off')
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	performDCT("./sample_image/sample_image/airplane.png")
	performDCT("./sample_image/sample_image/sails.png")
