import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt


def extract_blocks(img, block_size=4):
	orig_rows, orig_cols = img.shape
	# Compute new dimensions (padded) that are multiples of block_size
	new_rows = ((orig_rows + block_size - 1) // block_size) * block_size
	new_cols = ((orig_cols + block_size - 1) // block_size) * block_size

	# Create a padded image and copy the original image into it
	padded_img = np.zeros((new_rows, new_cols), dtype=img.dtype)
	padded_img[:orig_rows, :orig_cols] = img

	# Extract blocks from the padded image
	blocks = []
	for i in range(0, new_rows, block_size):
		for j in range(0, new_cols, block_size):
			block = padded_img[i:i + block_size, j:j + block_size]
			blocks.append(block.flatten())
	return np.array(blocks), (new_rows, new_cols), (orig_rows, orig_cols)


def reconstruct_image(blocks, padded_shape, orig_shape, block_size=4):
	new_rows, new_cols = padded_shape
	reconstructed = np.zeros((new_rows, new_cols), dtype=np.uint8)
	idx = 0
	for i in range(0, new_rows, block_size):
		for j in range(0, new_cols, block_size):
			reconstructed[i:i + block_size, j:j + block_size] = blocks[idx].reshape((block_size, block_size))
			idx += 1
	# Crop the image back to the original dimensions
	orig_rows, orig_cols = orig_shape
	return reconstructed[:orig_rows, :orig_cols]


def compute_mse(original, reconstructed):
	return np.mean((original.astype("float") - reconstructed.astype("float"))**2)


def compute_psnr(mse, max_val=255.0):
	return 10 * np.log10((max_val**2) / mse) if mse > 0 else float('inf')


def run_lloyd(data, centroids, tol=1e-5, max_iter=100):
	"""
	Performs the standard Lloyd iterations (like k-means) given initial centroids.
	"""
	for iteration in range(max_iter):
		# Assignment: find the nearest centroid for each data point.
		distances = np.linalg.norm(data[:, None] - centroids, axis=2)
		nearest = np.argmin(distances, axis=1)

		# Update: compute new centroids as the mean of assigned data points.
		new_centroids = []
		for j in range(centroids.shape[0]):
			assigned = data[nearest == j]
			if len(assigned) > 0:
				new_centroids.append(np.mean(assigned, axis=0))
			else:
				new_centroids.append(centroids[j])
		new_centroids = np.array(new_centroids)

		# Check for convergence.
		if np.linalg.norm(new_centroids - centroids) < tol:
			break
		centroids = new_centroids
	return centroids


def lbg_algorithm(data, num_centroids, tol=1e-5, max_iter=100):
	"""
	LBG algorithm: start with one centroid (the overall mean), then iteratively split
	centroids until reaching the desired number.
	"""
	# Start with the mean of the data.
	centroid = np.mean(data, axis=0)
	centroids = np.array([centroid])
	epsilon = np.std(data) * 0.01  # small perturbation factor

	while centroids.shape[0] < num_centroids:
		# Split each centroid.
		new_centroids = []
		for c in centroids:
			new_centroids.append(c + epsilon)
			new_centroids.append(c - epsilon)
		centroids = np.array(new_centroids)

		# Run Lloyd iterations on the current centroids.
		centroids = run_lloyd(data, centroids, tol, max_iter)

		# If we have too many centroids, randomly select the desired number.
		if centroids.shape[0] > num_centroids:
			indices = np.random.choice(centroids.shape[0], num_centroids, replace=False)
			centroids = centroids[indices]
	return centroids


def compute_distances(data, centroids):
	# ||x - y||² = ||x||² + ||y||² - 2x·y
	data_sq = np.sum(data**2, axis=1)[:, None]
	centroids_sq = np.sum(centroids**2, axis=1)
	dot_product = np.dot(data, centroids.T)
	return data_sq - 2 * dot_product + centroids_sq


def quantize_blocks(blocks, codebook):
	distances = compute_distances(blocks, codebook)
	indices = np.argmin(distances, axis=1)
	return codebook[indices], indices


def process_single_image(image_path, codebook_sizes=[128, 256], block_size=4):
	img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	if img is None:
		raise ValueError("Image not found or cannot be loaded.")

	blocks = extract_blocks(img, block_size)

	print(f"Processing {image_path} with {blocks.shape[0]} blocks.")

	for L in codebook_sizes:
		print(f"\nTraining codebook with L = {L}...")
		codebook = lbg_algorithm(blocks, L)

		quantized_blocks, _ = quantize_blocks(blocks, codebook)
		reconstructed = reconstruct_image(quantized_blocks, img.shape, block_size)

		mse = compute_mse(img, reconstructed)
		psnr = compute_psnr(mse)
		print(f"Codebook Size: {L}, MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")

		plt.figure(figsize=(6, 6))
		plt.title(f"Quantized Image (L = {L})")
		plt.imshow(reconstructed, cmap='gray')
		plt.axis("off")
		plt.show()


def process_multi_image_training(original_image_path, training_image_dir, codebook_sizes=[128, 256], block_size=4):
	# Load the original image
	orig_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
	if orig_img is None:
		raise ValueError("Original image not found.")

	# Extract blocks from original image (with padding)
	orig_blocks = extract_blocks(orig_img, block_size)

	# Load training images
	training_paths = glob.glob(os.path.join(training_image_dir, "*"))
	training_paths = training_paths[:10]  # Limit to first 10 images
	training_blocks = []
	for path in training_paths:
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		if img is None:
			print(f"Skipping invalid image: {path}")
			continue
		blocks = extract_blocks(img, block_size)
		if blocks.shape[0] == 0:
			print(f"Skipping image with no blocks: {path}")
			continue
		training_blocks.append(blocks)

	if not training_blocks:
		raise ValueError("No valid training blocks found.")

	training_blocks = np.vstack(training_blocks)
	print(f"Training on {len(training_paths)} images with {training_blocks.shape[0]} blocks.")

	for L in codebook_sizes:
		print(f"\n[Multi-Image Training] Training codebook with L = {L}...")
		codebook_multi = lbg_algorithm(training_blocks, L)

		quantized_blocks, _ = quantize_blocks(orig_blocks, codebook_multi)
		reconstructed = reconstruct_image(quantized_blocks, orig_img.shape, block_size)

		mse = compute_mse(orig_img, reconstructed)
		psnr = compute_psnr(mse)
		print(f"[Multi-Image] Codebook Size: {L}, MSE: {mse:.2f}, PSNR: {psnr:.2f} dB")

		plt.figure(figsize=(6, 6))
		plt.title(f"Quantized (Multi-Image Codebook, L = {L})")
		plt.imshow(reconstructed, cmap='gray')
		plt.axis("off")
		plt.show()


if __name__ == "__main__":
	# For single-image training:
	single_image_path = "sample_image/sample_image/airplane.png"
	# process_single_image(single_image_path, codebook_sizes=[128, 256], block_size=4)

	# For multi-image training:
	training_folder = "sample_image/sample_image"
	process_multi_image_training(single_image_path, training_folder, codebook_sizes=[128, 256], block_size=4)
