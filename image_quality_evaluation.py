import cv2
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt

# ==============================================================================
# 0. Setup and Image Loading
# ==============================================================================

def load_images():
    print("--- 0. Setup: Loading Images ---")

    '''
    TODO: Load the original image 'bonn.jpg' and noisy image 'bonn_noisy.jpg'
    Convert both to grayscale and prepare the noisy image in float format (0-1 range)
    Calculate and print the PSNR of the noisy image compared to the original
    '''

    # Load images here
    original_img_color = cv2.imread("bonn.jpg", cv2.IMREAD_COLOR)  # Load color image
    original_img_gray = cv2.cvtColor(original_img_color, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    noisy_img = cv2.imread("bonn_noisy.jpg", cv2.IMREAD_GRAYSCALE)  # Load noisy image in grayscale
    noisy_img_float_01 = noisy_img.astype(np.float32) / 255.0  # Convert noisy image to float range 0-1

    # Calculate PSNR of noisy image
    psnr_noisy = psnr(original_img_gray, noisy_img, data_range=255)

    # Display original and noisy images
    # TODO: Create a figure showing original and noisy images side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    images =[original_img_gray,noisy_img]
    titles = ['Original Image', 'Noisy Image']
    for ax, img, title in zip(axes.flatten(), images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return original_img_gray, noisy_img, noisy_img_float_01


# ==============================================================================
# Custom Filter Definitions (for parts a, b, c)
# ==============================================================================

def custom_gaussian_filter(image, kernel_size, sigma):
    """
    Custom Gaussian Filter - Implement convolution from scratch
    
    Args:
        image: Input image (float, 0-1 range)
        kernel_size: Size of the Gaussian kernel (odd integer)
        sigma: Standard deviation of the Gaussian
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO: 
    1. Create Gaussian kernel using the formula: G(x,y) = exp(-(x^2 + y^2)/(2*sigma^2))
    2. Normalize the kernel so it sums to 1
    3. Pad the image using reflect mode
    4. Apply convolution manually using nested loops
    """
    
    # 1D axis values, defines spatial window for the kernel
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1) 
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # 2. Normalize the kernel so it sums to 1
    kernel = kernel / np.sum(kernel)
    
    # 3. Pad the image using 'reflect' mode
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='reflect')
    
    # 4. Manual convolution using nested loops
    filtered = np.zeros_like(image, dtype=np.float32) #The output image is initialized with zeros
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.sum(region * kernel)
    return filtered



def custom_median_filter(image, kernel_size):
    """
    Custom Median Filter - Implement median calculation from scratch
    
    Args:
        image: Input image (float, 0-1 range)
        kernel_size: Size of the median filter window (odd integer)
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO:
    1. Pad the image using reflect mode
    2. For each pixel, extract the neighborhood window
    3. Calculate the median of the window
    4. Assign the median value to the output pixel
    """
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract neighborhood window
            window = padded_img[i:i+kernel_size, j:j+kernel_size]
            # Calculate median and assign
            filtered[i, j] = np.median(window)
            
    return filtered


def custom_bilateral_filter(image, diameter, sigma_color, sigma_space):
    """
    Custom Bilateral Filter
    
    Args:
        image: Input image (float, 0-1 range)
        d: Diameter of the pixel neighborhood
        sigma_color: Filter sigma in the color space (0-1 range for float images)
        sigma_space: Filter sigma in the coordinate space
    
    Returns:
        Filtered image (float, 0-1 range)
    
    TODO:
    1. Pad the image
    2. For each pixel:
       a. Calculate spatial weights based on distance from center
       b. Calculate range weights based on intensity difference
       c. Combine weights and compute weighted average
    3. Normalize by sum of weights
    """
    pad = diameter // 2
    padded_img = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image, dtype=np.float32)
    ax = np.arange(-pad, pad + 1)
    xx, yy = np.meshgrid(ax, ax)
    spatial_gauss = np.exp(-(xx**2 + yy**2) / (2 * sigma_space ** 2))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i + diameter, j:j + diameter]
            intensity_diff = region - image[i, j]
            range_gauss = np.exp(-(intensity_diff ** 2) / (2 * sigma_color ** 2))
            weights = spatial_gauss * range_gauss
            weighted_sum = np.sum(weights * region)
            filtered[i, j] = weighted_sum / (np.sum(weights) + 1e-8)

    return filtered


def main():
    # 0. Setup: Load images and print PSNR of noisy
    original_img_gray, noisy_img, noisy_img_float_01 = load_images()

    # ==============================================================================
    # 1. Filter Application (Parts a, b, c)
    # ==============================================================================
    print("\n--- 1. Filter Application (Parts a, b, c) ---")

    # Default Parameters
    K_DEFAULT = 7
    S_DEFAULT = 2.0
    D_DEFAULT = 9
    SC_DEFAULT = 100  # cv2 range (0-255)
    SS_DEFAULT = 75

    # -------------------------- a) Gaussian Filter --------------------------
    print("a) Applying Gaussian Filter...")
    '''
    TODO: 
    1. Apply Gaussian filter using cv2.GaussianBlur()
    2. Apply your custom Gaussian filter
    3. Calculate PSNR for both results
    4. Display the results in a figure with 3 subplots (noisy, cv2 result, custom result)
    '''

    denoised_gaussian_cv2 = cv2.GaussianBlur(noisy_img, (K_DEFAULT, K_DEFAULT), S_DEFAULT)
    psnr_gaussian_cv2 = psnr(original_img_gray, denoised_gaussian_cv2, data_range=255)

    denoised_gaussian_custom_float = custom_gaussian_filter(noisy_img_float_01, K_DEFAULT, S_DEFAULT)
    denoised_gaussian_custom = (denoised_gaussian_custom_float * 255).astype(np.uint8)
    psnr_gaussian_custom = psnr(original_img_gray, denoised_gaussian_custom, data_range=255)

    # Display results here
    print("PSNR Gaussian OpenCV:", psnr_gaussian_cv2)
    print("PSNR Gaussian Custom:", psnr_gaussian_custom)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(noisy_img, cmap='gray')
    axes[0].set_title('Noisy Image')
    axes[1].imshow(denoised_gaussian_cv2, cmap='gray')
    axes[1].set_title(f'Gaussian OpenCV\nPSNR: {psnr_gaussian_cv2:.2f}')
    axes[2].imshow(denoised_gaussian_custom, cmap='gray')
    axes[2].set_title(f'Gaussian Custom\nPSNR: {psnr_gaussian_custom:.2f}')
    for ax in axes:
        ax.axis('off')
    plt.show()

    # -------------------------- b) Median Filter --------------------------
    print("b) Applying Median Filter...")
    '''
    TODO:
    1. Apply Median filter using cv2.medianBlur()
    2. Apply your custom Median filter
    3. Calculate PSNR for both results
    4. Display the results in a figure with 3 subplots
    '''

    denoised_median_cv2 = cv2.medianBlur(noisy_img, K_DEFAULT)
    psnr_median_cv2 = psnr(original_img_gray, denoised_median_cv2, data_range=255)

    denoised_median_custom_float = custom_median_filter(noisy_img_float_01, K_DEFAULT)
    denoised_median_custom = (denoised_median_custom_float * 255).astype(np.uint8)
    psnr_median_custom =  psnr(original_img_gray, denoised_median_custom, data_range=255)

    # Display results here
    print("PSNR Median OpenCV:", psnr_median_cv2)
    print("PSNR Median Custom:", psnr_median_custom)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(noisy_img, cmap='gray')
    axes[0].set_title('Noisy Image')
    axes[1].imshow(denoised_median_cv2, cmap='gray')
    axes[1].set_title(f'Median OpenCV\nPSNR: {psnr_median_cv2:.2f}')
    axes[2].imshow(denoised_median_custom, cmap='gray')
    axes[2].set_title(f'Median Custom\nPSNR: {psnr_median_custom:.2f}')
    for ax in axes:
        ax.axis('off')
    plt.show()

    # -------------------------- c) Bilateral Filter --------------------------
    print("c) Applying Bilateral Filter...")
    '''
    TODO:
    1. Apply Bilateral filter using cv2.bilateralFilter()
    2. Apply your custom Bilateral filter (remember to scale sigma_color for 0-1 range)
    3. Calculate PSNR for both results
    4. Display the results in a figure with 3 subplots
    '''

    denoised_bilateral_cv2 = cv2.bilateralFilter(noisy_img, D_DEFAULT, SC_DEFAULT, SS_DEFAULT)
    psnr_bilateral_cv2 = psnr(original_img_gray, denoised_bilateral_cv2, data_range=255)

    denoised_bilateral_custom_float = custom_bilateral_filter(noisy_img_float_01, D_DEFAULT, SC_DEFAULT/255.0, SS_DEFAULT)
    denoised_bilateral_custom = (denoised_bilateral_custom_float * 255).astype(np.uint8)

    psnr_bilateral_custom = psnr(original_img_gray, denoised_bilateral_custom, data_range=255)

    # Display results here
    print("PSNR Bilateral OpenCV:", psnr_bilateral_cv2)
    print("PSNR Bilateral Custom:", psnr_bilateral_custom)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(noisy_img, cmap='gray')
    axes[0].set_title('Noisy Image')
    axes[1].imshow(denoised_bilateral_cv2, cmap='gray')
    axes[1].set_title(f'Bilateral OpenCV\nPSNR: {psnr_bilateral_cv2:.2f}')
    axes[2].imshow(denoised_bilateral_custom, cmap='gray')
    axes[2].set_title(f'Bilateral Custom\nPSNR: {psnr_bilateral_custom:.2f}')
    for ax in axes:
        ax.axis('off')
    plt.show()


    # ==============================================================================
    # 2. Performance Comparison (Part d)
    # ==============================================================================
    print("\n--- d) Performance Comparison ---")
    '''
    TODO:
    1. Compare PSNR values of all three filters
    2. Determine which filter performs best
    3. Display side-by-side comparison of all filtered images
    4. Print the results with the best performing filter highlighted
    '''

    psnr_values = {
        'Gaussian OpenCV': psnr_gaussian_cv2,
        'Gaussian Custom': psnr_gaussian_custom,
        'Median OpenCV': psnr_median_cv2,
        'Median Custom': psnr_median_custom,
        'Bilateral OpenCV': psnr_bilateral_cv2,
        'Bilateral Custom': psnr_bilateral_custom
    }

    # Find best filter by PSNR
    best_filter_name = max(psnr_values, key=psnr_values.get)
    best_psnr_value = psnr_values[best_filter_name]

    # Display side-by-side comparison of all filtered images
    fig, axes = plt.subplots(2, 4, figsize=(16,8))
    imgs_to_show = [
        (noisy_img, 'Noisy Image'),
        (denoised_gaussian_cv2, f'Gaussian OpenCV\nPSNR: {psnr_gaussian_cv2:.2f}'),
        (denoised_gaussian_custom, f'Gaussian Custom\nPSNR: {psnr_gaussian_custom:.2f}'),
        (denoised_median_cv2, f'Median OpenCV\nPSNR: {psnr_median_cv2:.2f}'),
        (denoised_median_custom, f'Median Custom\nPSNR: {psnr_median_custom:.2f}'),
        (denoised_bilateral_cv2, f'Bilateral OpenCV\nPSNR: {psnr_bilateral_cv2:.2f}'),
        (denoised_bilateral_custom, f'Bilateral Custom\nPSNR: {psnr_bilateral_custom:.2f}')
    ]

    for ax, (img, title) in zip(axes.flatten(), imgs_to_show):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


    # ==============================================================================
    # 3. Parameter Optimization (Part e)
    # ==============================================================================

    def run_optimization(original_img, noisy_img):
        """
        Optimize parameters for all three filters to maximize PSNR
        
        Args:
            original_img: Original clean image
            noisy_img: Noisy image to be filtered
        
        Returns:
            Dictionary containing optimal parameters and best PSNR for each filter
        
        TODO:
        1. For Gaussian filter: iterate over kernel_sizes and sigma values
        2. For Median filter: iterate over kernel_sizes
        3. For Bilateral filter: iterate over d, sigma_color, and sigma_space values
        4. Track the best PSNR and corresponding parameters for each filter
        5. Return results as a dictionary
        
        """

        best_results = {
            'gaussian': {'psnr': 0, 'params': None},
            'median': {'psnr': 0, 'params': None},
            'bilateral': {'psnr': 0, 'params': None}
        }
        
        # 1. Gaussian filter params
        for kernel_size in [3, 5, 7, 9, 11]:
            for sigma in [0.5, 1.0, 1.5, 2.0, 2.5]:
                denoised = cv2.GaussianBlur(noisy_img, (kernel_size, kernel_size), sigma)
                cur_psnr = psnr(original_img, denoised, data_range=255)
                if cur_psnr > best_results['gaussian']['psnr']:
                    best_results['gaussian'] = {'psnr': cur_psnr, 'params': (kernel_size, sigma)}
                    
        # 2. Median filter params
        for kernel_size in [3, 5, 7, 9, 11]:
            denoised = cv2.medianBlur(noisy_img, kernel_size)
            cur_psnr = psnr(original_img, denoised, data_range=255)
            if cur_psnr > best_results['median']['psnr']:
                best_results['median'] = {'psnr': cur_psnr, 'params': kernel_size}
        
        # 3. Bilateral filter params
        for diameter in [5, 7, 9, 11]:
            for sigma_color in [25, 50, 75, 100]:
                for sigma_space in [25, 50, 75, 100]:
                    denoised = cv2.bilateralFilter(noisy_img, diameter, sigma_color, sigma_space)
                    cur_psnr = psnr(original_img, denoised, data_range=255)
                    if cur_psnr > best_results['bilateral']['psnr']:
                        best_results['bilateral'] = {'psnr': cur_psnr, 'params': (diameter, sigma_color, sigma_space)}

        return best_results
        


    '''
    TODO:
    1. Call run_optimization() function
    2. Extract optimal parameters for each filter
    3. Apply filters using optimal parameters
    4. Display the optimized results in a 2x2 grid (noisy + 3 optimal filters)
    5. Print the optimal parameters clearly
    '''

    # 1. Call run_optimization() function
    optimal_results = run_optimization(original_img_gray, noisy_img)

    # 2. Extract optimal parameters for each filter
    gaussian_kernel, gaussian_sigma = optimal_results['gaussian']['params']
    median_kernel = optimal_results['median']['params']
    bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space = optimal_results['bilateral']['params']

    # 3. Apply filters using the optimal parameters
    denoised_gaussian_opt = cv2.GaussianBlur(noisy_img, (gaussian_kernel, gaussian_kernel), gaussian_sigma)

    denoised_median_opt = cv2.medianBlur(noisy_img, median_kernel)

    denoised_bilateral_opt = cv2.bilateralFilter(noisy_img, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)

    # 4. Display the optimized results in a 2x2 grid (noisy + 3 filtered images)
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.flatten()

    axes[0].imshow(noisy_img, cmap='gray')
    axes[0].set_title('Noisy Image')
    axes[1].imshow(denoised_gaussian_opt, cmap='gray')
    axes[1].set_title(f'Gaussian Filter\nKernel={gaussian_kernel}, Sigma={gaussian_sigma}')
    axes[2].imshow(denoised_median_opt, cmap='gray')
    axes[2].set_title(f'Median Filter\nKernel={median_kernel}')
    axes[3].imshow(denoised_bilateral_opt, cmap='gray')
    axes[3].set_title(f'Bilateral Filter\nDiameter={bilateral_diameter}, SigmaColor={bilateral_sigma_color}, SigmaSpace={bilateral_sigma_space}')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # 5. Print the optimal parameters clearly
    print("Optimal Parameters Found:")
    print(f"Gaussian Filter - Kernel size: {gaussian_kernel}, Sigma: {gaussian_sigma}")
    print(f"Median Filter - Kernel size: {median_kernel}")
    print(f"Bilateral Filter - Diameter: {bilateral_diameter}, SigmaColor: {bilateral_sigma_color}, SigmaSpace: {bilateral_sigma_space}")

    print("\n--- Discussion ---")
    print("Based on your PSNR results, the bilateral filter achieved the highest score, indicating it handled " \
    "both Gaussian and salt-and-pepper noise most effectively. The Gaussian filter performed nearly as well," \
    " mainly reducing Gaussian noise but slightly blurring edges. The median filter, while specifically suited"
    " for salt-and-pepper noise, had the lowest PSNR here, likely because the mixture of noise types limited " \
    "its effectiveness. Overall, your experimental results align with theory: bilateral filtering is best for " \
    "mixed noise, median excels for impulse noise, and Gaussian is optimal for Gaussian noise")


if __name__ == "__main__":
    main()

