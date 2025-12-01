import numpy as np
import matplotlib.pyplot as plt
from skimage import io

#import sys
#sys.path.append('/Users/marcelsiudmak/Code/py/SiOC/4.11.2025')
from zad1 import interpolate, kernels

# ==================== LOGGING ====================

RESULTS_FILE = "wynikizad2.txt"

def log(message):
    print(message)
    with open(RESULTS_FILE, "a") as f:
        f.write(str(message) + "\n")

# ==================== UPSCALER ====================

class ImageUpscaler:
    
    def __init__(self, scale_factor: float, kernel_name: str = "cubic(t)"):
       
        self.scale_factor = scale_factor
        self.kernel_name = kernel_name
        
        if kernel_name not in kernels:
            raise ValueError(f"Kernel '{kernel_name}' not found. Available: {list(kernels.keys())}")
        
        self.kernel = kernels[kernel_name]
    
    def upscale(self, image: np.ndarray) -> np.ndarray:
        # Obsługa obrazów kolorowych: rekurencyjne wywołanie dla każdego kanału
        if image.ndim == 3:
            channels = [self._upscale_single_channel(image[:, :, i]) for i in range(image.shape[2])]
            return np.stack(channels, axis=2)
        else:
            return self._upscale_single_channel(image)

    def _upscale_single_channel(self, image: np.ndarray) -> np.ndarray:
        
        # Normalizujemy obraz do [0, 1]
        is_float = np.issubdtype(image.dtype, np.floating)
        img_float = image.astype(float) / 255.0 if not is_float and image.max() > 1 else image.astype(float)
        
        height, width = img_float.shape
        d_row = 1.0
        d_col = 1.0
        
        # Oblicz docelowe wymiary (obsługa float scale_factor)
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)

        # ===== KROK 1: Przeskaluj wiersze =====
        image_rows_scaled = np.zeros((height, new_width))
        col_positions = np.arange(width)
        new_col_positions = np.linspace(0, width - 1, new_width)
        
        for i in range(height):
            interp_func = interpolate(img_float[i, :], col_positions, self.kernel, d_col)
            image_rows_scaled[i, :] = interp_func(new_col_positions)
        
        # ===== KROK 2: Przeskaluj kolumny =====
        image_upscaled = np.zeros((new_height, new_width))
        row_positions = np.arange(height)
        new_row_positions = np.linspace(0, height - 1, new_height)
        
        for j in range(new_width):
            interp_func = interpolate(image_rows_scaled[:, j], row_positions, self.kernel, d_row)
            image_upscaled[:, j] = interp_func(new_row_positions)
        
        # Konwertuj z powrotem na [0, 255] jeśli wejście nie było float
        image_upscaled = np.clip(image_upscaled, 0, 1)
        if not is_float:
             image_upscaled = (image_upscaled * 255).astype(np.uint8)
        
        return image_upscaled


class ImageDownscaler:
    
    def __init__(self, scale_factor: float, downscale_method: str = "mean"):
        """
        downscale_method: 'mean' (uśrednianie) lub 'max' (max pooling)
        """
        self.scale_factor = scale_factor
        self.downscale_method = downscale_method
    
    def downscale(self, image: np.ndarray) -> np.ndarray:
        # Dla całkowitych krotności używamy okien
        # Dla niecałkowitych (np. 1.5x) - interpolacja (używamy ImageUpscaler z odwrotnością)
        if not float(self.scale_factor).is_integer():
             # Fallback do interpolacji dla niecałkowitych scale_factor
             # Uwaga: To jest uproszczenie. Prawdziwy downscaling powinien mieć anty-aliasing.
             # Tutaj użyjemy interpolacji 'cubic' jako metody zmniejszania.
             upscaler = ImageUpscaler(scale_factor=1/self.scale_factor, kernel_name="cubic(t)")
             return upscaler.upscale(image)

        scale_int = int(self.scale_factor)
        
        if self.downscale_method in ["mean", "max"]:
            return self._pooling_downscale(image, scale_int, self.downscale_method)
        else:
            raise ValueError(f"Nieznana metoda downscalingu: {self.downscale_method}")
    
    
    def _pooling_downscale(self, image: np.ndarray, factor: int, method: str) -> np.ndarray:
        img = image.astype(float)
        h, w = img.shape[:2]
        
        out_h = h // factor
        out_w = w // factor
        
        # Przytnij obraz do wielokrotności factor
        img = img[:out_h*factor, :out_w*factor]
        
        if img.ndim == 2:
            # Reshape do (out_h, factor, out_w, factor)
            # A potem operacja na osiach 1 i 3
            reshaped = img.reshape(out_h, factor, out_w, factor)
            if method == "mean":
                out = reshaped.mean(axis=(1, 3))
            elif method == "max":
                out = reshaped.max(axis=(1, 3))
        else:
            c = img.shape[2]
            reshaped = img.reshape(out_h, factor, out_w, factor, c)
            if method == "mean":
                out = reshaped.mean(axis=(1, 3))
            elif method == "max":
                out = reshaped.max(axis=(1, 3))

        if np.issubdtype(image.dtype, np.integer):
            out = np.clip(out, 0, 255).astype(image.dtype)
        else:
            out = out.astype(image.dtype)

        return out


# ==================== METRICS ====================

def mse_image(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    
    diff = img1.astype(float) - img2.astype(float)
    return np.mean(diff ** 2)


# ==================== TESTING ====================

import os

def save_single_plot(image: np.ndarray, title: str, filepath: str):
    """
    Pomocnicza funkcja do zapisywania pojedynczego obrazu z tytułem.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close(fig)
    print(f"Zapisano: {filepath}")


def test_grayscale_pipeline(img_path: str, scale_factor: int = 2, kernels_to_test: list = None):
    log(f"\n=== TESTOWANIE SKALI SZAROŚCI (ZADANIE PODSTAWOWE) ===")
    if kernels_to_test is None:
        kernels_to_test = ["triangle(t)", "cubic(t)"]

    # Utwórz katalog na wyniki
    results_dir = "results/grayscale"
    os.makedirs(results_dir, exist_ok=True)

    # Załaduj obraz jako grayscale
    img_original = io.imread(img_path, as_gray=True)
    
    # Konwersja do uint8 [0, 255] dla spójności
    if img_original.max() <= 1.0:
        img_original = (img_original * 255).astype(np.uint8)
    else:
        img_original = img_original.astype(np.uint8)

    log(f"Oryginalny obraz (Grayscale): {img_original.shape}")
    save_single_plot(img_original, f'Oryginał\n{img_original.shape}', os.path.join(results_dir, "original.png"))

    # 1. Downscale (wspólny dla wszystkich testów upscalingu)
    downscaler = ImageDownscaler(scale_factor=scale_factor, downscale_method="mean")
    img_downscaled = downscaler.downscale(img_original)
    
    log(f"Po pomniejszeniu (Downscaled): {img_downscaled.shape}")
    save_single_plot(img_downscaled, f'Pomniejszony (Mean)\n{img_downscaled.shape}', os.path.join(results_dir, "downscaled_mean.png"))

    results = {}

    # 2. Upscale i porównanie dla każdego kernela
    for idx, k_name in enumerate(kernels_to_test):
        log(f"\n--- Testowanie interpolacji: {k_name} ---")
        upscaler = ImageUpscaler(scale_factor=scale_factor, kernel_name=k_name)
        
        # Upscaling
        img_reconstructed = upscaler.upscale(img_downscaled)
        
        # Oblicz MSE
        mse_val = mse_image(img_original, img_reconstructed)
        results[k_name] = mse_val
        log(f"MSE ({k_name}): {mse_val:.4f}")

        # Zapisz wynik
        safe_k_name = k_name.replace("(", "").replace(")", "").replace("/", "_")
        save_single_plot(img_reconstructed, f'Upscaled: {k_name}\nMSE: {mse_val:.2f}', os.path.join(results_dir, f"upscaled_{safe_k_name}.png"))

    log("\n=== WYNIKI MSE (GRAYSCALE) ===")
    for k, v in results.items():
        log(f"{k}: {v:.4f}")


def test_extensions(img_path: str):
    log(f"\n=== TESTOWANIE ROZSZERZEŃ (KOLOR, MAX POOLING, ETC.) ===")
    
    # Utwórz katalog na wyniki
    results_dir = "results/extensions"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Wczytaj obraz (kolorowy)
    img_color = io.imread(img_path) # Domyślnie kolor
    log(f"Obraz wejściowy: {img_color.shape}, typ: {img_color.dtype}")
    save_single_plot(img_color, "Oryginał (Kolor)", os.path.join(results_dir, "original_color.png"))

    # ===== TEST 1: Max Pooling vs Mean Pooling (Downscale) =====
    scale = 2
    log(f"\n--- TEST 1: Mean vs Max Pooling (Scale {scale}x) ---")
    
    # Mean Pooling
    down_mean = ImageDownscaler(scale, "mean").downscale(img_color)
    rec_mean = ImageUpscaler(scale, "cubic(t)").upscale(down_mean)
    mse_mean = mse_image(img_color, rec_mean)
    
    save_single_plot(down_mean, f"Mean Pooling\nRec. MSE: {mse_mean:.2f}", os.path.join(results_dir, "downscaled_mean.png"))
    
    # Max Pooling
    down_max = ImageDownscaler(scale, "max").downscale(img_color)
    rec_max = ImageUpscaler(scale, "cubic(t)").upscale(down_max)
    mse_max = mse_image(img_color, rec_max)
    
    save_single_plot(down_max, f"Max Pooling\nRec. MSE: {mse_max:.2f}", os.path.join(results_dir, "downscaled_max.png"))
    
    log(f"MSE (Mean Pooling + Cubic Upscale): {mse_mean:.4f}")
    log(f"MSE (Max Pooling + Cubic Upscale): {mse_max:.4f}")
    

    # ===== TEST 2: Niecałkowite skalowanie (1.5x) =====
    float_scale = 1.5
    log(f"\n--- TEST 2: Niecałkowite skalowanie ({float_scale}x) ---")
    
    # Zmniejsz o 1.5x
    down_float = ImageDownscaler(float_scale).downscale(img_color)
    log(f"Po zmniejszeniu 1.5x: {down_float.shape}")
    save_single_plot(down_float, f"Downscaled {float_scale}x", os.path.join(results_dir, "downscaled_float.png"))
    
    # Powiększ z powrotem o 1.5x
    up_float = ImageUpscaler(float_scale, "cubic(t)").upscale(down_float)
    log(f"Po powiększeniu 1.5x: {up_float.shape}")
    
    mse_float = mse_image(img_color, up_float)
    log(f"MSE (Float Scaling): {mse_float:.4f}")
    save_single_plot(up_float, f"Reconstructed {float_scale}x\nMSE: {mse_float:.2f}", os.path.join(results_dir, "reconstructed_float.png"))


    # ===== TEST 3: Skalowanie sekwencyjne vs Bezpośrednie (8x) =====
    # Porównamy: Oryginał -> Down 8x -> Up 8x (Direct)
    # vs Oryginał -> Down 2x -> Down 2x -> Down 2x -> Up 2x -> Up 2x -> Up 2x (Sequential)
    
    log(f"\n--- TEST 3: Skalowanie sekwencyjne (8x Multi-stage) ---")
    
    # 1. Direct Pipeline
    # Downscale 8x
    down_direct = ImageDownscaler(8, "mean").downscale(img_color)
    # Upscale 8x
    up_direct = ImageUpscaler(8, "cubic(t)").upscale(down_direct)
    
    mse_direct = mse_image(img_color, up_direct)
    log(f"MSE (Direct 8x): {mse_direct:.4f}")
    save_single_plot(up_direct, f"Direct 8x\nMSE: {mse_direct:.2f}", os.path.join(results_dir, "direct_8x.png"))

    # 2. Sequential Pipeline
    # Downscale sequence: 2x -> 2x -> 2x
    d1 = ImageDownscaler(2, "mean").downscale(img_color)
    d2 = ImageDownscaler(2, "mean").downscale(d1)
    d3 = ImageDownscaler(2, "mean").downscale(d2) # To jest odpowiednik 8x
    
    # Upscale sequence: 2x -> 2x -> 2x
    u1 = ImageUpscaler(2, "cubic(t)").upscale(d3)
    u2 = ImageUpscaler(2, "cubic(t)").upscale(u1)
    up_seq = ImageUpscaler(2, "cubic(t)").upscale(u2)
    
    mse_seq = mse_image(img_color, up_seq)
    log(f"MSE (Sequential 3x2x): {mse_seq:.4f}")
    save_single_plot(up_seq, f"Sequential 3 stages (2x)\nMSE: {mse_seq:.2f}", os.path.join(results_dir, "sequential_8x.png"))
    
    # Zapisz też obrazek pośredni (najmniejszy)
    save_single_plot(d3, "Smallest (Downscaled 8x)", os.path.join(results_dir, "smallest_8x.png"))


    # ===== TEST 4: Pure Upscaling Comparison (Original -> Up) =====
    # Porównamy: Oryginał -> Up 4x (Direct) vs Oryginał -> Up 2x -> Up 2x (Sequential)
    # Brak obrazu referencyjnego (ground truth), więc porównujemy metody między sobą.
    # Dodatkowo: Cycle Consistency MSE (Original vs Downscaled(Upscaled))
    
    log(f"\n--- TEST 4: Pure Upscaling Comparison (Original -> Up 4x) ---")
    
    # A: Direct 4x
    pure_direct = ImageUpscaler(4, "cubic(t)").upscale(img_color)
    log(f"Pure Direct 4x shape: {pure_direct.shape}")
    
    # Consistency Check A: Downscale back to 1x and compare with Original
    check_direct = ImageDownscaler(4, "mean").downscale(pure_direct)
    mse_cons_direct = mse_image(img_color, check_direct)
    log(f"Consistency MSE (Direct): {mse_cons_direct:.4f}")
    
    save_single_plot(pure_direct, f"Pure Direct 4x\nCons. MSE: {mse_cons_direct:.2f}", os.path.join(results_dir, "pure_direct_4x.png"))
    
    # B: Sequential 2x -> 2x
    pure_step1 = ImageUpscaler(2, "cubic(t)").upscale(img_color)
    pure_seq = ImageUpscaler(2, "cubic(t)").upscale(pure_step1)
    log(f"Pure Sequential 4x shape: {pure_seq.shape}")
    
    # Consistency Check B: Downscale back to 1x and compare with Original
    check_seq = ImageDownscaler(4, "mean").downscale(pure_seq)
    mse_cons_seq = mse_image(img_color, check_seq)
    log(f"Consistency MSE (Sequential): {mse_cons_seq:.4f}")
    
    save_single_plot(pure_seq, f"Pure Sequential 2x->2x\nCons. MSE: {mse_cons_seq:.2f}", os.path.join(results_dir, "pure_sequential_4x.png"))
    
    # Porównanie MSE między metodami
    mse_pure_diff = mse_image(pure_direct, pure_seq)
    log(f"MSE (Pure Direct vs Pure Sequential): {mse_pure_diff:.4f}")


if __name__ == "__main__":
    IMAGE_PATH = 'cat.png'
    
    # Clear results file
    with open(RESULTS_FILE, "w") as f:
        f.write("Image Upscaling Results\n")
        f.write("=======================\n")

    # 1. Zadanie podstawowe (Grayscale, MSE comparison)
    test_grayscale_pipeline(img_path=IMAGE_PATH, scale_factor=2, kernels_to_test=["triangle(t)", "cubic(t)"])
    
    # 2. Rozszerzenia (Color, Max Pooling, Float scale, Sequential)
    test_extensions(IMAGE_PATH)
