import numpy as np
import matplotlib.pyplot as plt

# ==================== SETTINGS ====================

DISPLAY_PLOTS = False
SAVE_PLOTS = False
PLOTS_DIR = "./plots/"

X_MIN, X_MAX, N_SAMPLES = -4, 4, 64
x = np.linspace(X_MIN, X_MAX, N_SAMPLES)
d = (X_MAX - X_MIN) / (N_SAMPLES - 1)

# ==================== FUNCTIONS ====================

functions = {
    "sin(x)": lambda x: np.sin(x),
    "sin(1/x)": lambda x: np.sin(1 / x),
    "sign(sin(8x))": lambda x: np.sign(np.sin(8 * x))
}

# ==================== KERNELS ====================

kernels = {
    "box(t)": lambda t: (t >= 0) & (t < 1),
    "box(t/2)": lambda t: (t >= -0.5) & (t < 0.5),
    "triangle(t)": lambda t: np.clip(1 - np.abs(t), 0, None),
    "sinc(t)": np.sinc,
    "cubic(t)": lambda t: np.where(
        np.abs(t) <= 1, 1.5 * np.abs(t)**3 - 2.5 * np.abs(t)**2 + 1,
        np.where((np.abs(t) <= 2), -0.5 * np.abs(t)**3 + 2.5 * np.abs(t)**2 - 4 * np.abs(t) + 2, 0)
    )
}

# ==================== INTERPOLATION ====================

def interpolate(Y, x, h, d):
    def interp(t):
        diff = (t[None, :] - x[:, None]) / d
        
        vals = h(diff).astype(float)
        weights = np.sum(vals, axis=0)
        return np.sum(Y[:, None] * vals, axis=0) / weights
    return interp


# ==================== MSE CRITERION ====================
def mse_criterion(f, interp, x):
    f_vals = f(x)
    interp_vals = interp(x)
    return np.mean((f_vals - interp_vals) ** 2)


# ==================== PLOTING ====================

def plot_interpolation(f, Y, h, d, x_fine, x_gen, f_name, h_name):
    interp = interpolate(Y, x, h, d)
    plt.plot(x_fine, f(x_fine), '--', color='grey', label='Original')
    plt.plot(x_fine, interp(x_fine), color='orange', label='Interpolated')
    plt.scatter(x, Y, color='black', s=10, label='Samples')
    plt.scatter(x_gen, interp(x_gen), color='red', s=10, label='Generated')
    plt.title(f'Interpolation {f_name} using {h_name} kernel\n{len(Y)}→{len(x_gen)} points MSE: {mse_criterion(f, interp, x_gen):.6f}')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    if SAVE_PLOTS:
        plt.savefig(f"{PLOTS_DIR}{f_name.replace('/', '_')}_{h_name.replace('/', '_')}_{len(x_gen)}.png")
    if DISPLAY_PLOTS:
        plt.show()
    plt.clf()

# ==================== MAIN ====================

RESULTS_FILE = "wynikizad1.txt"

def main():
    # Clear results file
    with open(RESULTS_FILE, "w") as f:
        f.write("Interpolation Results\n")
        f.write("=====================\n\n")

    x_fine = np.linspace(X_MIN, X_MAX, N_SAMPLES * 1000)
    for f_name, f in functions.items():
        Y = f(x)
        for h_name, h in kernels.items():
            for scale in [2, 4, 10]:
                x_gen = np.linspace(X_MIN, X_MAX, (N_SAMPLES) * scale)
                
                # Calculate MSE explicitly here to save it
                interp = interpolate(Y, x, h, d)
                mse = mse_criterion(f, interp, x_gen)
                
                # Save to file
                with open(RESULTS_FILE, "a") as file:
                    file.write(f"Function: {f_name}\n")
                    file.write(f"Kernel: {h_name}\n")
                    file.write(f"Scale: {scale}x ({len(Y)} -> {len(x_gen)} points)\n")
                    file.write(f"MSE: {mse:.6f}\n")
                    file.write("-" * 30 + "\n")

                plot_interpolation(f, Y, h, d, x_fine, x_gen, f_name, h_name)

if __name__ == "__main__":
    main()