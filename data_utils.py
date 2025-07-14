import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

def get_err(matlab_data: np.ndarray, my_data: np.ndarray):
    err = np.mean(np.abs(matlab_data - my_data))/np.max(np.abs(my_data-matlab_data))
    return err

def interpolate_mat(x_mat, y_mat, x_eval):
    y_interp = interp1d(x_mat, y_mat)
    y_eval = y_interp(x_eval)
    return y_eval

def get_fft(time: np.ndarray, signal: np.ndarray, title: str, zero_pad_factor: int = 4):
    dt = time[1] - time[0]
    N = len(signal)

    # Optional: subtract mean to reduce DC bias before windowing
    signal = signal - np.mean(signal)

    # Apply Hann window
    window = hann(N, sym=False)
    signal_windowed = signal * window

    # Zero padding
    N_padded = zero_pad_factor * N
    signal_padded = np.pad(signal_windowed, (0, N_padded - N))

    print(f'Original length = {N}, Zero-padded length = {N_padded}')
    print(f'Frequency resolution = {2 * np.pi / (N_padded * dt):.4f} rad/s')

    # FFT
    fft_vals = np.fft.rfft(signal_padded)
    fft_freqs = np.fft.rfftfreq(N_padded, d=dt)

    # Amplitude normalization (window affects amplitude, so also scale accordingly)
    amplitudes = np.abs(fft_vals) / N  # Normalize to original N to keep comparable magnitude

    # Plot
    title_nospaces = title.replace(" ", "_")
    plt.figure()
    plt.scatter(2 * np.pi * fft_freqs, amplitudes, s=10)
    plt.xlabel(r"$\omega$ (rad/s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim((0, 10))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"figs/{title_nospaces}.png", dpi=300)
    plt.show()
