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

def get_fft(time: np.ndarray, signal: np.ndarray):
    dt = time[1]-time[0]
    signal = signal * hann(time.shape[0], sym=False)

    # i should shift to get a 0 mean value otherwise windowing mess up

    print(f'Frequency resolution = {1/time[-1]} Hz')

    fft_vals = np.fft.rfft(signal)
    fft_freqs = np.fft.rfftfreq(len(signal), d=dt)

    amplitudes = np.abs(fft_vals)/len(signal)

    plt.figure()
    plt.scatter(fft_freqs[1:], amplitudes[1:])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.xlim((0,30))
    plt.show()
