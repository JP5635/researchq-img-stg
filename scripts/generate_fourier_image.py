"""
Generate an illustrative image for the Fourier transform:
- Left: time-domain signal (sum of sinusoids + noise)
- Right: magnitude of FFT (frequency-domain)
Saves to ../assets/fourier_transform.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def make_signal(fs=1000, T=1.0):
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    # Components: 50 Hz and 120 Hz
    sig = 1.0 * np.sin(2 * np.pi * 50 * t) + 0.6 * np.sin(2 * np.pi * 120 * t)
    # Add a small transient and some noise
    sig += 0.3 * np.sin(2 * np.pi * 300 * t) * (np.exp(-5 * t))
    sig += 0.2 * np.random.normal(scale=1.0, size=t.shape)
    return t, sig


def compute_fft(sig, fs):
    N = len(sig)
    fft_vals = np.fft.rfft(sig)
    fft_mag = np.abs(fft_vals) / N
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    return freqs, fft_mag


def plot_and_save(path):
    fs = 2000
    T = 1.0
    t, sig = make_signal(fs=fs, T=T)
    freqs, mag = compute_fft(sig, fs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time domain
    axes[0].plot(t, sig, color="#1f77b4")
    axes[0].set_title("Time domain signal")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlim(0, 0.1)  # zoom first 100ms for clarity

    # Frequency domain
    # `use_line_collection` may not be supported in all matplotlib versions; omit it for compatibility
    axes[1].stem(freqs, mag, basefmt=" ")
    axes[1].set_title("Magnitude spectrum (FFT)")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlim(0, 500)

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved image to: {path}")


if __name__ == '__main__':
    import os
    root = os.path.dirname(os.path.dirname(__file__))
    out = os.path.join(root, 'assets', 'fourier_transform.png')
    plot_and_save(out)
