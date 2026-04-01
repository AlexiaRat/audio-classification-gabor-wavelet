import numpy as np
import matplotlib.pyplot as plt
import scipy.fft


# ──────────────────────────────────────────────────────────────
# Single Mexican Hat wavelet filter modulated by cos/sin carriers
# ──────────────────────────────────────────────────────────────
def mexican_hat_filter(size, sigma, freq):
    n = np.arange(size)
    mu = size / 2.0
    t = (n - mu) / sigma
    mexican_hat = (1 - t**2) * np.exp(-t**2 / 2.0)
    mexican_hat = mexican_hat / np.max(np.abs(mexican_hat))
    offset = mexican_hat[0]
    mexican_hat = mexican_hat - offset
    cos_wave = np.cos(2 * np.pi * freq * n)
    sin_wave = np.sin(2 * np.pi * freq * n)
    cos_h = mexican_hat * cos_wave
    sin_h = mexican_hat * sin_wave
    return cos_h, sin_h


def create_mexican_hat_filterbank(M, size, fs):
    def hz_to_mel(f):
        return 1127 * np.log(1 + f / 700.0)

    def mel_to_hz(fmel):
        return 700 * (np.exp(fmel / 1127.0) - 1)

    A = hz_to_mel(0)
    B = hz_to_mel(fs / 2.0)
    mel_points = np.linspace(A, B, M + 2)
    hz_points = mel_to_hz(mel_points)
    centers = hz_points[1:-1]
    lengths = np.diff(hz_points)
    segment_lengths = lengths[:-1]
    freqs = centers / fs
    sigmas = fs / segment_lengths

    filters_cos = []
    filters_sin = []

    for i in range(M):
        cos_h, sin_h = mexican_hat_filter(size, sigmas[i], freqs[i])
        filters_cos.append(cos_h)
        filters_sin.append(sin_h)

    filters_cos = np.array(filters_cos)
    filters_sin = np.array(filters_sin)

    return filters_cos, filters_sin, freqs, sigmas



if __name__ == "__main__":
    size = 1102
    freq_test = 0.00267
    sigma_test = 187.21221

    cos_h, sin_h = mexican_hat_filter(size, sigma_test, freq_test)
    
    plt.figure(figsize=(12, 5))
    plt.plot(cos_h, linewidth=1.5, color='green')
    plt.title('Mexican Hat Filter - cos', fontsize=14, fontweight='bold')
    plt.xlabel('Time (samples)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('RatAlexia-Maria_343C2_mexican_hat_cos.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(sin_h, linewidth=1.5, color='red')
    plt.title('Mexican Hat Filter - sin', fontsize=14, fontweight='bold')
    plt.xlabel('Time (samples)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('RatAlexia-Maria_343C2_mexican_hat_sin.png', dpi=300, bbox_inches='tight')
    plt.close()

    M = 12
    fs = 44100

    filters_cos, filters_sin, freqs, sigmas = create_mexican_hat_filterbank(M, size, fs)

    spectra_cos = []
    for i in range(M):
        spectrum_cos = scipy.fft.fft(filters_cos[i])
        magnitude = np.abs(spectrum_cos)
        magnitude = magnitude - magnitude[0]
        spectra_cos.append(magnitude)
    spectra_cos = np.array(spectra_cos)

    half_size = size // 2
    freq_axis = np.linspace(0, fs/2, half_size)

    plt.figure(figsize=(14, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, M))
    for i in range(M):
        plt.plot(freq_axis, spectra_cos[i, :half_size],
                label=f'Filter {i+1}', linewidth=2, color=colors[i])
    plt.title('Mexican Hat Filters Spectrum (Magnitude)', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.legend(loc='upper right', ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('RatAlexia-Maria_343C2_spectru_mexican.png', dpi=300, bbox_inches='tight')
    plt.close()

    
