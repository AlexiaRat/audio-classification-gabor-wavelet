import numpy as np
import matplotlib.pyplot as plt
import scipy.fft


# ──────────────────────────────────────────────────────────────
# Single Gabor filter: Gaussian envelope modulated by cos/sin
# Returns both cosine and sine components for magnitude computation
# ──────────────────────────────────────────────────────────────
def gabor_filter(size, sigma, freq):
    n = np.arange(size)
    mu = size / 2.0
    gaussian = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((n - mu) ** 2) / (2 * sigma ** 2))
    cos_wave = np.cos(2 * np.pi * freq * n)
    sin_wave = np.sin(2 * np.pi * freq * n)
    cos_h = gaussian * cos_wave
    sin_h = gaussian * sin_wave
    return cos_h, sin_h


# ──────────────────────────────────────────────────────────────
# Create bank of M Gabor filters with center frequencies
# distributed on the Mel scale between 0 Hz and fs/2.
# Uses Hz↔Mel conversion for perceptually uniform spacing.
# ──────────────────────────────────────────────────────────────
def create_gabor_filterbank(M, size, fs):
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
        cos_h, sin_h = gabor_filter(size, sigmas[i], freqs[i])
        filters_cos.append(cos_h)
        filters_sin.append(sin_h)

    filters_cos = np.array(filters_cos)
    filters_sin = np.array(filters_sin)

    return filters_cos, filters_sin, freqs, sigmas




if __name__ == "__main__":
    size = 1102
    freq_test = 0.00267
    sigma_test = 187.21221

    cos_h, sin_h = gabor_filter(size, sigma_test, freq_test)
    
    plt.figure(figsize=(12, 5))
    plt.plot(cos_h, linewidth=1.5, color='blue')
    plt.title('Gabor Filter - cos', fontsize=14, fontweight='bold')
    plt.xlabel('Time (samples)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('RatAlexia-Maria_343C2_gabor_cos.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(sin_h, linewidth=1.5, color='orange')
    plt.title('Gabor Filter - sin', fontsize=14, fontweight='bold')
    plt.xlabel('Time (samples)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('RatAlexia-Maria_343C2_gabor_sin.png', dpi=300, bbox_inches='tight')
    plt.close()

    M = 12
    fs = 44100

    filters_cos, filters_sin, freqs, sigmas = create_gabor_filterbank(M, size, fs)

    spectra_cos = []
    for i in range(M):
        spectrum_cos = scipy.fft.fft(filters_cos[i])
        spectra_cos.append(np.abs(spectrum_cos))
    spectra_cos = np.array(spectra_cos)

    half_size = size // 2
    freq_axis = np.linspace(0, fs/2, half_size)

    plt.figure(figsize=(14, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, M))
    for i in range(M):
        plt.plot(freq_axis, spectra_cos[i, :half_size],
                label=f'Filter {i+1}', linewidth=2, color=colors[i])
    plt.title('Gabor Filters Spectrum (Magnitude)', fontsize=14, fontweight='bold')
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.legend(loc='upper right', ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('RatAlexia-Maria_343C2_spectru_filtre.png', dpi=300, bbox_inches='tight')
    plt.close()

    
