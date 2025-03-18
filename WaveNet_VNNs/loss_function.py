import torch
import math
import torch.nn as nn

def pwelch(x, fs=3152, nperseg=512, noverlap=341, nfft=512, window='hamming'):
    step = nperseg - noverlap
    device = x.device
    # Create the window
    if window == 'hann':
        window = torch.hann_window(nperseg, periodic=True, dtype=x.dtype, device=device)
    elif window == 'hamming':
        window = torch.hamming_window(nperseg, periodic=True, dtype=x.dtype, device=device)
    # Split signal into overlapping segments
    segments = x.unfold(0, nperseg, step)
    # Apply the window to each segment
    segments = segments * window
    # FFT and power spectrum calculation
    fft_segments = torch.fft.fft(segments, nfft, dim=-1)
    psd = torch.abs(fft_segments[:, :nfft // 2 + 1]) ** 2
    # Scale PSD based on MATLAB scaling
    win_power = (window**2).sum()
    psd /= win_power * fs
    psd[:, 1:-1] *= 2  # Double for single-sided spectrum
    # Average over segments
    psd_mean = psd.mean(dim=0)
    # Frequency vector
    freqs = torch.fft.fftfreq(nfft, 1 / fs)[:nfft // 2 + 1]
    return freqs, psd_mean

def compute_a_weighting(frequencies):
    """Compute A-weighting for a given set of frequencies."""
    f1 = 20.6
    f2 = 107.7
    f3 = 737.9
    f4 = 12194.0
    A1000 = -2.0
    num = (f4 ** 2) * (frequencies ** 4)
    den = (frequencies ** 2 + f1 ** 2) * torch.sqrt(frequencies ** 2 + f2 ** 2) * torch.sqrt(frequencies ** 2 + f3 ** 2) * (frequencies ** 2 + f4 ** 2)
    A = 20.0 * torch.log10(num / den) - A1000
    return A

class dBA_Loss(nn.Module):
    """dBA loss."""
    def __init__(self, fs, nfft, f_up, f_low=1):
        super(dBA_Loss, self).__init__()
        self.fs = fs
        self.nfft = nfft
        self.f_up = f_up
        self.f_low = f_low

    def forward(self, x):
        batch_size = x.shape[0]
        loss = 0.0
        for i in range(batch_size):
            f, pxx = pwelch(x[i, :], self.fs, nperseg=512, noverlap=341, nfft=512, window='hamming')
            A_weighting = compute_a_weighting(f)
            f_resolution = self.fs / self.nfft
            Lev_f_up = math.floor(self.f_up / f_resolution + 1)
            Lev_f_low = math.floor(self.f_low / f_resolution + 1)
            pxy_dBA = 10 * torch.log10(pxx) + A_weighting.to(x.device)
            level_A = 10 * torch.log10(torch.sum(10 ** (pxy_dBA[Lev_f_low-1:Lev_f_up] / 10), dim=0))
            loss += torch.sum(level_A)
        return loss / batch_size  

class NMSE_Loss(nn.Module):
    """NMSE loss."""
    def __init__(self):
        super(NMSE_Loss, self).__init__()

    def forward(self, en, dn):
        return 10 * torch.log10(torch.sum((en.squeeze())**2) / torch.sum((dn.squeeze())**2))


