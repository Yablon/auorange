import math

import librosa
import numpy as np
import scipy
from scipy.fftpack import ifft

from auorange.utils import levinson_durbin


def load_wav(path, sample_rate):
  sr, raw_data = scipy.io.wavfile.read(path)
  if sample_rate != sr:
    raise ValueError('sample rate not equal')
  raw_data = raw_data.astype(np.float32)
  return (raw_data + 32768) / 65535. * 2 - 1


def save_wav(wav, path, sample_rate):
  data = (wav + 1) / 2 * 65535. - 32768
  scipy.io.wavfile.write(path, sample_rate, data.astype(np.int16))


class AudioLPC:

  def __init__(self, lpc_order, clip_lpc, sample_rate, f0=40.):
    self.lpc_order = lpc_order
    theta = (2 * np.pi * f0 / sample_rate)**2
    self.lag_window = np.exp(
        [[-0.5 * theta * i**2] for i in range(lpc_order + 1)])
    self.clip_lpc = clip_lpc

  def autocorrelation_to_lpc(self, ac):
    ac = ac[0:self.lpc_order + 1, :]
    ac = ac * self.lag_window
    return levinson_durbin(self.lpc_order, ac)

  def linear_to_autocorrelation(self, linear):
    power = linear**2
    fft_power = np.concatenate([power, power[::-1, :][1:-1, :]], axis=0)
    return ifft(fft_power, n=fft_power.shape[-1], axis=0).real

  def linear_to_lpc(self, linear, repeat=None):
    autocorrelation = self.linear_to_autocorrelation(linear)
    lpcs = self.autocorrelation_to_lpc(autocorrelation)
    lpcs = -1 * lpcs[::-1, :]
    if repeat is not None:
      return np.repeat(lpcs, repeat, axis=-1)
    return lpcs

  def lpc_predict(self, lpcs, signal_slice):
    pred = np.sum(lpcs * signal_slice, axis=0)
    if self.clip_lpc:
      pred = np.clip(pred, -1., 1.)
    return pred

  def lpc_reconstruction(self, lpcs, audio):
    num_points = lpcs.shape[-1]
    if audio.shape[0] == num_points:
      audio = np.pad(audio, ((self.lpc_order, 0)), 'constant')
    elif audio.shape[0] != num_points + self.lpc_order:
      raise RuntimeError('dimensions of lpcs and audio must match')
    indices = np.reshape(np.arange(self.lpc_order), [-1, 1]) + np.arange(
        lpcs.shape[-1])
    signal_slices = audio[indices]
    pred = self.lpc_predict(lpcs, signal_slices)
    origin_audio = audio[self.lpc_order:]
    error = origin_audio - pred
    return origin_audio, pred, error


class LibrosaAudioFeature:

  def __init__(self, sample_rate, n_fft, num_mels, hop_length, win_length,
               lpc_extractor):
    self.sample_rate = sample_rate
    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.num_mels = num_mels
    self._mel_basis = librosa.filters.mel(self.sample_rate,
                                          self.n_fft,
                                          n_mels=self.num_mels,
                                          fmin=20.,
                                          fmax=sample_rate / 2)
    self._inv_mel_basis = np.linalg.pinv(self._mel_basis)
    self.lpc_extractor = lpc_extractor

  def mel_spectrogram(self, y):
    D = self._stft(y)
    S = self._linear_to_mel(np.abs(D))
    return normalize_spec(S)

  def linear_spectrogram(self, y):
    D = np.abs(self._stft(y))
    return normalize_spec(D)

  def mel_to_linear(self, mel):
    mel = denormalize_spec(mel)
    return np.maximum(np.dot(self._inv_mel_basis, mel), 1e-12)

  def _linear_to_mel(self, spectrogram):
    return np.dot(self._mel_basis, spectrogram)

  def mel_to_lpc(self, mel):
    inv_linear = self.mel_to_linear(mel)
    return self.lpc_extractor.linear_to_lpc(inv_linear, repeat=self.hop_length)

  def lpc_audio(self, mel, audio):
    lpcs = self.mel_to_lpc(mel)
    lpcs = lpcs[:, :audio.shape[-1]]
    return self.lpc_extractor.lpc_reconstruction(lpcs, audio)

  def _stft(self, y):
    return librosa.stft(y=y,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        win_length=self.win_length,
                        pad_mode='constant')

  def _istft(self, y):
    return librosa.istft(y,
                         hop_length=self.hop_length,
                         win_length=self.win_length)


def normalize_spec(spectrogram):
  return np.log(1. + 10000 * spectrogram)


def denormalize_spec(spectrogram):
  return (np.exp(spectrogram) - 1.) / 10000
