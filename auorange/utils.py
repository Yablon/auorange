import matplotlib.pyplot as plt
import numpy as np


def plot(array):
  fig = plt.figure(figsize=(30, 5))
  ax = fig.add_subplot(111)
  ax.xaxis.label.set_color('grey')
  ax.yaxis.label.set_color('grey')
  ax.xaxis.label.set_fontsize(23)
  ax.yaxis.label.set_fontsize(23)
  ax.tick_params(axis='x', colors='grey', labelsize=23)
  ax.tick_params(axis='y', colors='grey', labelsize=23)
  plt.plot(array)
  plt.show()


def plot_spec(M):
  M = np.flip(M, axis=0)
  plt.figure(figsize=(18, 4))
  plt.imshow(M, interpolation='nearest', aspect='auto')
  plt.show()


def levinson_durbin(n, pAC):
  """levinson durbin's recursion

  Args:
      n (int): lpc order
      pAC (np.array): autocorrelation

  Returns:
      np.array: lpc coefficients
  """
  num_frames = pAC.shape[-1]
  pLP = np.zeros(shape=(n, num_frames), dtype=np.float32)
  pTmp = np.zeros(shape=(n, num_frames), dtype=np.float32)

  E = np.copy(pAC[0, :])
  for i in range(n):
    ki = np.copy(pAC[i + 1, :])
    for j in range(i):
      ki += pLP[j, :] * pAC[i - j, :]
    ki = ki / E

    c = np.maximum(1e-5, 1 - ki * ki)
    E *= c

    pTmp[i, :] = -ki
    for j in range(i):
      pTmp[j, :] = pLP[j, :] - ki * pLP[i - j - 1, :]
    for j in range(i + 1):
      pLP[j, :] = pTmp[j, :]

  return pLP


def multiband_linear_spectrogram(full_band_linear, keep_linear_shape=True):
  """split full band linear spectrogram to sub-band linear spectrogram

  Args:
      full_band_linear (np.array): full band linear spectrogram generated from mel_to_linear,
                                   shape is [FFT_SIZE // 2 + 1, num_frames]
      keep_linear_shape (bool, optional): whether to keep the shape of sub-band linear spectrogram
                                          same as full band linear spectrogram. Defaults to True.

  Returns:
      np.array: sub-band linear spectrograms, shape is [FFT_SIZE // 2 + 1, num_subbands, num_frames]
  """
  num_linear_bins = linear.shape[0]
  fft_size = (num_linear_bins - 1) * 2
  num_bands = fft_size // 2 // 4
  get_band = lambda idx: np.expand_dims(
      full_band_linear[idx * num_bands:(idx + 1) * num_bands + 1, :], axis=1)
  if keep_linear_shape:

    def pad_to_linear_shape(band, idx):
      pad_width = [[
          idx * num_bands, num_linear_bins - (idx + 1) * num_bands - 1
      ], [0, 0], [0, 0]]
      return np.pad(band, pad_width)

    bands = np.concatenate(
        [pad_to_linear_shape(get_band(i), i) for i in range(4)], axis=1)

  else:
    bands = np.concatenate([get_band(i) for i in range(4)], axis=1)
  return bands
