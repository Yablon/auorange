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
