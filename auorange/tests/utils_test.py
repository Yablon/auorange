import unittest
import numpy as np
from auorange.utils import multiband_linear_spectrogram


class TestMultibandLinear(unittest.TestCase):

  def testSplitFullBandLinear(self):
    fft_size = 16
    full_band_linear = np.arange(fft_size // 2 + 1).reshape([-1, 1])
    output = multiband_linear_spectrogram(full_band_linear,
                                          keep_linear_shape=False)
    np.testing.assert_array_equal(output[:, 0, :].reshape([-1]), [0, 1, 2])
    np.testing.assert_array_equal(output[:, 1, :].reshape([-1]), [2, 3, 4])
    np.testing.assert_array_equal(output[:, 2, :].reshape([-1]), [4, 5, 6])
    np.testing.assert_array_equal(output[:, 3, :].reshape([-1]), [6, 7, 8])

    output = multiband_linear_spectrogram(full_band_linear,
                                          keep_linear_shape=True)
    np.testing.assert_array_equal(output[:, 0, :].reshape([-1]),
                                  [0, 1, 2] + [0] * 6)
    np.testing.assert_array_equal(output[:, 1, :].reshape([-1]),
                                  [0] * 2 + [2, 3, 4] + [0] * 4)
    np.testing.assert_array_equal(output[:, 2, :].reshape([-1]),
                                  [0] * 4 + [4, 5, 6] + [0] * 2)
    np.testing.assert_array_equal(output[:, 3, :].reshape([-1]),
                                  [0] * 6 + [6, 7, 8])


if __name__ == '__main__':
  unittest.main()
