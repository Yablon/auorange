import matplotlib.pyplot as plt

from auorange import auorange
from auorange.utils import plot, plot_spec

wav_name = 'wavs/LJ001-0001.wav'
sample_rate = 22050
n_fft = 2048
num_mels = 80
hop_length = 275
win_length = 1100
lpc_order = 16
clip_lpc = True

wav_data = auorange.load_wav(wav_name, 22050)

audio_lpc = auorange.AudioLPC(lpc_order, clip_lpc, sample_rate, f0=40.)
audio_processor = auorange.LibrosaAudioFeature(sample_rate, n_fft, num_mels,
                                               hop_length, win_length,
                                               audio_lpc)

mel_spec = audio_processor.mel_spectrogram(wav_data)

audio, pred, error = audio_processor.lpc_audio(mel_spec, wav_data)
auorange.save_wav(pred, 'wavs/pred.wav', sample_rate)
auorange.save_wav(audio, 'wavs/audio.wav', sample_rate)
auorange.save_wav(error, 'wavs/error.wav', sample_rate)

fig = plt.figure(figsize=(30, 5))
plt.subplot(311)
plt.ylabel('pred')
plt.xlabel('time')
plt.plot(pred)
plt.subplot(312)
plt.ylabel('audio')
plt.xlabel('time')
plt.plot(audio)
plt.subplot(313)
plt.ylabel('error')
plt.xlabel('time')
plt.plot(error)
plt.show()
