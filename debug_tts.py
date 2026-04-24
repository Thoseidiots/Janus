import sys
sys.path.insert(0, 'C:\\Janus')
import torch
from janus_tts_v2 import JanusTTSv2, Phonemizer, length_regulate, SAMPLE_RATE

tts = JanusTTSv2('C:\\Janus\\janus_tts_v2_weights.pt')
ph = Phonemizer()
ids = torch.tensor([ph.phonemes_to_ids(ph.text_to_phonemes("hello janus"))], dtype=torch.long)

enc = tts.text_encoder(ids)
print("Encoded shape:", enc.shape)

T = enc.shape[1]
durs = torch.full((1, T), 6.0)
frames = length_regulate(enc, durs)
print("Frame features shape:", frames.shape)

mel = tts.decoder(frames, tts.style_vector)
print("Mel shape:", mel.shape)

wav = tts.vocoder(mel)
print("Waveform shape:", wav.shape)
print("Expected duration:", round(wav.shape[1] / SAMPLE_RATE, 2), "seconds")
