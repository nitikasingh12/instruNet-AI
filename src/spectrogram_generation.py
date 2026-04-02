import os
import numpy as np
import librosa

# ── Paths ───────────────────────────────────────────────────────
dataset_path = "C:/Users/NITIKA KUMARI/instrunet-ai/data/processed"
output_path  = "C:/Users/NITIKA KUMARI/instrunet-ai/data/spectrograms_npy"

os.makedirs(output_path, exist_ok=True)

# ── Parameters ──────────────────────────────────────────────────
SAMPLE_RATE    = 22050
DURATION       = 3
N_SAMPLES      = SAMPLE_RATE * DURATION
N_MELS         = 128
HOP_LENGTH     = 512
N_FFT          = 2048
TARGET_FRAMES  = 128

# ── Process each instrument folder ──────────────────────────────
for instrument in os.listdir(dataset_path):
    instrument_path = os.path.join(dataset_path, instrument)
    save_folder     = os.path.join(output_path, instrument)
    os.makedirs(save_folder, exist_ok=True)

    for file in os.listdir(instrument_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(instrument_path, file)

        try:
            # load audio, trim silence, fix length
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            y, _  = librosa.effects.trim(y, top_db=30)

            if len(y) > N_SAMPLES:
                y = y[:N_SAMPLES]
            else:
                y = np.pad(y, (0, max(0, N_SAMPLES - len(y))), 'constant')

            # mel spectrogram
            mel    = librosa.feature.melspectrogram(
                y=y, sr=sr,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                n_mels=N_MELS,
                fmax=SAMPLE_RATE // 2
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)

            # delta features
            delta1 = librosa.feature.delta(log_mel, order=1)
            delta2 = librosa.feature.delta(log_mel, order=2)

            # stack channels: (freq, time, 3)
            features = np.stack([log_mel, delta1, delta2], axis=-1)

            # pad/truncate time to TARGET_FRAMES
            current_frames = features.shape[1]
            if current_frames > TARGET_FRAMES:
                features = features[:, :TARGET_FRAMES, :]
            elif current_frames < TARGET_FRAMES:
                pad      = TARGET_FRAMES - current_frames
                features = np.pad(features, ((0,0),(0,pad),(0,0)), mode='constant')

            # save as .npy
            save_path = os.path.join(save_folder, file.replace('.wav', '.npy'))
            np.save(save_path, features.astype(np.float32))

        except Exception as e:
            print(f"Error processing {file}: {e}")

print("Spectrograms saved as .npy arrays.")