import math
from pydub import AudioSegment
import librosa
import numpy as np

import torch
import torchaudio
from torch import Tensor

def split_audio(file_path, chunk_size=30000):
    """
    file_path: path of audio file
    chunk_size: length to break into small pieces (Default value : 60,000ms)
    """
    sound = AudioSegment.from_file(file_path)
    chunks = math.ceil(len(sound) / chunk_size)
    file_name = file_path.split(".")[0]

    for i in range(chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        if end > len(sound):
            end = len(sound)
        chunk = sound[start:end]
        chunk.export(f"{file_name}-chunk{i}.wav", format="wav")
    
    return chunks

def parser(audio_path, audio_extension: str = 'pcm'):
    
    signal, _ = librosa.load(audio_path, sr=16000)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)

def join_text_chunks(text_list, chunk_size):
    new_list = []
    temp_text = ''
    for i, text in enumerate(text_list):
        temp_text += text
        if (i + 1) % chunk_size == 0:
            new_list.append(temp_text)
            temp_text = ''

    if temp_text:
        new_list.append(temp_text)
    return new_list
