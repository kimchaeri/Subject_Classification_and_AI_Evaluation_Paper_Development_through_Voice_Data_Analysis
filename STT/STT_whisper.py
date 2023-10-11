import os
import io
import json
import warnings
import argparse
import subprocess
import librosa
import sys

import torch
import torch.nn as nn
import torchaudio

import whisper

from speechbrain.pretrained import EncoderClassifier

from utils.voice_data_utils import *

if torch.cuda.is_available():
    device = 'cuda'
    print(device)
else:
    device = 'cpu'
    print(device)

model = whisper.load_model("large")

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset_path', type=str, help='path of audio dataset', default=None)
args = argparser.parse_args()
print(args)

dataset_path = args.dataset_path

dataset_detail_month = dataset_path.split('/')[-1]

file_name = os.listdir(dataset_path)

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

audio_text_dict = {}

for i in range(len(file_name)):
    audio_file_path = os.path.join(dataset_path, file_name[i])
    print(audio_file_path)
    audio = AudioSegment.from_file(audio_file_path, format="m4a")
    new_audio_file_path = f"/home/s20225103/voice_data_analysis/data/exported_data/{dataset_detail_month}/{file_name[i].split('.')[0]}.wav"
    audio.export(new_audio_file_path, format="wav")
    
    subprocess.run('rm /home/s20225103/voice_data_analysis/*.wav', shell=True)

    length = split_audio(new_audio_file_path)

    final_text = ""
    for m in range(length):
        split_file_path = f"/home/s20225103/voice_data_analysis/data/exported_data/{dataset_detail_month}/{file_name[i].split('.')[0]}-chunk{m}.wav"
        print(split_file_path)
        sound = AudioSegment.from_wav(split_file_path)
        sound = sound.set_channels(1)
        sound.export(split_file_path, format="wav")
        
        signal = language_id.load_audio(split_file_path)
        prediction =  language_id.classify_batch(signal)[3][0]
        print(prediction)

        if prediction=='ko: Korean':
            result = model.transcribe(split_file_path, language="ko")
            print(result["text"])
            final_text+=result["text"]+" "

        elif prediction=='en: English':
            result = model.transcribe(split_file_path, language="en")
            print(result["text"])      
            final_text+=result["text"]+" "
                      
        else:
            result = model.transcribe(split_file_path, language="ko")
            print(result["text"])
            final_text+=result["text"]+" "
    
    audio_text_dict[audio_file_path] = final_text
    subprocess.run(f'rm /home/s20225103/voice_data_analysis/exported_data/{dataset_detail_month}/*.wav', shell=True)
    with open(f"./json/{dataset_detail_month}/extracted_text_whisper.json", 'w') as outfile:
        json.dump(audio_text_dict, outfile, indent=4, ensure_ascii=False) 
        