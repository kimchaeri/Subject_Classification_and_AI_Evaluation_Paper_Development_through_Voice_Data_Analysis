import os
import io
import json
import urllib
import requests
import argparse
import warnings
import subprocess
import numpy as np

from google.cloud import speech_v1p1beta1 as speech
from google.oauth2 import service_account
from pyAudioAnalysis import audioSegmentation as aS
from pydub import AudioSegment

from speechbrain.pretrained import EncoderClassifier
import librosa

import torch
import torchaudio

from utils.voice_data_utils import *

warnings.filterwarnings('ignore')


if torch.cuda.is_available():
    device = 'cuda'
    print(device)
else:
    device = 'cpu'
    print(device)

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
        sound = AudioSegment.from_wav(split_file_path)
        sound = sound.set_channels(1)
        sound.export(split_file_path, format="wav")
        
        signal = language_id.load_audio(split_file_path)
        prediction =  language_id.classify_batch(signal)[3][0]
        print(prediction)
        
        with io.open(split_file_path, "rb") as audio_file:
            content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)

        if prediction=='ko: Korean':
            config = speech.types.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                enable_automatic_punctuation=True,
                language_code='ko-KR'
            )
            
        if prediction=='en: English':
            config = speech.types.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                enable_automatic_punctuation=True,
                language_code='en-US'
            )
        
        else: 
            config = speech.types.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                enable_automatic_punctuation=True,
                language_code='ko-KR'
            )
            
        response = client.recognize(config=config, audio=audio)

        for result in response.results:
            extracted_text = result.alternatives[0].transcript
            print(extracted_text)
            final_text+=extracted_text+" "
            
    text_list.append(final_text)
    audio_text_dict[audio_file_path] = final_text

    with open(f"./json/{dataset_detail_month}/extracted_text_google.json", 'w') as outfile:
        json.dump(audio_text_dict, outfile, indent=4, ensure_ascii=False) 



        
        
