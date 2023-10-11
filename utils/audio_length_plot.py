import os
import argparse
import librosa
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

from utils.voice_data_utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset_path', type=str, help='path of audio dataset', default=None)
args = argparser.parse_args()
print(args)

dataset_path = args.dataset_path

audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav') or f.endswith('.m4a')]
print(audio_files)
durations = []
for audio_file in audio_files:
    audio_path = os.path.join(dataset_path, audio_file)
    duration = librosa.get_duration(path=audio_path) /60
    durations.append(duration)

font_path = '/home/s20225103/voice_data_analysis/NanumGothic.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
rc('font', family=font_prop.get_name())

bins = range(0, int(max(durations)) + 6, 5)
plt.hist(durations, bins=bins)
plt.xticks(bins)
plt.xlabel('분', fontproperties=font_prop)
plt.ylabel('개수', fontproperties=font_prop)
plt.title('오디오 데이터 길이 분포', fontproperties=font_prop)
plt.grid(True)

plt.savefig("./audio_length.png")
plt.show()