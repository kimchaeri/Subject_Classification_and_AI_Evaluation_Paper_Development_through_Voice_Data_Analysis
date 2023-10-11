import os
import json
import argparse
import urllib.request
import shutil
import numpy as np

import gensim
from konlpy.tag import Okt
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.cm as cm

import torch

from utils.voice_data_utils import *

okt = Okt()

if torch.cuda.is_available():
    device = 'cuda'
    print(device)
else:
    device = 'cpu'
    print(device)

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset_month', type=str, help='month of audio dataset', default=None)
args = argparser.parse_args()
print(args)

dataset_month = args.dataset_month

with open(f"./json/{dataset_month}/extracted_text_whisper.json", 'r') as f:    
    audio_text_dict = json.load(f)

path_list = list(audio_text_dict.keys())
text_list = list()

for i in range(len(path_list)):
    translated_text = audio_text_dict[path_list[i]]
    translated_text_split = translated_text.split('.')
    text_list.extend(translated_text_split)

filtered_text_list = [item for item in text_list if item != '']
new_list = join_text_chunks(filtered_text_list, 10)

url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ko/master/stopwords-ko.txt"
stopwords = urllib.request.urlopen(url).read().decode('utf-8').splitlines()
stopwords.append("거")
stopwords.append("안")
stopwords.append("해")
stopwords.append("뭐")
stopwords.append("그냥")
stopwords.append("지금")
stopwords.append("음")
stopwords.append("말")
stopwords.append("아아")
stopwords.append("더")
stopwords.append("씨")
stopwords.append("수")
stopwords.append("진짜")
stopwords.append("다시")
stopwords.append("게")
stopwords.append("요")
stopwords.append("전")

processed_docs = [[term for term in okt.nouns(doc) if term not in stopwords] for doc in new_list]

dictionary = Dictionary(processed_docs)

corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

num_topics = 5  
num_words = 10 
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

colors = cm.rainbow(np.linspace(0, 1, num_words))

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

for idx, topic in lda_model.show_topics(formatted=False, num_words=num_words):
    print(topic)
    font_path = '/home/s20225103/voice_data_analysis/NanumGothic.ttf'
    font_prop = font_manager.FontProperties(fname=font_path)
    rc('font', family=font_prop.get_name())

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 1, 1)

    words = [word for word, _ in topic]
    print(words)
    weights = [weight for _, weight in topic]
    print(weights)
    
    ax.barh(range(num_words), weights, color=colors, align='center')
    ax.set_yticks(range(num_words), words, fontproperties=font_prop, fontsize=15)  
    ax.set_yticklabels(words, fontproperties=font_prop, fontsize=15)
    ax.invert_yaxis()  
    ax.set_xlabel('Weight', fontsize=18)
    ax.set_title(f'Topic {idx}', fontsize=18)
    plt.xticks(fontsize=12)
    fig.savefig(f"./topic_{idx}.png", bbox_inches='tight', pad_inches=0.1)
