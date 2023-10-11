import os
import json
import argparse
import urllib.request
import shutil
import numpy as np

from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from konlpy.tag import Okt

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib.cm as cm

import torch

from nltk.tokenize import word_tokenize

from utils.voice_data_utils import *

okt = Okt()

class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        sent = sent[:1000000]
        word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result

custom_tokenizer = CustomTokenizer(okt)    

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

'''
for i in range(len(path_list)):
    translated_text = audio_text_dict[path_list[i]]
    text_list.append(translated_text)
'''
vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

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
stopwords.append("있어요")
stopwords.append("하고")
stopwords.append("예요")
stopwords.append("맞아요")
stopwords.append("너무")
stopwords.append("근데")
stopwords.append("이에요")
stopwords.append("많이")
stopwords.append("근데")
stopwords.append("''")

num_words=10

processed_documents = []
for document in new_list:
    processed_tokens = [token for token in okt.morphs(document) if token not in stopwords]
    processed_document = ' '.join(processed_tokens)
    processed_documents.append(processed_document)

model = BERTopic(embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
                 vectorizer_model=vectorizer,
                 nr_topics=5,
                 top_n_words=num_words,
                 calculate_probabilities=True)

print(len(processed_documents))
topics, probs = model.fit_transform(processed_documents)
for i in range(0, 5):
    word_list, weight_list = list(), list()
    try :
        for j in range(len(model.get_topic(i))):
            word = model.get_topic(i)[j][0]
            weight = model.get_topic(i)[j][1]
            weight = round(weight, 3)
            word_list.append(word)
            weight_list.append(weight)

        print(i,'번째 토픽 :', end=' ')
        for m in range(num_words):
            print(word_list[m]+"*"+str(weight_list[m]), end=' ')
        print("\n")
    except TypeError:
        print("TypeError")
        continue

colors = cm.rainbow(np.linspace(0, 1, num_words))

for i in range(0, 5):
    font_path = '/home/s20225103/voice_data_analysis/NanumGothic.ttf'
    font_prop = font_manager.FontProperties(fname=font_path)
    rc('font', family=font_prop.get_name())

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 1, 1)

    words = [word for word, _ in model.get_topic(i)]
    print(words)
    weights = [weight for _, weight in model.get_topic(i)]
    print(weights)
    ax.barh(range(num_words), weights, color=colors, align='center')
    ax.set_yticks(range(num_words), words, fontproperties=font_prop, fontsize=15)  
    ax.set_yticklabels(words, fontproperties=font_prop, fontsize=15)
    ax.invert_yaxis()  
    ax.set_xlabel('Weight', fontsize=18)
    ax.set_title(f'Topic {i}', fontsize=18)
    plt.xticks(fontsize=12)
    fig.savefig(f"./topic_{i}_BERTopic.png", bbox_inches='tight', pad_inches=0.1)
