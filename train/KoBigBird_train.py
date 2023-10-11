import os
import json
import tqdm
import argparse

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from utils.KoBigBirddataset import *
from utils.accuracy import *
from backbone.kobigbird_classifier import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if torch.cuda.is_available():
    device = 'cuda'
    print(device)
else:
    device = 'cpu'
    print(device)
    
argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_folder', type=str, help='folder name of checkpoint', default=None)
argparser.add_argument('--save_every', type=int, help='save point of checkpoints', default=20)
argparser.add_argument('--epoch', type=int, help='epoch num', default=111)
argparser.add_argument('--freeze', type=str2bool, help='freeze pretrained model or not', default=True)
args = argparser.parse_args()
print(args)

checkpoint_folder = args.checkpoint_folder
save_every = args.save_every
epoch = args.epoch
freeze = args.freeze

max_len = 1024
batch_size = 16
warmup_ratio = 0.1
num_epochs = epoch
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

'''
식음료 -> 음식
주거와 생활 -> 부동산 및 주거
회사/아르바이트 -> 직장
교육 -> 공부
스포츠/레저 -> 스포츠
계절/날씨 -> 날씨
미용 -> 자기 관리
상거래전반, 상거래 전반 -> 전자상거래
방송/연예, 영화/만화 -> 드라마 및 영화  
'''

TRAIN_DATA_PATH = '/home/s20225103/voice_data_analysis/data/AIHub/data/AIHub_data_train_label_adjust.json'
VAL_DATA_PATH = '/home/s20225103/voice_data_analysis/data/AIHub/data/AIHub_data_val_label_adjust.json'

with open(TRAIN_DATA_PATH, 'r') as f:
    aihub_train_data_json = json.load(f)
    
LABEL_NAME = {"음식" : 0, "부동산 및 주거" : 1, "교통" : 2, "직장" : 3, "군대" : 4, "공부" :5, "가족" : 6, "연애/결혼" : 7, "반려동물" : 8, "스포츠" : 9, "게임" : 10, "여행" : 11, "날씨" : 12, "사회이슈" : 13, "타 국가 이슈" : 14, "자기 관리" : 15, "건강" : 16, "전자상거래": 17, "드라마 및 영화" : 18}
with open(TRAIN_DATA_PATH, 'r') as f:
    aihub_train_data_json = json.load(f)

with open(VAL_DATA_PATH, 'r') as f:
    aihub_val_data_json = json.load(f)
    
aihub_train_data_list = list()
for i in range(len(aihub_train_data_json)):
    each_data_list = list()
    sub = aihub_train_data_json[str(i)]['subject']
    conv = aihub_train_data_json[str(i)]['conversation']

    each_data_list.append(conv)
    each_data_list.append(str(LABEL_NAME[sub]))
    aihub_train_data_list.append(each_data_list)

aihub_val_data_list = list()
for i in range(len(aihub_val_data_json)):
    each_data_list = list()
    sub = aihub_val_data_json[str(i)]['subject']
    conv = aihub_val_data_json[str(i)]['conversation']

    each_data_list.append(conv)
    each_data_list.append(str(LABEL_NAME[sub]))
    aihub_val_data_list.append(each_data_list)

try:
    kobigbird_model = AutoModel.from_pretrained("monologg/kobigbird-bert-base")

except requests.exceptions.ConnectionError as e:
    print("ConnectionError")
    time.sleep(30)
    kobigbird_model = AutoModel.from_pretrained("monologg/kobigbird-bert-base")

tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

model = KoBigBirdClassifier(kobigbird_model, max_len=max_len, batch_size=batch_size, device=device, dr_rate=0.5, freeze_opt=freeze)
model = model.to(device)
#model = nn.DataParallel(model).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

data_train = KoBigBirdDataset(aihub_train_data_list, 0, 1, tokenizer, max_len, True, False)
data_test = KoBigBirdDataset(aihub_val_data_list, 0, 1, tokenizer, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, drop_last=True)

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

for e in range(num_epochs):
    checkpoints_dir = os.path.join("/home/s20225103/voice_data_analysis/checkpoints", checkpoint_folder)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_data = data['input_data']
        label_data = data['label_data']
        label_data = label_data.long().to(device)

        out = model(input_data)
        loss = loss_fn(out, label_data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  
        train_acc += calc_accuracy(out, label_data)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    
    if e % save_every == 0:
        checkpoint_name = "-".join(["checkpoint", str(e) + ".pt"])
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            os.path.join(checkpoints_dir, checkpoint_name),
        )
        
    model.eval()
    for batch_id, data in enumerate(test_dataloader):
        input_data = data['input_data']
        label_data = data['label_data']
        input_data = input_data.to(device)
        label_data = label_data.long().to(device)
        
        out = model(input_data)
        test_acc += calc_accuracy(out, label_data)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))