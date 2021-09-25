import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter

from preprocessor import SMILES_Tokenizer
from dataset import CustomDataset
from networks import RNN_Decoder
from trainer import train_step

device = torch.device("cpu") # pytorch의 cuda GPU 사용
BATCH_SIZE = 64 # 연산 한 번에 들어가는 데이터 크기 / 전체를 한 번에 넣으면 학습 시간이 매우 오래걸리기 때문
EPOCHS = 25
NUM_TRAIN_EXAMPLES = 27000
# 전체 트레이닝 셋이 신경망을 통과한 횟수. 즉, 전체 트레이닝 셋을 25회 모델에 학습시킨다는 의미 / epochs를 높일 수록 손실 값이 내려가게 됨
# But, 너무 높이게 되면 과적합이 발생하기 때문에 적절한 파라미터를 찾는 것이 중요
num_layers = 1 # 레이어 개수 / RNN은 보통 1개
dropout_rate = 0.1
# 과적합 해결책 / 학습 시 뉴런을 임의로 삭제하여 학습하는 방법 / 훈련 시에는 임의의 비율(dropout ratio) 만큼 뉴런을 삭제
embedding_dim = 128
# 임베딩층은 크게 두 가지 인자를 받는데 첫번째 인자는 단어장의 크기, 두번째 인자는 임베딩 벡터의 차원
learning_rate = 1e-4 # 일반적으로 0.01이 초기값 / 학습 비용(cost)을 관찰 후 조금씩 조정 / 너무 크면 오버슈팅 문제 발생
vision_pretrain = True # 뭔지모르겠음(?)
save_path = f'./models/best_model.pt'


def get_dataset():
    train = pd.read_csv('data/train.csv')
    dev = pd.read_csv('data/dev.csv')
    train_dev = pd.concat([train, dev])

    for idx, row in tqdm(train_dev.iterrows()):  # index가 for문을 도는 동안 train 데이터프레임에 있는 행을 반복한다는 의미
        file = row['uid']
        smiles = row['SMILES']

    max_len = train_dev.SMILES.str.len().max()
    tokenizer = SMILES_Tokenizer(max_len)
    tokenizer.fit(train_dev.SMILES)
    seqs = tokenizer.txt2seq(train_dev.SMILES)
    labels = train_dev[['S1_energy(eV)', 'T1_energy(eV)']].to_numpy()
    seqs, labels = shuffle(seqs, labels, random_state=42)
    train_seqs = seqs[:NUM_TRAIN_EXAMPLES]
    train_labels = labels[:NUM_TRAIN_EXAMPLES]
    val_seqs = seqs[NUM_TRAIN_EXAMPLES:]
    val_labels = labels[NUM_TRAIN_EXAMPLES:]

    train_dataset = CustomDataset(train_seqs, train_labels)
    val_dataset = CustomDataset(val_seqs, val_labels)

    return train_dataset, val_dataset, max_len


if __name__=='__main__':

    train_dataset, val_dataset, max_len = get_dataset()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)

    model = RNN_Decoder(embedding_dim=embedding_dim, max_len=max_len, num_layers=num_layers, rate=dropout_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()

    loss_plot, val_loss_plot = [], []

    writer = SummaryWriter()

    for epoch in range(EPOCHS):
        total_loss, total_val_loss = 0, 0

        tqdm_dataset = tqdm(enumerate(train_dataloader))
        training = True
        for batch, batch_item in tqdm_dataset:
            batch_loss = train_step(batch_item, training, model, optimizer, criterion)
            total_loss += batch_loss

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Total Loss': '{:06f}'.format(total_loss / (batch + 1))
            })

        writer.add_scalar('Loss/train', total_loss/(batch+1), epoch)
        loss_plot.append(total_loss / (batch + 1))

        tqdm_dataset = tqdm(enumerate(val_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss = train_step(batch_item, training, model, optimizer, criterion)
            total_val_loss += batch_loss

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Total Val Loss': '{:06f}'.format(total_val_loss / (batch + 1))
            })
        val_loss_plot.append(total_val_loss / (batch + 1))
        writer.add_scalar('Loss/train', total_loss/(batch+1), epoch)

        if np.min(val_loss_plot) == val_loss_plot[-1]:
            torch.save(model, save_path)
