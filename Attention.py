import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary as summary
import yaml
with open('param.yml') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

DIM=param['DIM']
BATCH_SIZE=param['BATCH_SIZE']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM=param['HIDDEN_DIM']


# print(x_em.shape)
class EmbeddingLayer(nn.Module):
    def __init__(self, HIDDEN_DIM=HIDDEN_DIM):
        super().__init__()
        self.HIDDEN_DIM=HIDDEN_DIM
        # self.p=torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device)
        self.embedding=nn.Embedding(10, self.HIDDEN_DIM)
    def forward(self, x):
        x=self.embedding(x)
        x=x.view(BATCH_SIZE, -1, self.HIDDEN_DIM)
        # pal=self.embedding(self.p)
        # print('embedding: {}'.format(self.embedding.weight))
        return x
        # return x, pal
        
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, HIDDEN_DIM=HIDDEN_DIM):
        super().__init__()

        self.HIDDEN_DIM = HIDDEN_DIM # 임베딩 차원
        #self.heads = heads #multihead 추가
        #self.head_dim = HIDDEN_DIM // heads
        self.fc_q = nn.Linear(HIDDEN_DIM, HIDDEN_DIM) # Query 값에 적용될 FC 레이어
        self.fc_k = nn.Linear(HIDDEN_DIM, HIDDEN_DIM) # Key 값에 적용될 FC 레이어
        self.fc_v = nn.Linear(HIDDEN_DIM, HIDDEN_DIM) # Key 값에 적용될 FC 레이어
        # self.fc_pal = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        # self.alpha=nn.Parameter(torch.randn(1))
        # self.alpha_extractor = nn.Sequential(nn.Flatten(), nn.Linear(DIM*DIM*HIDDEN_DIM, 1), nn.Sigmoid())
        # self.alpha = nn.Parameter(torch.Tensor([0.8]))

        self.scale = self.HIDDEN_DIM**(0.5)

    def forward(self, x, mask):
        # query: [batch_size, query_len, HIDDEN_DIM]
        # key: [batch_size, key_len, HIDDEN_DIM]
        
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)
        # print('QKV: {}'.format(Q))
        # P = self.fc_pal(pal)
        # Q = Q.view(BATCH_SIZE, -1, self.heads, self.head_dim)
        # K = K.view(BATCH_SIZE, -1, self.heads, self.head_dim)
        # 
        # Attention Energy 계산
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale

        attention = torch.softmax(energy, dim=-1)
        
        A=attention*mask
        #value 곱
        scale=torch.matmul(torch.matmul(A, torch.ones(DIM*DIM, 1).to(device)).permute(0, 2, 1), torch.ones(DIM*DIM, 1).to(device))
        
        
        # color_energy = torch.matmul(V, P.permute(0, 2, 1)) / self.scale
        # color_att=torch.softmax(color_energy, dim=-1)
        # print('att: {}'.format(color_att))
        # print('P_B: {}'.format(P_B.shape))
        # color_pred = torch.matmul(color_att, P)
        # print('pred: {}'.format(color_pred))
        # print('V: {}'.format(V))
        # print('pal: {}'.format(P))
        # print('alpha: {}'.format(self.alpha))
        
        # print(V.shape)
        # ma=torch.matmul(A, F.sigmoid(self.alpha)*V+(1-F.sigmoid(self.alpha))*color_pred)/scale
        # ma=torch.matmul(A, F.sigmoid(self.alpha)*V+(1-F.sigmoid(self.alpha))*color_pred)/scale
        # alpha = self.alpha_extractor(x).unsqueeze(-1)
        ma=torch.matmul(A, V)/scale
        # print(ma)
        #output
        return ma
    
class FFN(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(), #dropout, zoneout, relu 특성 조합
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )

# model = MultiHeadAttentionLayer(
#         HIDDEN_DIM=256,
# )

# model.to(device)
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f'The model has {count_parameters(model):,} trainable parameters')

# x=torch.FloatTensorensor.randn(25, 256) -> 앞에 x_em 있음
# summary(model, (batch_size, 25, 256))

# result = model(x.to(device), Mask_expert_transition(1, 2).to(device))
# print(result.shape)
