import torch
import torch.nn as nn
from iff_ok import LatticeMaskExpert
from Attention import MultiHeadAttentionLayer, FFN, EmbeddingLayer
import yaml
with open('param.yml') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)
device = ("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM=param['HIDDEN_DIM']
INCREASED=param['INCREASED']
NUM_CLASSES=param['NUM_CLASSES']
EXPANSION=param['EXPANSION']
DEPTH=param['DEPTH']
DIM=param['DIM']
Drop_out=param['Drop_out']
# MASK=Mask_For_Trans().to(device).forward()
import time
# mh=MultiHeadAttentionLayer().to(device)
   
# class MLP(nn.Module):
    # def __init__(self, in_features=HIDDEN_DIM, hidden_features=INCREASED, out_features=HIDDEN_DIM, dropout_p = 0):
    #     super(MLP, self).__init__()

    #     self.in_features = in_features
    #     self.hidden_features = hidden_features
    #     self.out_features = out_features

    #     # Neural Network 
    #     self.layer1 = nn.Linear(in_features, self.hidden_features)
    #     self.gelu = nn.GELU()
    #     self.linear2 = nn.Linear(self.hidden_features, out_features)
    #     self.drop = nn.Dropout(dropout_p)
    
    # def forward(self, x):
    #     # x shape: (n_samples, n_patches + 1, in_features)
    #     linear1 = self.layer1(x)
    #     gelu = self.gelu(linear1)
    #     gelu = self.drop(gelu)
    #     linear2 = self.linear2(gelu)
    #     output = self.drop(linear2)
    #     return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = HIDDEN_DIM,
                 drop_p: float = Drop_out,
                 forward_expansion: int = EXPANSION,
                 forward_drop_p: float = Drop_out,
                 ** kwargs):
        super().__init__()
        # self.mask=mask
        # self.mask=mask
        self.mh=MultiHeadAttentionLayer()
        # self.mlp=MLP()
        self.dropout=nn.Dropout(drop_p)
        self.fnn=FFN(emb_size=emb_size, expansion=forward_expansion, drop_p=forward_drop_p)
        #self.layernorm=nn.LayerNorm(HIDDEN_DIM)
        
    def forward(self, x, mask):
        # residual=x.view(x.shape[0], -1, HIDDEN_DIM)
        #추가함
        residual=x
        x=self.mh(x, mask)
        
        x=self.dropout(x)
        x=x+residual
        #x=self.layernorm(x) #여기임
         #추가함
        residual2=x
        x=self.fnn(x)
        x=self.dropout(x)
        # x=self.layernorm(x) #추가함
        # x=self.mlp(x)
        # x=self.dropout(x)
        x=x+residual2
        #x=self.layernorm(x) #여기임
		# x=self.layernorm(x) #추가함
        # x=self.mlp(x)
        # x=self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth=DEPTH, **kwargs):
        super().__init__()
        # self.mask=Mask_For_Trans().to(device).forward() #added
        # self.block=TransformerEncoderBlock()
        self.d=depth
        self.blocks=nn.ModuleList([TransformerEncoderBlock() for _ in range(self.d)])
        self.masks=nn.ModuleList([LatticeMaskExpert(HIDDEN_DIM).to(device) for _ in range(self.d)])
    def forward(self, x):
        for i in range(self.d):
            # x=self.blocks[i](x, pal, self.masks[i].to(device).forward())
            # start = time.time()
            mask = self.masks[i](x)
            # print("mask time :", time.time() - start)
            # start = time.time()
            x=self.blocks[i](x, mask)
            # print("block time :", time.time() - start)
            # print(x)
            # x=self.block(x)
        return x

class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = HIDDEN_DIM, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc=nn.Linear(emb_size, emb_size//2)
        self.mam=nn.Linear(emb_size//2, n_classes)
        self.layernorm=nn.LayerNorm(emb_size//2)
        self.softmax=nn.LogSoftmax(dim=-1)
        self.relu=nn.ReLU()
    def forward(self, x):
        x=self.fc(x)
        #x=self.layernorm(x)
        x=self.relu(x)
        x=self.mam(x)
        x=self.softmax(x)
        # hosung changed this 10/20 22:30.
        #x=x.view(-1, NUM_CLASSES)
        # print('classifi:')
        # print(x.shape)
        return x

class MaskFormer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.emb=EmbeddingLayer()
        self.te=TransformerEncoder()
        self.chead=ClassificationHead()
    def forward(self, x):
        # print("emb_start")
        embed=self.emb(x)
        # print("te_start")
        te=self.te(embed)
        # print("head_start")
        output=self.chead(te)
        # print("end")
        return output

# class MaskFormer(nn.Sequential):t
#     def __init__(self,**kwargs):
#         super().__init__(
#             EmbeddingLayer(),
#             TransformerEncoder(),
#             ClassificationHead()
#         )

        
# model=MaskFormer()
# print(model)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(f'The model has {count_parameters(model):,} trainable parameers')
