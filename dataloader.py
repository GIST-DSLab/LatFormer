import torch
from torch.utils.data import Dataset
import pandas as pd

DIM=10
TRAIN_PATH="/home/sundong/jiwon/MaskFomer/color_generate_trainset.csv"
VALID_PATH="/home/sundong/jiwon/MaskFomer/color_generate_validset.csv"
TEST_PATH="/home/sundong/jiwon/MaskFomer/color_generate_testset.csv"

class CustomDataset(Dataset):
    def __init__(self, csv_path):
        df=pd.read_csv(csv_path)
        self.inp=df.iloc[:, 0].values
        self.outp=df.iloc[:, 1].values

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        str_list=list(self.inp[idx])
        int_list=[eval(i) for i in str_list]
        i_list=torch.LongTensor(int_list)
        inp=torch.reshape(i_list, (DIM,DIM))
        
        ostr_list=list(self.outp[idx])
        oint_list=[eval(i) for i in ostr_list]
        o_list=torch.LongTensor(oint_list)
        outp=torch.reshape(o_list, (DIM,DIM))
        return inp, outp

dataset = CustomDataset(TRAIN_PATH)
validset = CustomDataset(VALID_PATH)
testset = CustomDataset(TEST_PATH)

#batchsize, 100







# df=pd.read_csv(csv_path)
# a=df.iloc[:, 0].values
# print(a)
# my_list=list(a[0])
# i_list=[eval(i) for i in my_list]
# t_list=torch.LongTensor(i_list)
# t1_list=torch.reshape(t_list, (5,5))
# print(t1_list)


# b=df.iloc[:, 1].values.reshapes(250000,1)
# b=df.iloc[:, 1].values
# print(b)
# by_list=list(a=b[0])
# b_list=[eval(i) for i in by_list]
# tb_list=torch.LongTensor(b_list)
# tb1_list=torch.reshape(tb_list, (5,5))
# print(tb1_list)
# print(df.head)
