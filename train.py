from dataloader import dataset, validset, testset
from transformer import MaskFormer
from earlystopping import EarlyStopping
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import tqdm
import time
import yaml
import wandb
import numpy as np
import random
import pandas as pd

with open('param.yml') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

use_wandb = param['use_wandb']
SEED = param['SEED']
Name=param['name']
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(SEED)
random.seed(SEED)

EPOCHS=param['EPOCHS']
BATCH_SIZE=param['BATCH_SIZE']
NUM_CLASSES=param['NUM_CLASSES']
device = ("cuda" if torch.cuda.is_available() else "cpu")
max_norm=0.5
if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        config=param
    )
    wandb.run.name=f"Ex-{param['name']}"

def listToString(str_list):
    result = ""
    for s in str_list:
        result += str(s)
    return result.strip()

trainloader = DataLoader(
    dataset,
    batch_size = BATCH_SIZE,
    shuffle = True, 
    drop_last=True,
)
valloader = DataLoader(
    validset,
    batch_size = BATCH_SIZE,
    shuffle = True,
    drop_last=True,
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle = False,
    drop_last=True,
)

def train_model(model):

    # train the model
    model.train()
    train_loss = 0

    # for x_minibatch, y_minibatch in tqdm(trainloader, desc='Training'):
    for _, (x_minibatch, y_minibatch) in enumerate(tqdm.tqdm(trainloader, desc='Training')):
        # print(type(x_minibatch))
        optimizer.zero_grad()
        
        # hosung changed this 10/20 22:30.
        x_minibatch = x_minibatch.to(device) # (N, H*W)
        y_minibatch_preds = model(x_minibatch)  # (N, H*W, C) as a logit
        #y_minibatch_preds = y_minibatch_preds.view(BATCH_SIZE, -1, NUM_CLASSES) #64, 100, 10
        y_minibatch = y_minibatch.to(device)    # (N, H*W)
        y_minibatch = y_minibatch.view(BATCH_SIZE,-1)  # (N, H*W)
        
        #y_minibatch=F.one_hot(y_minibatch, num_classes=NUM_CLASSES).type(torch.FloatTensor).to(device)    # 64, 100, 10
        # y_minibatch=y_minibatch.type(torch.FloatTensor).to(device)    
        # y_minibatch_1=y_minibatch_1.view(-1, NUM_CLASSES).to(device)                          # 400, 10 -- 100, 10
        loss = criterion1(y_minibatch_preds.permute(0,2,1), y_minibatch) # (N, C, H*W) with longtensor target (N, H*W)
        # + criterion2(y_minibatch_preds_B, y_minibatch)
        # print(f'loss: {loss}')
        # print(f'pred: {y_minibatch_preds}')
        # print(f'Real: {y_minibatch}')
        
        # print(f'criterion2: {criterion2(y_minibatch_preds_B, y_minibatch)}')
        
        # pred = torch.argmax(y_minibatch_preds, dim=1)
        # pred = pred.view(BATCH_SIZE, -1)
        # #     # print(pred.shape)
        # y_label=torch.argmax(y_minibatch, dim=1)
        # y_label=y_label.view(BATCH_SIZE, -1)
        # print(f'y_label: {y_label[:5]}')
        # print(f'pred: {pred[:5]}')
        if use_wandb:
            wandb.log({
                'Train_several_Loss': loss,
                'Epoch': epoch,
                })
           
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        train_loss += loss.item()
    train_loss/=(len(trainloader.dataset))
    train_loss*=BATCH_SIZE
    # print(train_loss)
    # print(type(train_loss))
    print('Average Training Loss: {:.4f}'.format(train_loss))
    if use_wandb:
        wandb.log({
            'Train_Loss': train_loss,
			'Epoch': epoch,
            })
    return train_loss
def test_model(model):
    model.eval()

    test_loss = 0
    correct = 0
    real_correct=0
    step = 0

    with torch.no_grad():
        for _, (x_minibatch, y_minibatch) in enumerate(tqdm.tqdm(valloader, desc='Test')):
            step += 1
            # hosung changed this 10/20 22:30.
            x_minibatch = x_minibatch.to(device)
            y_minibatch_preds = model(x_minibatch) # (N, H*W, C)
            y_minibatch = y_minibatch.to(device)
            y_minibatch = y_minibatch.view(BATCH_SIZE,-1) # (N, H*W)
            #y_minibatch=F.one_hot(y_minibatch, num_classes=NUM_CLASSES).type(torch.FloatTensor) # 64, 100, 10
            #y_minibatch=y_minibatch.view(-1, NUM_CLASSES).to(device)
            # print(y_minibatch.shape)
            # for i in range(BATCH_SIZE)
            loss = criterion1(y_minibatch_preds.permute(0,2,1), y_minibatch) # (N, C, H*W) with longtensor target #(N, H*W)
            test_loss += loss.item()

            pred = torch.argmax(y_minibatch_preds, dim=-1) # (N, H*W)
            pred = pred.view(BATCH_SIZE, -1) # (N, H*W)
            # print(pred.shape)
            
            #y_label=torch.argmax(y_minibatch, dim=1)
            #y_label=y_label.view(BATCH_SIZE, -1)
            y_label = y_minibatch # (N, H*W)
            correct += pred.eq(y_label).sum().item()
            # print(f'x_label: {x_minibatch[2]}')
            # print(f'y_label: {y_label[2]}')
            # print(f'pred: {pred[2]}')
            for i in range(BATCH_SIZE):
                if (torch.equal(y_label[i],pred[i])):
                    real_correct+=1
                    # print(f'correct_x_label: {x_minibatch[i]}')
                    # print(f'correct_y_label: {y_label[i]}')
                    # print(f'correct_pred: {pred[i]}')
    # test_loss/=(step)
    test_loss/=(len(valloader.dataset))
    accuracy = correct / (100*BATCH_SIZE*step)
    real_acc=real_correct / (BATCH_SIZE*step)
    print('Average Test Loss: {:.4f}'.format(test_loss))
    print('Accuracy:{}/{}({:.2f}%)'.format(correct, 100*BATCH_SIZE*step, accuracy*100))
    print('Real acc:{}/{}({:.2f}%)'.format(real_correct, BATCH_SIZE*step, real_acc*100))
    if use_wandb:
        wandb.log({
            'Avg_Te_Loss': test_loss,
            'Acc': accuracy*100,
            'Correct' : correct,
            'Real_Correct' : real_correct,
            'Real_acc': real_acc*100,
            'Epoch': epoch,
            })
    
    return test_loss
model=MaskFormer().to(device)
# es=EarlyStopping(patience=10, delta=0, mode='min', verbose=True)
def weights_init(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif type(m) == nn.Embedding:
        torch.nn.init.xavier_uniform_(m.weight)

model.apply(weights_init)

# nSamples=torch.FloatTensor([0.25, 0.9166, 0.9166, 0.9166, 0.9166, 0.9166, 0.9166, 0.9166, 0.9166, 0.9166]).to(device)

# criterion1=nn.CrossEntropyLoss(nSamples)

criterion1= nn.NLLLoss()     #nn.CrossEntropyLoss()
# criterion2=nn.MSELoss()
# criterion=nn.HuberLoss()
optimizer=optim.AdamW(model.parameters(), lr=param['Lr'])
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5, verbose=True)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
start_time = time.time()
for epoch in range(EPOCHS):   # 데이터셋을 수차례 반복합니다.
    print(f'----- Epoch {epoch+1} -----')
    current_lr = get_lr(optimizer)
    print('current lr: {}'.format(current_lr))
    tr_loss = train_model(model)
    loss = test_model(model)
    scheduler.step()
    #es(loss)
    #if es.early_stop:
        #torch.save(model.state_dict(), f"{param['name']}.pth")
        #break
    print('time: %.4f min' % ((time.time() - start_time)/60))

cor=0
re_cor=0
st=0
te_loss=0
wrong_y=[]
wrong_pred=[]
#with torch.no_grad():
    #for _, (x_minibatch, y_minibatch) in enumerate(tqdm.tqdm(testloader, desc='Test')):
        #st+=1
        #x_minibatch = x_minibatch.to(device)
        #y_minibatch_preds = model(x_minibatch) # (N, H*W, C)
        #y_minibatch = y_minibatch.to(device)
        #y_minibatch = y_minibatch.view(BATCH_SIZE,-1) # (N, H*W)
								            #y_minibatch=F.one_hot(y_minibatch, num_classes=NUM_CLASSES).type
        #loss = criterion1(y_minibatch_preds.permute(0,2,1), y_minibatch) # (N, C, H*W) with longtensor target #(N, H*W)
        #te_loss += loss.item()
        #pred = torch.argmax(y_minibatch_preds, dim=-1) # (N, H*W)
        #pred = pred.view(BATCH_SIZE, -1) # (N, H*W)
        #y_label = y_minibatch # (N, H*W)
        #cor += pred.eq(y_label).sum().item()
        #for i in range(BATCH_SIZE):
            #if (torch.equal(y_label[i],pred[i])):
                #re_cor+=1
            #else:
                #a=listToString(y_label[i].flatten())
                #wrong_y.append(a)
                #b=listToString(pred[i].flatten())
                #wrong_pred.append(b)
#accuracy = cor / (100*BATCH_SIZE*st)
#real_acc = re_cor / (BATCH_SIZE*st)
#print('Accuracy:{}/{}({:.2f}%)'.format(cor, 100*BATCH_SIZE*st, accuracy*100))
#print('Real acc:{}/{}({:.2f}%)'.format(re_cor, BATCH_SIZE*st, real_acc*100))
print("hi")
#df = pd.DataFrame(wrong_y, columns=['input'])
#df['output']=wrong_pred
#df.to_csv("name.csv", index=False)

# param['name'] += 1
# with open('./param.yml', 'w') as f:
#     yaml.dump(param, f)
