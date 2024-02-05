import numpy as np
import matplotlib.pyplot as plt
'''
N = 5

def shift_identity():
    return np.zeros((N,),dtype=np.int64)

def shift_reflection():
    return np.arange(N-1,-N,-2)

def shift_translation(d):
    return np.zeros((N,),dtype=np.int64) -d

def shift_rotation():
    return np.arange(N-1, N*N*(N-1)+1 , N-1) - np.floor(np.arange(0,N*N)/N)

def shift_upscale(h):
    return -(np.arange(0,N)%h + (h-1)*(np.arange(0,N)//h))

def make_ker(shift):
    L = shift.shape[0]
    freqdom = np.exp(-2j * np.pi / L * shift.reshape(L,1) @ np.arange(L).reshape(1,L))
    return np.round(np.fft.ifft(freqdom,axis=1).real).astype(np.int64)

def roll(k):
    L = k.shape[0]
    kcirc = np.concatenate([k,k],axis=1)
    kr = np.zeros((L,L))
    for i in range(L):
        kr[i] = kcirc[i,L-i:2*L-i]
    return kr

def make_ker2(shift):
    return roll(make_ker(shift)).astype(np.int64)

def apply(k, org):
    L = k.shape[0]
    M = np.zeros((L,L))
    circ = np.concatenate([org,org,org], axis=0)
    for j in range(L):
        for i in range(L):
            M[j,i] = (circ[j:j+L,i] @ k[j])
    return M

def apply2(kr,org):
    return kr@org


ref  = make_ker2(shift_reflection())
tr1  = make_ker2(shift_translation(1))
tr2  = make_ker2(shift_translation(2))
trn1 = make_ker2(shift_translation(-1))
sc2  = make_ker2(shift_upscale(2))
sch = sc2*0.5
eye  = make_ker2(shift_identity())
rot  = make_ker2(shift_rotation())


L = N*N

t = np.arange(N)
print(t)


print('\ntest 1d shift\n')
x = np.eye(N)
x = apply2(tr2,x)
print(x.astype(np.int64) @ t)

x = apply2(trn1,x)
print(x.astype(np.int64) @ t)

x = apply2(trn1,x)
print(x.astype(np.int64) @ t)



print('\ntest rotation\n')
# test rotation

t = np.zeros((N,N))
t[:N//2, :N//2] = np.arange(4).reshape(N//2,N//2)+1

print(t.astype(np.int64))
t = t.flatten()
print('move (2,1)')
xx = np.kron(apply2(sc2,x),apply2(sc2,x))
#print(xx.astype(np.int64))
print((xx@(t) ).astype(np.int64).reshape(N,N))
print('rot')
xx = apply2(rot,xx)
#print(xx.astype(np.int64))
print((xx@(t) ).astype(np.int64).reshape(N,N))
print('rot')
xx = apply2(rot,xx)
#print(xx.astype(np.int64))
print((xx@t).astype(np.int64).reshape(N,N))
print('rot')
xx = apply2(rot,xx)
#print(xx.astype(np.int64))
print((xx@t).astype(np.int64).reshape(N,N))
print('rot')
xx = apply2(rot,xx)
#print(xx.astype(np.int64))
print((xx@t).astype(np.int64).reshape(N,N))
x = np.eye(N)
'''

import torch
from torch import nn
from torch.nn import functional as F

device = ("cuda" if torch.cuda.is_available() else "cpu")

class LatticeMaskExpert(nn.Module):

    N = 10
    def __init__(self,hidden_dim) -> None:
        super().__init__()
        N = self.N

        def _shift_reflection():
            return np.arange(N-1,-N,-2)

        def _shift_translation(d):
            return np.zeros((N,),dtype=np.int64) -d

        def _shift_rotation():
            return np.arange(N-1, N*N*(N-1)+1 , N-1) - np.floor(np.arange(0,N*N)/N)

        def _shift_upscale(h):
            return -(np.arange(0,N)%h + (h-1)*(np.arange(0,N)//h))
        
        def _roll(k):
            L = k.shape[0]
            kcirc = np.concatenate([k,k],axis=1)
            kr = np.zeros((L,L))
            for i in range(L):
                kr[i] = kcirc[i,L-i:2*L-i]
            return kr

        def _make_ker(shift):
            L = shift.shape[0]
            freqdom = np.exp(-2j * np.pi / L * shift.reshape(L,1) @ np.arange(L).reshape(1,L))
            return _roll(np.fft.ifft(freqdom,axis=1).real.round())

        max_trans = np.ceil(np.log2(self.N)).astype(np.int64)
        self.max_trans = max_trans
        self.translate_masks = [torch.Tensor(_make_ker(_shift_translation(t))).to(device) for t in [ 2**j for j in range(max_trans-1, -1, -1) ]]
        self.rotation_mask   = torch.Tensor(_make_ker(_shift_rotation())).to(device)
        self.reflection_mask = torch.Tensor(_make_ker(_shift_reflection())).to(device)
        self.scale_masks     = [torch.Tensor(_make_ker(_shift_upscale(t))).to(device) for t in range(1,1+max_trans,1)]

        # Change This Model!!!
        self.extractor = nn.Sequential(nn.Flatten(),  nn.Linear(N*N*hidden_dim,  4*max_trans+2+3+1+4))

    def batchwisekron(self,a,b):
        return torch.stack([torch.kron(ai,bi) for ai, bi in zip(a,b)])
    
    def forward(self,x):
        batches = x.size(0)
        alphas = self.extractor(x)
        
        """ alphas = torch.Tensor([[-100,-100,-100,-100,100, -100,-100,-100,-100,100,\
                                 0,0,0, \
                                 0,0, \
                                 0,0,0,0,0, 0,0,0,0,0, 0, \
                                 0,0,100,0 ]]) """

        tralphas_x = F.sigmoid(alphas[:, :self.max_trans, None, None])
        tralphas_y = F.sigmoid(alphas[:, self.max_trans:self.max_trans*2, None, None])

        rotalphas = F.sigmoid(alphas[:, 2*self.max_trans:2*self.max_trans+3, None, None])
        
        refalpha_x = F.sigmoid(alphas[:, 2*self.max_trans+3, None, None])
        refalpha_y = F.sigmoid(alphas[:, 2*self.max_trans+4, None, None])

        scalphas_x = F.softmax(alphas[:, 2*self.max_trans+5 : 3*self.max_trans+5, None, None],dim=-3)
        scalphas_y = F.softmax(alphas[:, 3*self.max_trans+5 : 4*self.max_trans+5, None, None],dim=-3)
        scalpha_t = F.sigmoid(alphas[:, 4*self.max_trans+5, None, None])
        aggalphas = F.softmax(alphas[:, 4*self.max_trans+6:, None, None], dim=-3)
        
        # translation
        trans_mask_x = torch.eye(self.N).unsqueeze(0).to(device)
        trans_mask_y = torch.eye(self.N).unsqueeze(0).to(device)

        for idx, m in enumerate(self.translate_masks):
            trans_mask_x = tralphas_x[:,idx] * torch.matmul(m,trans_mask_x) + (1-tralphas_x[:,idx]) * trans_mask_x
            trans_mask_y = tralphas_y[:,idx] * torch.matmul(m,trans_mask_y) + (1-tralphas_y[:,idx]) * trans_mask_y
        
        trans_mask = self.batchwisekron(trans_mask_x,trans_mask_y)

        # rotation
        rot_mask = torch.eye(self.N*self.N).unsqueeze(0).to(device)
        for idx in range(3):
            rot_mask = rotalphas[:,idx] * torch.matmul(self.rotation_mask,rot_mask) + (1-rotalphas[:,idx])*rot_mask

        # reflection
        ref_mask_x = torch.eye(self.N).unsqueeze(0).to(device)
        ref_mask_y = torch.eye(self.N).unsqueeze(0).to(device)
        ref_mask_x = refalpha_x * torch.matmul(self.reflection_mask, ref_mask_x) + (1-refalpha_x) * ref_mask_x
        ref_mask_y = refalpha_y * torch.matmul(self.reflection_mask, ref_mask_y) + (1-refalpha_y) * ref_mask_y
        ref_mask = self.batchwisekron(ref_mask_x, ref_mask_y)

        # scaling
        sc_mask_x = torch.zeros((batches, self.N, self.N)).to(device)
        sc_mask_y = torch.zeros((batches, self.N, self.N)).to(device)
        for idx, m in enumerate(self.scale_masks):
            sc_mask_x += m * scalphas_x[:, idx]
            sc_mask_y += m * scalphas_y[:, idx]
        
        scale_mask = self.batchwisekron(sc_mask_x,sc_mask_y)

        scale_mask = scalpha_t * torch.transpose(scale_mask, -1,-2) + (1-scalpha_t) * scale_mask

        # aggregate

        return aggalphas[:,0] * trans_mask + aggalphas[:,1] * ref_mask + aggalphas[:,2] * rot_mask + aggalphas[:,3] * scale_mask

# exp = LatticeMaskExpert(10)
# with torch.no_grad():
#     plt.pcolor(exp(torch.rand((1, 900,10 ))).squeeze())
#     plt.show()
