import torch
import torch.nn as nn

class TensorNormalization(nn.Module):     # [warning] now for cifar only
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)
    def extra_repr(self) -> str:
        return 'mean=%s, std=%s'%(self.mean, self.std)
    
def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

class MergeTemporalDim(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1)
    
    def extra_repr(self):
        return 'input.flatten(0,1)'

class ExpandTemporalDim(nn.Module):     # [Warning] 名字叫 expand, 有点误导性，实际上是从 T*B 中恢复 [T,B,...]
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)
    
    def extra_repr(self):
        return 'T={}'.format(self.T)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

class RateBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem + x[t, ...]
            spike = ((mem - 1.) > 0).float()
            mem = (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        out = out.mean(0).unsqueeze(0)
        grad_input = grad_output * (out > 0).float()
        return grad_input


class LIF(nn.Module):     # [notice] auto adjust input shape and control the output shape
    def __init__(self, T=4, bp_mode='bptt', thresh=1.0, tau=1., gama=1.0, input_decay=False):
        super(LIF, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.ratebp = RateBp.apply
        self.bp_mode = bp_mode
        self.T = T
        self.input_decay = input_decay
        
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim()
        
        if self.T == None:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        if self.bp_mode == 'bptr':      # bp_mode = BPTR
            x = self.expand(x)
            x = self.ratebp(x)
            x = self.merge(x)
 
        elif self.bp_mode == 'bptt':        # bp_mode = BPTT
            x = self.expand(x)
            mem = 0
            spike_pot = []
            for t in range(self.T):
                if self.input_decay==False:
                    mem = mem * self.tau + x[t, ...]
                elif self.input_decay==True:
                    mem = (mem + x[t, ...])*self.tau
                spike = self.act(mem - self.thresh, self.gama)
                mem = (1 - spike) * mem
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
            
        else:       # bp_mode = None, which backs to ANN with ReLU()
            x = self.relu(x)
            
        return x     # output shape [T*B, C, H, W]
    
    def extra_repr(self):
        return 'T={}, threshhold={}, tau={}, gama={}, mode={}, input_decay={}'.format(self.T, self.thresh, self.tau, self.gama, self.bp_mode, self.input_decay)

def add_dimention(x, T):
    x.unsqueeze_(0)     # [warning] T 放在 dim 1？
    x = x.repeat(T, 1, 1, 1, 1)
    return x


class SeqToANNContainer(nn.Module):    
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):     # input shape [T, B, C, H, W] 
        y_shape = [x_seq.shape[0], x_seq.shape[1]]    
        y_seq = self.module(x_seq.flatten(0, 1))     
        y_shape.extend(y_seq.shape[1:])
        
        return y_seq.reshape(y_shape)     # output shape [T, B, C, H, W]
 