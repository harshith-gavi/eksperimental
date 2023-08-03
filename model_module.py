import os
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import math

b_j0 = 0.1  # neural threshold baseline
gamma = .5  # gradient scale
lens = 0.3

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        # temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma
        # return grad_input

act_fun_adp = ActFun_adp.apply

def mem_update_adp(inputs, mem, spk, tau_adp,tau_m, b):
    alpha = tau_m    
    rho = tau_adp

    b = rho * b + (1 - rho) * spk
    B = b_j0 + 1.8 * b

    d_mem = (-mem + inputs) * alpha
    mem = mem + d_mem
    inputs_ = mem - B

    spk = act_fun_adp(inputs_)
    mem = (1-spk)*mem

    return mem, spk, B, b


def output_Neuron(inputs, mem, tau_m):
    """
    The read out neuron is leaky integrator without spike
    """
    d_mem = inputs
    mem = (1-tau_m) * mem + d_mem * tau_m
  
    return mem


class sigmoid_beta(nn.Module):
    def __init__(self, alpha = 1.3):
        super(sigmoid_beta,self).__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True

    def forward(self, x):
        if (self.alpha == 0.0):
            return x
        else:
            return torch.sigmoid(self.alpha*x)

class SNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_timesteps, P):
        super(SNN, self).__init__()
        
        self.P = P
        
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        self.n_timesteps = n_timesteps
        
        self.rnn_name = 'SNN'

        self.layer1_x = nn.Linear(input_size, hidden_size)
        self.layer1_r = nn.Linear(hidden_size, hidden_size)
        self.layer1_tauM = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.layer1_tauAdp = nn.Linear(hidden_size+hidden_size, hidden_size)

        self.layer2_x = nn.Linear(hidden_size, hidden_size)
        self.layer2_r = nn.Linear(hidden_size, hidden_size)
        self.layer2_tauM = nn.Linear(hidden_size+hidden_size, hidden_size)
        self.layer2_tauAdp = nn.Linear(hidden_size+hidden_size, hidden_size)
   
        self.layer3_x = nn.Linear(hidden_size, output_size)
        self.layer3_tauM = nn.Linear(output_size+output_size, output_size)

        self.act1m = sigmoid_beta()
        self.act1a = sigmoid_beta()
        self.act2m = sigmoid_beta()
        self.act2a = sigmoid_beta()
        self.act3 = sigmoid_beta()

        nn.init.xavier_normal_(self.layer1_x.weight)
        nn.init.orthogonal_(self.layer1_r.weight)
        nn.init.xavier_normal_(self.layer1_tauM.weight)
        nn.init.xavier_normal_(self.layer1_tauAdp.weight)

        nn.init.xavier_normal_(self.layer2_x.weight)
        nn.init.orthogonal_(self.layer2_r.weight)
        nn.init.xavier_normal_(self.layer2_tauM.weight)
        nn.init.xavier_normal_(self.layer2_tauAdp.weight)

        nn.init.xavier_normal_(self.layer3_x.weight)
        nn.init.xavier_normal_(self.layer3_tauM.weight)

        nn.init.zeros_(self.layer1_x.bias)
        nn.init.zeros_(self.layer1_tauM.bias)
        nn.init.zeros_(self.layer1_tauAdp.bias)
        nn.init.zeros_(self.layer2_x.bias)
        nn.init.zeros_(self.layer2_tauM.bias)
        nn.init.zeros_(self.layer2_tauAdp.bias)
        nn.init.zeros_(self.layer3_x.bias)
        nn.init.zeros_(self.layer3_tauM.bias)
 
        
    def forward(self, inputs, h,i=None):
        self.fr = 0
        outputs, hiddens = [], []
 
        b, l, dim = inputs.shape
        T = l

        for x_i in range(l):
            x = torch.squeeze(inputs, dim = 1)
            
            if l>99:
                x = torch.split(inputs, split_size_or_sections=1, dim=1)
                i = x_i
                x = x[x_i]
                x = torch.squeeze(x, dim = 1)          
            
            dense_x = self.layer1_x(x) + self.layer1_r(h[1])
            tauM1 = self.act1m(self.layer1_tauM(torch.cat((dense_x, h[0]), dim = -1)))
            tauAdp1 = self.act1a(self.layer1_tauAdp(torch.cat((dense_x, h[2]), dim = -1)))        
            mem_1, spk_1, _, b_1 = mem_update_adp(dense_x, mem = h[0], spk = h[1], tau_adp = tauAdp1, tau_m = tauM1, b = h[2])

            
            dense_x2 = self.layer2_x(spk_1) + self.layer2_r(h[1])
            tauM2 = self.act2m(self.layer2_tauM(torch.cat((dense_x2, h[3]), dim = -1)))
            tauAdp2 = self.act2a(self.layer2_tauAdp(torch.cat((dense_x2, h[5]), dim = -1)))  
            mem_2, spk_2, _, b_2 = mem_update_adp(dense_x2, mem = h[3], spk = h[4], tau_adp = tauAdp2, tau_m = tauM2, b = h[5])


            dense3_x = self.layer3_x(spk_2)
            tauM3 = self.act3(self.layer3_tauM(torch.cat((dense3_x, h[6]), dim = -1)))
            mem_3 = output_Neuron(dense3_x, mem = h[6], tau_m = tauM3)

            h = (mem_1,spk_1,b_1,
                mem_2,spk_2,b_2, 
                mem_3)

            f_output = F.log_softmax(mem_3, dim=1)
            outputs.append(f_output)
            hiddens.append(h)

            self.fr = self.fr+ (spk_1.detach().cpu().numpy().mean()+spk_2.detach().cpu().numpy().mean())/2.
                
        final_state = h
        self.fr = self.fr/T
        return outputs, final_state, hiddens

class SeqModel(nn.Module):
    def __init__(self, ninp, nhid, nout, wnorm=True, n_timesteps=1.4, parts=100):
        super(SeqModel, self).__init__()
        self.nout = nout
        self.nhid = nhid
        self.rnn_name = 'Seq-SNN'
        self.network = SNN(input_size=ninp, hidden_size=nhid, output_size=nout,n_timesteps=n_timesteps, P=parts)

    def forward(self, inputs, hidden):
        outputs, hidden, hiddens= self.network.forward(inputs, hidden)
        recon_loss = torch.zeros(1, device=inputs.device)
        return outputs, hidden, recon_loss

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (weight.new(bsz,self.nhid).uniform_(),
                weight.new(bsz,self.nhid).zero_(),
                weight.new(bsz,self.nhid).fill_(b_j0),

                weight.new(bsz,self.nhid).uniform_(),
                weight.new(bsz,self.nhid).zero_(),
                weight.new(bsz,self.nhid).fill_(b_j0),

                weight.new(bsz,self.nout).zero_()
                )
