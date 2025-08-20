import torch
import torch.nn as nn
import torch.nn.functional as F

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PolicyNetwork(nn.Module): #Actor Network. Aim: Output a probability dist of each action from that particular state.
    def __init__(self,state_dim,action_dim):
        super(PolicyNetwork,self).__init__()
        self.layer1=nn.Linear(state_dim,256)
        self.layer2=nn.Linear(256,256)
        self.layer3=nn.Linear(256,128)
        self.mean=nn.Linear(128,action_dim)
        self.log_std=nn.Parameter(torch.zeros(action_dim)) #log_std is a parameter and is calculated during Training
    def forward(self,state):
        x=F.relu(self.layer1(state))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        mean=self.mean(x)  #Outputs the mean of the action distribution (vector of size 2 — one mean for acceleration, one for angle).

        #Clamping std to avoid explodin NaN values
        clamped_log_std=self.log_std.clamp(min=-4,max=1)
        std=torch.exp(clamped_log_std)
        return mean,std
    
class ValueNetwork(nn.Module): #Critic Network. Aim: Output a numerical value for a particular state.
    def __init__(self,state_dim):
        super(ValueNetwork,self).__init__()
        self.layer1=nn.Linear(state_dim,128)
        self.layer2=nn.Linear(128,128)
        self.layer3=nn.Linear(128,64)
        self.value=nn.Linear(64,1)
    def forward(self,state):
        x=F.relu(self.layer1(state))
        x=F.relu(self.layer2(x))
        x=F.relu(self.layer3(x))
        value=self.value(x)
        return value    