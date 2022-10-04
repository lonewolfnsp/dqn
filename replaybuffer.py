import torch 
import random 
from collections import namedtuple, deque 

Exp = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer():    
    
    def __init__(self, capacity, batchsize, device ):        
        self.Memories=deque(maxlen=capacity)
        self.batchsize=batchsize
        self.device=device 

    def add(self, *args): 
        if len(args)==5:
            state = torch.tensor(args[0], dtype=torch.float)
            action =  torch.tensor(args[1], dtype=torch.int64)
            reward = torch.tensor(args[2], dtype=torch.float)
            next = torch.tensor(args[3], dtype=torch.float)
            done = torch.tensor(args[4], dtype=torch.float)

            exp=Exp(state, action, reward, next, done)
            self.Memories.append(exp)
    
    def __len__(self):
        return len(self.Memories)
        
            
    def sample(self):
        exps=random.sample(self.Memories, k=self.batchsize )
        states=[None]*self.batchsize
        actions=[None]*self.batchsize
        rewards=[None]*self.batchsize
        nexts=[None]*self.batchsize
        dones=[None]*self.batchsize
        for i in range(self.batchsize):
            states[i]=exps[i].state
            actions[i]=exps[i].action
            rewards[i]=exps[i].reward
            nexts[i]=exps[i].next_state
            dones[i]= exps[i].done

        return torch.stack(states).to(self.device), \
            torch.stack(actions).to(self.device), \
            torch.stack(rewards).to(self.device), \
            torch.stack(nexts).to(self.device), \
            torch.stack(dones).to(self.device)

