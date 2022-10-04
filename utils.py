import torch 
import torch.nn as nn 
import torch.optim  as optim 
import copy 
import time
import numpy as np




def saveTrainedModel(model, filename):
    name=f'{filename}.pth'
    torch.save(model.state_dict(), name)

def loadModel(model, filename):
    name=f'{filename}.pth'
    model.load_state_dict(torch.load(name) )    

class DQN (nn.Module):    

    def __init__(self, num_states, num_actions, layers):
        super().__init__() 
        
        self.model=nn.Sequential(
            nn.Linear(num_states, layers[0]),
            nn.ReLU(),
            nn.Linear(layers[0] , layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1] , num_actions)        
        )
        
    def forward(self, states):                
        return self.model(states)             

def softupdate(target, model, tau):
    for target_param, model_param in zip(target.parameters(), model.parameters()):        
        target_param.data.copy_(tau*model_param.data + (1.-tau)*target_param.data)

def getQAction(dqnModel, state, device):
    qState=torch.tensor(state, dtype=torch.float).unsqueeze(0) # batch with 1 sample
    qsa=dqnModel(qState.to(device))
    action=torch.argmax(qsa, dim=1).item()
    return action 

def runDQNAgent(dqnModel, env, device, max_step=-1, fps=25):
    state=env.reset()
    rewards=0.  
    dqnModel.eval()   
    breakout= max_step>-1
    step=0
    while True:
        action=getQAction(dqnModel, state, device)
        next_state, reward, done, _,_ = env.step(action)
        rewards+=reward 
        state=next_state
        env.render()
        print(f'\r rewards gaained: {rewards:.2f}', end="")
        time.sleep(1.0/fps)
        step+=1
        if done or (breakout and step== max_step):
            print(f'\nTerminated with rewards={rewards:.2f}')
            break         