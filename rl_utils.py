
# MIT License

# Copyright (c) 2018 Arsenii Ashukha

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE

import torch 
import random

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

class ReplayMemory:
    def __init__(self, capacity, device="cpu"):
        self.capacity = capacity
        self.memory = []
        self.device = device

    def push(self, state, action, next_state, reward, done):
        transition = torch.Tensor([state]).to(self.device), torch.tensor([action], dtype=torch.long).to(self.device), torch.Tensor([next_state]).to(self.device), torch.Tensor([reward]).to(self.device), torch.Tensor([done]).to(self.device)
        self.memory.append(transition)
        if len(self.memory) > self.capacity: del self.memory[0]

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)
        
        batch_state = torch.cat(batch_state)
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_done = torch.cat(batch_done)
        batch_next_state = torch.cat(batch_next_state)
        
        return batch_state, batch_action, batch_reward.unsqueeze(1), batch_next_state, batch_done.unsqueeze(1)
    
    def __len__(self):
        return len(self.memory)
