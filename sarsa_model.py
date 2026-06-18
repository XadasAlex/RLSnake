import torch
import torch.nn as nn
import torch.optim as optim

class SarsaTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, next_action, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_action = torch.tensor(next_action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            next_action = torch.unsqueeze(next_action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()

        # on policy

        for idx in range(len(done)):
            if not done[idx]:
                next_q_values = self.model(next_state[idx])
                next_action_idx = torch.argmax(next_action[idx]).item()
                target_q = reward[idx] + self.gamma * next_q_values[next_action_idx]
            else:
                target_q = reward[idx]

            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = target_q

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss.item()
