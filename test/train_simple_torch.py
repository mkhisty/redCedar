import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerA = nn.Linear(2,20)
        self.sigmoidA = nn.Sigmoid()
        self.layerB = nn.Linear(20,20)
        self.sigmoidB = nn.Sigmoid()

        self.layerC = nn.Linear(20,2)
            
        self.loss_ = nn.MSELoss()
    

    
    def loss(self, x, xHat):
        return self.loss_(x, xHat)
    def forward(self, x):
        x = self.layerA(x)
        x = self.sigmoidA(x)
        x = self.layerB(x)
        x = self.sigmoidB(x)
        x = self.layerC(x)
        return x

def train_network(steps=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    myModel = Model().to(device)
    optimizer = optim.SGD(myModel.parameters(), lr=0.001)
    avg_losses = []
    lossAverage = 0
    for step in range(steps):
        x = random.uniform(0, math.pi*2)
        y = random.uniform(0, math.pi*2)
        input_tensor = torch.tensor([[x], [y]], dtype=torch.float32, device=device)
        target = torch.tensor( [[math.sin(x+y)], [math.cos(x+y)]], dtype=torch.float32, device=device)
        pred = myModel(input_tensor.T)  # shape (1,2)
        loss = myModel.loss(pred.T, target)
        lossAverage += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step+1) % 1000 == 0:
            pred0 = float(pred[0,0])
            pred1 = float(pred[0,1])
            print(f"Step {step}: input=({x:.2f},{y:.2f}) target=({target[0,0]:.2f},{target[1,0]:.2f}) pred=({pred0:.2f},{pred1:.2f}) loss={loss.item():.4f}")
            print(f"Average loss: {lossAverage/1000:.4f}")
            avg_losses.append(lossAverage/1000)
            lossAverage = 0
    print("\nFinal test:")
    for x_val, y_val in [(0,0), (1,1), (-1,-1), (math.pi/2, math.pi/2)]:
        input_tensor = torch.tensor([[x_val], [y_val]], dtype=torch.float32, device=device)
        target = torch.tensor([[math.sin(x_val + y_val)], [math.cos(x_val + y_val)]], dtype=torch.float32, device=device)

        pred = myModel(input_tensor.T)
        pred0 = float(pred[0,0])
        pred1 = float(pred[0,1])
        print(f"input=({x_val:.2f},{y_val:.2f}) target=({target[0,0]:.2f},{target[1,0]:.2f}) pred=({pred0:.2f},{pred1:.2f}), loss={myModel.loss(pred.T, target).item():.4f}")
    plt.plot([i*1000 for i in range(len(avg_losses))], avg_losses)
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Over Time')
    plt.show()

if __name__ == "__main__":
    train_network(steps=100000) 