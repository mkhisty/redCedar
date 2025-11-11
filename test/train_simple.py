import redCedar
import math
import random
import matplotlib.pyplot as plt

class model():
    def __init__(self):
        self.layerA = redCedar.Linear(2,20)
        self.ReLUA = redCedar.ReLU()
        self.layerB = redCedar.Linear(20,20)
        self.ReLUB = redCedar.ReLU()

        self.layerC = redCedar.Linear(20,20)
        self.ReLUC = redCedar.ReLU()
        
        self.layerD = redCedar.Linear(20,20)
        self.ReLUD = redCedar.ReLU()
        self.layerE = redCedar.Linear(20,2)
        self.loss_ = redCedar.MSELoss()
    def loss(self,x,xHat):
        return self.loss_.forward(x,xHat)
    def forward(self, x):
        x = self.layerA.forward(x)
        x = self.ReLUA.forward(x)
        x = self.layerB.forward(x)
        x = self.ReLUB.forward(x)
        x = self.layerC.forward(x)
        x = self.ReLUC.forward(x)
        x = self.layerD.forward(x)
        x = self.ReLUD.forward(x)
        x = self.layerE.forward(x)
        return x

def train_network(steps=10000):
    lossAverage = 0
    myModel = model()
    optimizer = redCedar.Optim(0.005)
    avg_losses = []
    for step in range(steps):
        x = random.uniform(0,10)
        y = random.uniform(0,10)
        input_tensor = redCedar.Tensor([2,1])
        input_tensor[0] = x
        input_tensor[1] = y
        target = [x-y,x+y]
        pred = myModel.forward(input_tensor)
        loss = myModel.loss(pred, redCedar.Tensor.fromList(target, [2,1]))
        lossAverage += loss.toFloat()
        if (step+1) % 1000 == 0:
            print(f"Step {step}: input=({x:.2f},{y:.2f}) target=({target[0]:.2f},{target[1]:.2f}) pred={pred.toList()} loss={loss.toFloat():.4f}")
            print(f"Average loss: {lossAverage/(1000):.4f}")
            avg_losses.append(lossAverage/(1000))
            lossAverage = 0
            print(optimizer.lr)
#        if (current_loss - current_loss * 0.05) < lastLoss < (current_loss + current_loss * 0.05):
#            optimizer.lr -= (optimizer.lr*0.001)
        optimizer.optimize(loss)

    print("\nFinal test:")
    for i in range(0,4):
        input_tensor = redCedar.Tensor([2,1])
        x = random.uniform(0,1)
        y = random.uniform(0,1)
        input_tensor[0] = x
        input_tensor[1] = y
        target = [math.sin(x+y), math.cos(x+y)]
        pred = myModel.forward(input_tensor)
        print(f"input=({x:.2f},{y:.2f}) target=({target[0]:.2f},{target[1]:.2f}) pred={pred.toList()}, loss={myModel.loss(pred, redCedar.Tensor.fromList(target, [2,1])).toFloat():.4f}")
    # Plot average losses
    plt.plot([i*1000 for i in range(len(avg_losses))], avg_losses)
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Over Time')
    plt.show()

if __name__ == "__main__":
    train_network(steps=100000)
