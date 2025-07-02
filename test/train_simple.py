import redCedar
import math
import random
import matplotlib.pyplot as plt

class model():
    def __init__(self):
        self.layerA = redCedar.Linear(2,20)
        self.sigmoidA = redCedar.Sigmoid()
        self.layerB = redCedar.Linear(20,20)
        self.sigmoidB = redCedar.Sigmoid()
        self.layerC = redCedar.Linear(20,2)
        self.sigmoidC = redCedar.Sigmoid()
        self.loss_ = redCedar.MSELoss()
    def loss(self,x,xHat):
        return self.loss_.forward(x,xHat)
    def forward(self, x):
        x = self.layerA.forward(x)
        x = self.sigmoidA.forward(x)
        x = self.layerB.forward(x)
        x = self.sigmoidB.forward(x)
        x = self.layerC.forward(x)
        x = self.sigmoidC.forward(x)
        return x

def train_network(steps=10000):
    lossAverage = 0
    myModel = model()
    optimizer = redCedar.Optim(0.001)
    avg_losses = []

    for step in range(steps):
        x = random.uniform(0,2*math.pi)
        y = random.uniform(0,2*math.pi)
        input_tensor = redCedar.Tensor([2,1])
        input_tensor[0] = x
        input_tensor[1] = y
        target = [(math.sin(x)+1)*0.5, (math.cos(y)+1)*0.5]
        pred = myModel.forward(input_tensor)
        loss = myModel.loss(pred, redCedar.Tensor.fromList(target, [2,1]))
        lossAverage += loss.toFloat()
        if step % 1000 == 0:
            print(f"Step {step}: input=({x:.2f},{y:.2f}) target=({target[0]:.2f},{target[1]:.2f}) pred={pred.toList()} loss={loss.toFloat():.4f}")
            print(f"Average loss: {lossAverage/(1000):.4f}")
            avg_losses.append(lossAverage/1000)
            lossAverage = 0
        optimizer.optimize(loss)

    print("\nFinal test:")
    for x_val, y_val in [(0,0), (1,1), (-1,-1), (math.pi/2, math.pi/2)]:
        input_tensor = redCedar.Tensor([2,1])
        input_tensor[0] = x_val
        input_tensor[1] = y_val
        target = [math.sin(x_val + y_val), math.cos(x_val + y_val)]
        pred = myModel.forward(input_tensor)
        print(f"input=({x_val:.2f},{y_val:.2f}) target=({target[0]:.2f},{target[1]:.2f}) pred={pred.toList()}, loss={myModel.loss(pred, redCedar.Tensor.fromList(target, [2,1])).toFloat():.4f}")
    # Plot average losses
    plt.plot([i*1000 for i in range(len(avg_losses))], avg_losses)
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Over Time')
    plt.show()

if __name__ == "__main__":
    train_network(steps=100000)
