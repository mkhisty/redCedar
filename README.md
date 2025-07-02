#redCedar


A Machine learning framework built from scratch, for learning purposes.

![image](https://github.com/user-attachments/assets/90e65b01-ffaa-494f-869b-e9c6282f1a6b)

In this example, it tried to fit [x,y] t0 [y^2x,x^2y]

Usage:
```python
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
        self.layerB = redCedar.Linear(20,20)
        self.sigmoidB = redCedar.Sigmoid()
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
        return x

myModel = model()
optimizer = redCedar.Optim(0.01)
x = random.uniform(0,1)
y = random.uniform(0,1)
input_tensor = redCedar.Tensor([2,1])
input_tensor[0] = x
input_tensor[1] = y
target = [(x**2)*y, (y**2)*x]
xHat = myModel.forward(input_tensor)
loss = myModel.loss(xHat, redCedar.Tensor.fromList(target, [2,1]))
optimizer.optimize(loss)
```


Its still a work in progress, suggestions would be appreciated.

To use, build using cmake, then put outputted file in same directory as python file. Example of python usage given in test/
