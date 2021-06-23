#Problema "FizzBuzz"
#Para cada um dos numeros de 1 a 100 escreva:
#"Fizz" se divisivel por 3,
#"Buzz" se divisivel por 5,
#"FizzBuzz" se divisivel por 15,
# apenas o numero nos outros casos.

from typing import List

import numpy as np

from dasxlib.train import train
from dasxlib.nn import NeuralNet
from dasxlib.layers import Linear, Tanh
from dasxlib.optim import SGD

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]

def binary_encode(x: int) -> List[int]:
    #codificação binária por 10
    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(net,
        inputs,
        targets,
        num_epochs=5000,
        optimizer=SGD(lr=0.001))

for x in range(1, 101):
    predicted = net.forward(binary_encode(x))
    predicted_index = np.argmax(predicted)
    actual_index = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "Fizz", "Buzz", "FizzBuzz"]
    print(x, labels[predicted_index], labels[actual_index])