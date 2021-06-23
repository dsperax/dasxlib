# Loss functions (funções de perda) medem o quão boas sao suas previsões,
# e podemos utilizar para justar os parametros da rede neural.

import numpy as np

from dasxlib.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    #Medium squared error: Erro quadrado médio
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum(predicted - actual)**2

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2*(predicted - actual)