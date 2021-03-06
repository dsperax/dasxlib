#As redes neurais são feitas de camadas(layers). Cada uma
#dessas camadas precisa passar a entrada (forward) e propagar gradientes
#para tras(backward). Ex -> uma rede neural pode ter o seguinte formato:
# inputs -> Linear -> Tanh -> Linear -> output
from typing import Callable, Dict

import numpy as np

from dasxlib.tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) ->Tensor:
        #Produz as saidas (outputs) correspondentes a esses inputs
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        #retropropaga (Backpropagate) o gradiente através da camada
        raise NotImplementedError

class Linear(Layer):
    #Calcula saidas = entrada @w (matriz múltipla por algum peso) + b(tendencia ou bias) (ax + b)
    def __init__(self, input_size: int, output_size: int) -> None:
        #inputs: (batch_size, input_size)
        #outputs: (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)
    
    def forward(self, inputs: Tensor) -> Tensor:
        #saidas(outputs) = entradas(inputs) @w + b (@ = multiplicação matricial)
        self.inputs = inputs #salva copia dos inputs para usar na backpropagation
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        # se y = f(x) e x = a * b + c
        # então:
        # dy/da = f'(x) * b
        # dy/db = f'(x) * a
        # dy/dc = f'(x)
        
        # se y = f(x) e x = a @ b + c 
        # então:
        # dy/da = b.T @ f'(x)
        # dy/db = a.T @ f'(x)
        # dy/dc = f'(x)

        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    #a camadas e ativação apenas aplica uma função aos elementos de entrada(inputs)
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        #se y = f(x) e x = g(z)
        #então dy/dz = f'(x) * g'(z)
        return self.f_prime(self.inputs) * grad

#Camadas de ativação (activation layers) - aqui pode criar quais quiser.

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


'''
Tentativa de implementar método de ficonacci
#iterator

class Fibonacci:
  def __init__(self, maximo=1000000):
    # Inicializa os dois primeiros numeros
    self.elemento_atual, self.proximo_elemento = 0, 1
    self.maximo = maximo

  def __iter__(self):
    # Retorna o objeto iterável (ele próprio: self)
    return self

  def __next__(self):
    # Fim da iteração, raise StopIteration
    if self.elemento_atual > self.maximo:
      raise StopIteration

    # Salva o valor a ser retornado
    valor_de_retorno = self.elemento_atual

    # Atualiza o próximo elemento da sequencia
    self.elemento_atual, self.proximo_elemento = self.proximo_elemento, self.elemento_atual + self.proximo_elemento

    return valor_de_retorno
        
# Executa nosso código
if __name__ == '__main__':
  # Cria nosso objeto iterável
  objeto_fibonacci = Fibonacci(maximo=1000000)

  # Itera nossa sequencia
  for fibonacci in objeto_fibonacci:
    print("Sequencia: {}".format(fibonacci))

#generator

def fibonacci(maximo):
  # Inicialização dos elementos
  elemento_atual, proximo_elemento = 0, 1
  
  # Defina a condição de parada
  while elemento_atual < maximo:
    # Retorna o valor do elemento atual
    yield elemento_atual

    elemento_atual, proximo_elemento = \
        proximo_elemento, elemento_atual + proximo_elemento

if __name__ == '__main__':
  # Cria um generator de números fibonacci menor que 1 milhão
  fibonacci_generator = fibonacci(1000000)

  # Mostra na tela toda a sequencia
  for numero_fibonacci in fibonacci_generator:
    print("Sequencia:",numero_fibonacci)

'''