#Usamos um otimizador para ajustar os parâmetros da rede neural
#baseado nos gradientes computados na retropropagação (backpropagation)

from dasxlib.nn import NeuralNet

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotADirectoryError

class SGD(Optimizer):
    #Gradiente de descida estcástica(Stochastic gradient descent) - suaviza a função
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    
    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad 
            #O gradiente define a direção que a função cresce mais rápido, isso significa
            #que se ajustarmos o parâmetro na direção oposta ao gradiente, essa será a
            #direção que a função decresce mais rápido, o que em teoria, assumindo que a 
            #função se comportará bem, isso fará com que a função de saída e as 
            #funções de perda sejam o gradiente menor, e a taxa de aprendizado que conhecemos, um
            #fator pequeno para nos certificarmos de que nossos passos não sejam muito grandes, 
            #então tomamos as derivadas apenas ou para pequenas mudanças e isso funcionará.