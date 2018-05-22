import numpy as np

# IMPORT DATA IS D * 1
with open('goblet_book.txt','w') as f:



# define initial parameters
def initialization(m,d,c):
    W = np.random.normal(0,0.001,[m,m])
    b = np.zeros([m,1])
    U = np.random.normal(0,0.001,[m,d])
    V = np.random.normal(0,0.001,[c,m])
    C = np.zeros([c,1])
return W, b, U, V, C

class Inter_parameters(object):
    def __init__(self,m,c):
        self.initial_learning_rate = 0.01
        self.regularization = 0.001
        self.T = 50 # MAX TIME STEP
        self.h = np.zeros([m,self.T-1])
        self.a = np.zeros([m,self.T-1])
        self.o = np.zeros([c,self.T-1])
        self.p = np.zeros([c,self.T-1])

Parameters = Inter_parameters()


class Changable_parameters(object):

    def __init__(self,W,b,U,V,C):
        self.W = W
        self.b = b
        self.U = U
        self.V = V
        self.C = C

    def softmax(x):
        y = np.exp(x)/np.sum(np.exp(x), axis=0)
        return y

    def Compute_P(self, Parameters, X):
        for i in range(len(Parameters.a)):
            if i == 0:
                h0 = np.zeros([m,1])
                Parameters.a[:,i] = self.W * h0 + self.U * X[i] + self.b
                Parameters.h[:,i] = np.tanh(self.a[:,i])
                Parameters.o[:,i] = self.V * self.h[:,i] + self.C
                Parameters.p[:,i] = softmax(self.o[:,i])
            else:
                Parameters.a[:, i] = self.W * self.h[:,i-1] + self.U * X[i] + self.b
                Parameters.h[:, i] = np.tanh(self.a[:, i])
                Parameters.o[:, i] = self.V * self.h[:, i] + self.C
                Parameters.p[:, i] = softmax(self.o[:, i])
        return Parameters

    def Compute_loss(Parameters, Y):
        loss = -np.sum(np.log(Y.T * Parameters.p))
        return loss





# def Compute_Gradient(Changable_parameters):
