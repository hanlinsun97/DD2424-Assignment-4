import numpy as np

# IMPORT DATA IS D * 1



# define initial parameters

class Changable_parameters(object):
    def __init__(self,m,d,c):
        self.W = np.random.normal(0,0.001,[m,m])
        self.b = np.zeros([m,1])
        self.U = np.random.normal(0,0.001,[m,d])
        self.V = np.random.normal(0,0.001,[c,m])
        self.C = np.zeros([c,1])

class Unchangable_parameters(object):
    def __init__(self,m,c):
        self.initial_learning_rate = 0.01
        self.regularization = 0.001
        self.T = 50 # MAX TIME STEP
        self.h = np.zeros([m,self.T-1])
        self.a = np.zeros([m,self.T-1])
        self.o = np.zeros([c,self.T-1])
        self.p = np.zeros([c,self.T-1])
    
def softmax(x):
    y = np.exp(x)/np.sum(np.exp(x), axis=0)
    return y

def Compute_P(Unchangable_parameters, Changable_parameters, X):
    for i in range(len(self.a)):
        if i == 0:
            h0 = np.zeros([m,1])
            self.a[:,i] = self.W * h0 + self.U * X[i] + self.b
            self.h[:,i] = np.tanh(self.a[:,i])
            self.o[:,i] = self.V * self.h[:,i] + self.C
            self.p[:,i] = softmax(self.o[:,i])
        else:
            self.a[:, i] = self.W * self.h[:,i-1] + self.U * X[i] + self.b
            self.h[:, i] = np.tanh(self.a[:, i])
            self.o[:, i] = self.V * self.h[:, i] + self.C
            self.p[:, i] = softmax(self.o[:, i])
    return Changable_parameters

def Compute_loss(Changable_parameters, Y):
    loss = -np.sum(np.log(Y.T * self.p))
    return loss

def Compute_Gradient(Changable_parameters):
        
