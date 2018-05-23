import numpy as np

# IMPORT DATA IS D * 1
with open('goblet_book.txt','r') as f:
    data = f.read()


m = 100
c = 10

def give_dictionary(data):
    dictionary = {}
    inverse_dictionary = {}
    for x in data:
        if x not in dictionary.keys():
            dictionary[x] = str(len(dictionary))
    for key in dictionary.keys():
        value = dictionary[key]
        inverse_dictionary[value] = key
    return dictionary, inverse_dictionary

def one_hot(dictionary, data, start, Parameters):  # Will not give the whole one-hoted list 80 * 1100000!
    len_dictionary = len(dictionary)
    one_hot_label = np.zeros([len_dictionary, Parameters.seq_len])
    data = list(data)
    for i in range(start, start + Parameters.seq_len):
        value = dictionary[data[i]]
        one_hot_label[int(value), i] = 1
    return one_hot_label



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
        self.seq_len = 25 # MAX TIME STEP
        self.h = np.zeros([m,self.seq_len-1])
        self.a = np.zeros([m,self.seq_len-1])
        self.o = np.zeros([c,self.seq_len-1])
        self.p = np.zeros([c,self.seq_len-1])

Parameters = Inter_parameters(m,c)


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
dictionary,_ = give_dictionary(data)
one_hot = one_hot(dictionary, data, 0, Parameters)
print(one_hot)



# def Compute_Gradient(Changable_parameters):
