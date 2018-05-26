import numpy as np
import matplotlib.pyplot as plt
# IMPORT DATA IS D * 1
from readfile import readfile

data = readfile()
data_new = []
for i in range(len(data)):
    for x in data[i]:
        data_new.append(x)
data = data_new
m = 100
d = 402
MAX_EPOCH = 100

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
    one_hot_label_x = np.zeros([len_dictionary, Parameters.seq_len])
    one_hot_label_y = np.zeros([len_dictionary, Parameters.seq_len])
    data = list(data)
    for i in range(start, start + Parameters.seq_len):
        value = dictionary[data[i]]
        one_hot_label_x[int(value), i-start] = 1
    for j in range(start+1, start+Parameters.seq_len + 1):
        value = dictionary[data[j]]
        one_hot_label_y[int(value), j-start-1] = 1
    return one_hot_label_x,  one_hot_label_y, len_dictionary


# define initial parameters
def initialization(m,d):
    W = np.random.rand(m,m) * 0.01
    b = np.zeros([m,1])
    U = np.random.rand(m,d) * 0.01
    V = np.random.rand(d,m) * 0.01
    C = np.zeros([d,1])
    return W, b, U, V, C

class Inter_parameters(object):
    def __init__(self,m,d):
        self.initial_learning_rate = 0.1
        self.regularization = 0.001
        self.seq_len = 25 # MAX TIME STEP
        self.h = np.zeros([m,self.seq_len])
        self.a = np.zeros([m,self.seq_len])
        self.o = np.zeros([d,self.seq_len])
        self.p = np.zeros([d,self.seq_len])
        self.h0 = np.zeros([m,1])

class Gradient_class(object):

    def __init__(self, Changable, Parameters):
        self.grad_W = np.zeros(np.shape(Changable.W))
        self.grad_b = np.zeros(np.shape(Changable.b))
        self.grad_U = np.zeros(np.shape(Changable.U))
        self.grad_V = np.zeros(np.shape(Changable.V))
        self.grad_c = np.zeros(np.shape(Changable.C))
        self.grad_h = np.zeros(np.shape(Parameters.h))
        self.grad_a = np.zeros(np.shape(Parameters.a))
        self.grad_o = np.zeros(np.shape(Parameters.o))


    def AdamGradOptimizer(self,Changable, Parameters, W_before, V_before, U_before, b_before, C_before):

        W_after = W_before + np.power(self.grad_W,2)
        V_after = V_before + np.power(self.grad_V,2)
        U_after = U_before + np.power(self.grad_U,2)
        b_after = b_before + np.power(self.grad_b,2)
        C_after = C_before + np.power(self.grad_c,2)
        Changable.W = Changable.W - Parameters.initial_learning_rate / np.sqrt(W_after + 1e-6) * self.grad_W
        Changable.V = Changable.V - Parameters.initial_learning_rate / np.sqrt(V_after + 1e-6) * self.grad_V
        Changable.U = Changable.U - Parameters.initial_learning_rate / np.sqrt(U_after + 1e-6) * self.grad_U
        Changable.b = Changable.b - Parameters.initial_learning_rate / np.sqrt(b_after + 1e-6) * self.grad_b
        Changable.C = Changable.C - Parameters.initial_learning_rate / np.sqrt(C_after + 1e-6) * self.grad_c
        W_before = W_after
        V_before = V_after
        U_before = U_after
        b_before = b_after
        C_before = C_after

        return Changable, W_after, V_after, U_after, b_after, C_after



class Changable_parameters(object):

    def __init__(self,W,b,U,V,C):
        self.W = W
        self.b = b
        self.U = U
        self.V = V
        self.C = C


    def Compute_P(self, Parameters, X):

        def softmax(x):
            y = np.exp(x)/np.sum(np.exp(x), axis=0)
            return y

        for i in range(Parameters.seq_len):
            if i == 0:
                h0 = Parameters.h0
                Parameters.a[:,i] = np.reshape(np.reshape(np.dot(self.W, h0),[-1,1]) + np.dot(self.U, np.reshape(X[:,i],[-1,1])) + self.b, m)
                Parameters.h[:,i] = np.tanh(Parameters.a[:,i])
                Parameters.o[:,i] = np.reshape(np.dot(self.V, np.reshape(Parameters.h[:,i],[-1,1])) + self.C, d)
                Parameters.p[:,i] = softmax(Parameters.o[:,i])
            else:
                Parameters.a[:,i] = np.reshape(np.dot(self.W, np.reshape(Parameters.h[:, i-1],[-1,1])) + np.dot(self.U, np.reshape(X[:,i],[-1,1])) + self.b, m)
                Parameters.h[:,i] = np.tanh(Parameters.a[:,i])
                Parameters.o[:,i] = np.reshape(np.dot(self.V, np.reshape(Parameters.h[:,i],[-1,1])) + self.C, d)
                Parameters.p[:,i] = softmax(Parameters.o[:,i])
        Parameters.h0 = Parameters.h[:,Parameters.seq_len-1]
        return Parameters

    def Compute_loss(self, Parameters, Y):
        loss = -np.sum(Y * np.log(Parameters.p))
        return loss

    def Compute_prediction(self,Parameters, inverse_dictionary, X):
        def softmax(x):
            y = np.exp(x)/np.sum(np.exp(x), axis=0)
            return y

        i = 0
        h0 = Parameters.h0
        Parameters.a[:,i] = np.reshape(np.reshape(np.dot(self.W, h0),[-1,1]) + np.dot(self.U, np.reshape(X,[-1,1])) + self.b, m)
        Parameters.h[:,i] = np.tanh(Parameters.a[:,i])
        Parameters.o[:,i] = np.reshape(np.dot(self.V, np.reshape(Parameters.h[:,i],[-1,1])) + self.C, d)
        Parameters.p[:,i] = softmax(Parameters.o[:,i])
        cp = np.cumsum(Parameters.p[:,0])
        a = np.random.rand()                                                             #TODO
        ixs = np.where(cp - a > 0)[0]
        ii = ixs[0]
        charac = np.zeros([d,1])
        charac[ii] = 1
        return charac

    def Compute_Gradient(self, Gradient, Parameters, Changable, X, Y):
        Gradient.grad_o = (Parameters.p - Y).T
        Gradient.grad_V = np.dot(Gradient.grad_o.T, Parameters.h.T)
    #    print(Gradient.grad_o.shape)
        for i in range(Parameters.seq_len)[::-1]:
            if i == Parameters.seq_len - 1:
                Gradient.grad_h[:,i] = np.dot(Gradient.grad_o[i,:].T, Changable.V)
                Gradient.grad_a[:,i] = Gradient.grad_h[:,i] * (1 - np.tanh(Parameters.a[:,i]) * np.tanh(Parameters.a[:,i]))
            else:
                Gradient.grad_h[:,i] = np.dot(Gradient.grad_o[i,:].T, Changable.V) + np.dot(Changable.W ,Gradient.grad_a[:,i+1])
                Gradient.grad_a[:,i] = Gradient.grad_h[:,i] * (1 - np.power(np.tanh(Parameters.a[:,i]),2))

        H_inter = Parameters.h
        H_inter = np.delete(H_inter, Parameters.seq_len-1, 1)
        H_inter = np.c_[np.reshape(Parameters.h0,[m,1]), H_inter]

        Gradient.grad_W = np.dot(Gradient.grad_a, H_inter.T)
        Gradient.grad_U = np.dot(Gradient.grad_a, X.T)
        Gradient.grad_b = np.reshape(np.sum(Gradient.grad_a,1), [m,1])
        Gradient.grad_c = np.reshape(np.sum(Gradient.grad_o,0), [-1,1])
        Gradient.grad_o = Gradient.grad_o.T
        return Gradient

    def Numerical_Gradient(self,Parameters,X,Y):
        # h0 = Parameters.h0
        # # print(np.dot(self.W, h0).shape)
        # # print(np.dot(self.U, np.reshape(X[:,0],[-1,1])).shape)
        # # print(self.b.shape)
        def Compute_loss(self, Parameters, Y):
            loss = -np.sum(Y * np.log(Parameters.p))
            return loss

        def Compute_P(self, Parameters, X):
            def softmax(x):
                y = np.exp(x)/np.sum(np.exp(x), axis=0)
                return y
            for i in range(Parameters.seq_len):
                if i == 0:
                    h0 = Parameters.h0
                    Parameters.a[:,i] = np.reshape(np.reshape(np.dot(self.W, h0),[-1,1]) + np.dot(self.U, np.reshape(X[:,i],[-1,1])) + self.b, m)
                    Parameters.h[:,i] = np.tanh(Parameters.a[:,i])
                    Parameters.o[:,i] = np.reshape(np.dot(self.V, np.reshape(Parameters.h[:,i],[-1,1])) + self.C, d)
                    Parameters.p[:,i] = softmax(Parameters.o[:,i])
                else:
                    Parameters.a[:,i] = np.reshape(np.dot(self.W, np.reshape(Parameters.h[:, i-1],[-1,1])) + np.dot(self.U, np.reshape(X[:,i],[-1,1])) + self.b, m)
                    Parameters.h[:,i] = np.tanh(Parameters.a[:,i])
                    Parameters.o[:,i] = np.reshape(np.dot(self.V, np.reshape(Parameters.h[:,i],[-1,1])) + self.C, d)
                    Parameters.p[:,i] = softmax(Parameters.o[:,i])
            Parameters.h0 = Parameters.h[:,Parameters.seq_len-1]
            return Parameters

        grad_V = np.zeros(np.shape(self.V))
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                self_try = self
                self_try.V[i][j] = self.V[i][j]-0.000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_1 = Compute_loss(self_try, Parameters, Y)
                self.V[i][j] += 0.000001
                self_try.V[i][j] = self.V[i][j] + 0.000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_2 = Compute_loss(self_try, Parameters, Y)
                grad_V[i][j] = (loss_2 - loss_1) / 0.000002

        return grad_V



def train(Changable, Parameters, data, dictionary, inverse_dictionary):

    train_step_in_a_epoch = int(len(list(data))/Parameters.seq_len)
    smooth_loss = 0
    SMOOTH_LOSS = []
    for epoch in range(3):
        start = 0
        for i in range(train_step_in_a_epoch):
            if start + Parameters.seq_len + 1 > len(list(data)):
                start = len(list(data)) - Parameters.seq_len - 1
            X, Y, d = one_hot(dictionary, data, start, Parameters)
            start = start + Parameters.seq_len
            Parameters = Changable.Compute_P(Parameters, X)
            loss = Changable.Compute_loss(Parameters, Y)


            if i == 0 and epoch == 0:
                smooth_loss = loss
            smooth_loss = 0.999*smooth_loss + 0.001*loss
            if i % 500 == 0:
                print("EPOCH : ", epoch, "STEP :", i)
                print(smooth_loss)
                SMOOTH_LOSS.append(smooth_loss)
            Gradient = Gradient_class(Changable, Parameters)
            Gradient = Changable.Compute_Gradient(Gradient, Parameters, Changable, X, Y)
            if i % 10000 == 0:
                Text = []
                Character = []
                X_test = np.zeros([d,1])
                X_test[1] = 1
                Text.append(X_test)
                for j in range(1000):
                    prediction = Changable.Compute_prediction(Parameters, inverse_dictionary, Text[j])
                    Text.append(prediction)
                for element in range(len(Text)):
                    pos = np.where(Text[element]==1)[0]
                    character = inverse_dictionary[str(int(pos))]
                    Character.append(character)
                output = ''.join(Character)
                print(output)

            if i == 0:
                W_before = 0
                V_before = 0
                U_before = 0
                b_before = 0
                C_before = 0
            else:
                W_before = W_after
                V_before = V_after
                U_before = U_after
                b_before = b_after
                C_before = C_after
            Changable, W_after, V_after, U_after, b_after, C_after  = Gradient.AdamGradOptimizer(Changable,Parameters,W_before, V_before, U_before, b_before, C_before)
    return Changable, SMOOTH_LOSS

dictionary, inverse_dictionary = give_dictionary(data)
print((dictionary))
Parameters = Inter_parameters(m,d)
W,b,U,V,C = initialization(m,d)
Changable = Changable_parameters(W,b,U,V,C)

Changable, SMOOTH_LOSS =  train(Changable, Parameters, data, dictionary, inverse_dictionary)
para_dic = {}
X_axis = range(0, 500*len(SMOOTH_LOSS), 500)

plt.figure()

plt.xlabel("epoch")
plt.ylabel("smooth loss")
plt.plot(X_axis,SMOOTH_LOSS,'r')
plt.savefig('smooth_loss.png')
plt.legend()

plt.show()

# prediction = Changable.Compute_prediction(Parameters)




# grad_V = Changable.Numerical_Gradient(Parameters,X ,Y)
# print("grad_V:",(grad_V - Gradient.grad_V)/grad_V)


# print(vars(prediction))
