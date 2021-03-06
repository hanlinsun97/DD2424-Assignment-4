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
data = data[0:20000]
data = "FROM DONALD TRUMP"
m = 150
d = 12
MAX_EPOCH = 100
print(data)

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

def one_hot_label(dictionary, data, Parameters):  # Will not give the whole one-hoted list 80 * 1100000!
    len_dictionary = len(dictionary)
    one_hot_label_x = np.zeros([len_dictionary, len(list(data))-1])
    one_hot_label_y = np.zeros([len_dictionary, len(list(data))-1])
    data = list(data)
    for i in range(len(list(data))-1):
        value_x = dictionary[data[i]]
        one_hot_label_x[int(value_x), i] = 1
        value = dictionary[data[i+1]]
        one_hot_label_y[int(value), i] = 1
    return one_hot_label_x,  one_hot_label_y

def one_hot(one_hot_label_x, one_hot_label_y, start, Parameters):
    if start + Parameters.seq_len > np.shape(one_hot_label_x)[1]:
        start = np.shape(one_hot_label_x) - Parameters.seq_len
    one_hot_label_x_part = one_hot_label_x[:,start:start + Parameters.seq_len]
    one_hot_label_y_part = one_hot_label_y[:,start:start + Parameters.seq_len]
    return one_hot_label_x_part, one_hot_label_y_part

# define initial parameters
def initialization(m,d):
    W = np.random.normal(0,1,[m,m]) * 0.01
    #W = np.random.rand(m,m) * 0.01
    b = np.zeros([m,1])
    U = np.random.normal(0,1,[m,d]) * 0.01
    #U = np.random.rand(m,d) * 0.01
    V = np.random.normal(0,1,[d,m]) * 0.01
    #V = np.random.rand(d,m) * 0.01
    C = np.zeros([d,1])
    return W, b, U, V, C

class Inter_parameters(object):
    def __init__(self,m,d):
        self.initial_learning_rate = 0.1
        self.seq_len = 2 # MAX TIME STEP
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
        self.grad_h = 0 #np.zeros(np.shape(Parameters.h))
        self.grad_a = 0 #np.zeros(np.shape(Parameters.a))
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
        Parameters.h0 = np.reshape(Parameters.h[:,Parameters.seq_len - 1], np.shape(Parameters.h0))
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
                grad_o = np.reshape(Gradient.grad_o[i,:], [1,d])
                x = np.reshape(X[:,i], [1,d])

                Gradient.grad_h = np.dot(grad_o, Changable.V)
                Gradient.grad_a = Gradient.grad_h * (1 - np.power(np.tanh(Parameters.a[:,i]),2))
                Gradient.grad_b = Gradient.grad_b + Gradient.grad_a.T
                Gradient.grad_U = Gradient.grad_U + np.dot(Gradient.grad_a.T, x)
            else:
                grad_o = np.reshape(Gradient.grad_o[i,:], [1,d])
                x = np.reshape(X[:,i], [1,d])
                Gradient.grad_h = np.dot(grad_o, Changable.V) + np.dot(Gradient.grad_a,Changable.W)

                Gradient.grad_W = Gradient.grad_W + np.dot(Gradient.grad_a.T ,np.reshape(Parameters.h[:,i], [1,m]))

                Gradient.grad_a = Gradient.grad_h * (1 - np.power(np.tanh(Parameters.a[:,i]),2))
                Gradient.grad_b = Gradient.grad_b + Gradient.grad_a.T
                Gradient.grad_U = Gradient.grad_U + np.dot(Gradient.grad_a.T, x)

        Gradient.grad_W = Gradient.grad_W + np.dot(Gradient.grad_a.T, Parameters.h0.T)
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
            return Parameters

        grad_W = np.zeros(np.shape(self.W))
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self_try = self
                self_try.W[i][j] = self.W[i][j]-0.00000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_1 = Compute_loss(self_try, Parameters, Y)
                self.W[i][j] += 0.00000001
                self_try.W[i][j] = self.W[i][j] + 0.00000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_2 = Compute_loss(self_try, Parameters, Y)
                grad_W[i][j] = (loss_2 - loss_1) / 0.00000002

        grad_V = np.zeros(np.shape(self.V))
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                self_try = self
                self_try.V[i][j] = self.V[i][j]-0.00000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_1 = Compute_loss(self_try, Parameters, Y)
                self.V[i][j] += 0.000001
                self_try.V[i][j] = self.V[i][j] + 0.000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_2 = Compute_loss(self_try, Parameters, Y)
                grad_V[i][j] = (loss_2 - loss_1) / 0.000002

        grad_U = np.zeros(np.shape(self.U))
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self_try = self
                self_try.U[i][j] = self.U[i][j]-0.000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_1 = Compute_loss(self_try, Parameters, Y)
                self.U[i][j] += 0.000001
                self_try.U[i][j] = self.U[i][j] + 0.000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_2 = Compute_loss(self_try, Parameters, Y)
                grad_U[i][j] = (loss_2 - loss_1) / 0.000002

        grad_b = np.zeros(np.shape(self.b))
        for i in range(self.b.shape[0]):
            for j in range(self.b.shape[1]):
                self_try = self
                self_try.b[i][j] = self.b[i][j]-0.000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_1 = Compute_loss(self_try, Parameters, Y)
                self.b[i][j] += 0.000001
                self_try.b[i][j] = self.b[i][j] + 0.000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_2 = Compute_loss(self_try, Parameters, Y)
                grad_b[i][j] = (loss_2 - loss_1) / 0.000002

        grad_c = np.zeros(np.shape(self.C))
        for i in range(self.C.shape[0]):
            for j in range(self.C.shape[1]):
                self_try = self
                self_try.C[i][j] = self.C[i][j]-0.000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_1 = Compute_loss(self_try, Parameters, Y)
                self.C[i][j] += 0.000001
                self_try.C[i][j] = self.C[i][j] + 0.000001
                Parameters = Compute_P(self_try, Parameters, X)
                loss_2 = Compute_loss(self_try, Parameters, Y)
                grad_c[i][j] = (loss_2 - loss_1) / 0.000002
        return grad_W, grad_V, grad_U, grad_b, grad_c



def train(Changable, Parameters, data, dictionary, inverse_dictionary):
    one_hot_label_x, one_hot_label_y = one_hot_label(dictionary, data, Parameters)
    train_step_in_a_epoch = int(len(list(data))/Parameters.seq_len)
    smooth_loss = 0
    SMOOTH_LOSS = []

    for epoch in range(10000):
        start = 0
        Parameters.h0 = np.zeros(Parameters.h0.shape)
        for i in range(train_step_in_a_epoch):

            if start + Parameters.seq_len + 1 > len(list(data)):
                start = len(list(data)) - Parameters.seq_len - 1

            X, Y = one_hot(one_hot_label_x, one_hot_label_y, start, Parameters)
            Out = []
            for k in range(np.shape(Y)[1]):
                y = Y[:,k]
                ps = np.where(y == 1)[0]
                s = str(int(ps))
                output = inverse_dictionary[s]
                Out.append(output)
            print(Out)
            start = start + Parameters.seq_len
            Parameters = Changable.Compute_P(Parameters, X)
            loss = Changable.Compute_loss(Parameters, Y)

            if i == 0 and epoch == 0:
                smooth_loss = loss
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss
            if i % 200 == 0:
                print("EPOCH : ", epoch, "STEP :", i)
                print(smooth_loss)
                SMOOTH_LOSS.append(smooth_loss)

            Gradient = Gradient_class(Changable, Parameters)
            Gradient = Changable.Compute_Gradient(Gradient, Parameters, Changable, X, Y)
            #
            # grad_W, grad_V, grad_U, grad_b, grad_c = Changable.Numerical_Gradient(Parameters, X, Y)
            # grad_W_an = Gradient.grad_W
            # grad_U_an = Gradient.grad_U
            # grad_V_an = Gradient.grad_V
            # grad_b_an = Gradient.grad_b
            # grad_c_an = Gradient.grad_c
            # # print(grad_U)
            # # print(grad_U_an)
            # print((grad_b -  grad_b_an)/grad_b)
            # print((grad_c -  grad_c_an)/grad_c)
            # print((grad_U[0] - grad_U_an[0]) / grad_U[0])
            # print((grad_W[0] - grad_W_an[0]) / grad_W[0])
            # print((grad_V[0] - grad_V_an[0]) / grad_V[0])
            # exit()

            if i % 10000 == 0:
                Text = []
                Character = []
                X_test = np.zeros([d,1])
                a = int(np.random.rand() * 11)
                X_test[a] = 1
                Text.append(X_test)
                for j in range(15):
                    prediction = Changable.Compute_prediction(Parameters, inverse_dictionary, Text[j])
                    Text.append(prediction)
                for element in range(len(Text)):
                    pos = np.where(Text[element]==1)[0]
                    character = inverse_dictionary[str(int(pos))]
                    Character.append(character)
                output = ''.join(Character)
                print(output + '  ' + 'EndOfTweet')

            if i == 0 and epoch == 0:
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
            # Parameters.h0 = np.reshape(h00,[m,1])
            # print(Parameters.h0)
    return Changable, SMOOTH_LOSS



dictionary, inverse_dictionary = give_dictionary(data)
Parameters = Inter_parameters(m,d)
W,b,U,V,C = initialization(m,d)
Changable = Changable_parameters(W,b,U,V,C)
Changable, SMOOTH_LOSS =  train(Changable, Parameters, data, dictionary, inverse_dictionary)
para_dic = {}
X_axis = range(0, 200*len(SMOOTH_LOSS), 200)
plt.figure()
plt.xlabel("epoch")
plt.ylabel("smooth loss")
plt.plot(X_axis,SMOOTH_LOSS,'r')
plt.savefig('smooth_loss.png')
plt.legend()
plt.show()
