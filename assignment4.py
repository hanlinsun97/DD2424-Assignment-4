import numpy as np

# IMPORT DATA IS D * 1
with open('goblet_book.txt','r') as f:
    data = f.read()


m = 100
d = 80

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

def numerical_gradient(Changable, Parameters, Y, X):
    loss = Changable.Compute_loss(Parameters, Y)


    for i in range(np.shape(Changable.W)[0]):
        for j in range(np.shape(Changable.W)[1]):
            Parameters_try = Parameters
            Changable_try = Changable
            Changable_try.W[i][j] = Changable_try.W[i][j] + 0.000001
            Parameters_try = Changable_try.Compute_P(Parameters_try,X)
            loss_try = Changable_try.Compute_loss(Parameters_try, Y)
            grad_W[i][j] = (loss - loss_try)/0.000001
    return grad_W

class Inter_parameters(object):
    def __init__(self,m,d):
        self.initial_learning_rate = 0.01
        self.regularization = 0.001
        self.seq_len = 25 # MAX TIME STEP
        self.h = np.zeros([m,self.seq_len])
        self.a = np.zeros([m,self.seq_len])
        self.o = np.zeros([d,self.seq_len])
        self.p = np.zeros([d,self.seq_len])
        self.h0 = np.zeros([m,1])

class Gradient(object):

    def __init__(self, Changable, Parameters):
        self.grad_W = np.zeros(np.shape(Changable.W))
        self.grad_b = np.zeros(np.shape(Changable.b))
        self.grad_U = np.zeros(np.shape(Changable.U))
        self.grad_V = np.zeros(np.shape(Changable.V))
        self.grad_C = np.zeros(np.shape(Changable.C))
        self.grad_h = np.zeros(np.shape(Parameters.h))
        self.grad_a = np.zeros(np.shape(Parameters.a))
        self.grad_o = np.zeros(np.shape(Parameters.o))


    def AdamGradOptimizer(self,Changable, Parameters):
        dictionary_grad = {}
        dictionary_grad['grad_W'] = Changable.W
        dictionary_grad['grad_b'] = Changable.b
        dictionary_grad['grad_U'] = Changable.U
        dictionary_grad['grad_V'] = Changable.V
        dictionary_grad['grad_C'] = Changable.C
        dictionary_grad['grad_h'] = Parameters.h
        dictionary_grad['grad_a'] = Parameters.a
        dictionary_grad['grad_o'] = Parameters.o

        for gradient in vars(self).items():
            m_inter_init = 0
            m_inter = np.zeros(Parameters.seq_len).tolist()
            variable = dictionary_grad[gradient[0]]
            if gradient[0] in ['grad_h', 'grad_a', 'grad_o']:
                continue
            elif gradient[0] in ['grad_b', 'grad_C']:
                continue                                                                 #TODO
            else:
                for i in range(Parameters.seq_len - 1):
                        if i == 0 :
                            m_inter[i] = m_inter_init + np.power(gradient[1][:,i],2)
                            variable[:,i+1] = variable[:,i] - (Parameters.initial_learning_rate / np.sqrt(m_inter[i] + 0.000001)) * gradient[1][:,i]
                        else:
                            m_inter[i] = m_inter[i-1] + np.power(gradient[1][:,i],2)
                            variable[:,i+1] = variable[:,i] - (Parameters.initial_learning_rate / np.sqrt(m_inter[i] + 0.000001)) * gradient[1][:,i]
        return Changable


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
                Parameters.a[:,i] = np.reshape(np.dot(self.W, h0) + np.dot(self.U, np.reshape(X[:,i],[-1,1])) + self.b, m)
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

    def Compute_prediction(self,Parameters):
        prediction = np.zeros([d, Parameters.seq_len])
        extraction = np.max(Parameters.p,0)
        for i in range(np.size(extraction)):
            position = np.where(Parameters.p == extraction[i])[0]
            prediction[position, i] = 1
        return prediction

    def accuracy(self,Parameters,prediction, Y):
        num = np.sum(np.power(prediction - Y, 2))
        accuracy = Parameters.seq_len - num / Parameters.seq_len
        return accuracy

    def Compute_Gradient(self, Gradient, Parameters, Changable, X, Y):
        Gradient.grad_o = (Parameters.p - Y).T
        Gradient.grad_V = np.dot(Gradient.grad_o.T, Parameters.h.T)
        print(Gradient.grad_o.shape)
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

        Gradient.grad_W = np.dot(Gradient.grad_a.T, H_inter)
        Gradient.grad_U = np.dot(Gradient.grad_a, X.T)
        Gradient.grad_b = np.reshape(np.sum(Gradient.grad_a,1), [m,1])
        Gradient.grad_c = np.sum(Gradient.grad_o,1)  #BUG
        print(Gradient.grad_c.shape)

        return Gradient



dictionary,_ = give_dictionary(data)
Parameters = Inter_parameters(m,d)
X, Y, d = one_hot(dictionary, data, 0, Parameters)

Parameters = Inter_parameters(m,d)
W,b,U,V,C = initialization(m,d)
Changable = Changable_parameters(W,b,U,V,C)
Parameters = Changable.Compute_P(Parameters, X)
prediction = Changable.Compute_prediction(Parameters)

Gradient = Gradient(Changable, Parameters)
Gradient = Changable.Compute_Gradient(Gradient, Parameters, Changable, X, Y)
#grad_W = numerical_gradient(Changable, Parameters, Y,X)
print(Gradient.grad_b.shape)
print(grad_W)
# Changable = Gradient.AdamGradOptimizer(Changable,Parameters)
# print(vars(prediction))
