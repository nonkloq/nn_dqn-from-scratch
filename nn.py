import numpy as np


class sigmoid:
    def __init__(self,beta=1):
        self.beta = beta
    
    def __call__(self,z,derivative=False):
    
        if derivative: return self.beta*z * (1 - z)
        return 1 / (1 + np.exp(-z*self.beta))

class relu:
    def __call__(self,z,derivative=False):
        if derivative: (z>0)*1
        return np.where(z>0,z,0)

class linear:
    def __call__(self,z,derivative=False):
        if derivative: return 1
        return z




AF_MAP = {
    "sigmoid":sigmoid(1),
    "linear":linear(),
    "relu":relu(),
}

def neuron_sum(x,W,b):
    return W@x +b

def MSE(t,y):
    return np.sum((y-t)**2)/2

class NeuralNetwork:
    def __init__(self,arch,af,eta=0.1,momentum=0.9,seed=None):
        if seed: np.random.seed(seed)
        self.arch = arch
        self.eta = eta
        self.alpha = momentum
        self.NN = [ (
                np.random.randn(arch[i],arch[i-1]),
                np.random.rand(arch[i])
                    ) for i in range(1,len(arch))]
        
        self.L = len(self.NN)
        # for GD Momentum
        self.prev_deltas = [0]*self.L
        self.prev_dE_dWs = [0]*self.L
        
        self.af = [AF_MAP[a] if isinstance(a,str) else a for a in af ]
        
    def forward(self,x):
        A = [x]
        i = 0
        for w,b in self.NN:
            A.append(
                self.af[i](
                    neuron_sum(A[-1],w,b)
                )
            )
            i+=1
            
        return A
    
    def backward(self,A,t):
        L = self.L
        deltas = [None]* L
        dE_dW = [None] * L
        L-=1
        for i in range(L,-1,-1):
            if i != L:deltas[i] = (self.NN[i+1][0].T @ deltas[i+1]) * self.af[i](A[i+1],True)
            else: deltas[i] = (A[i+1] - t) * self.af[i](A[i+1],True)

            dE_dW[i] = (A[i].reshape(-1,1) * deltas[i]).T
        return deltas, dE_dW
    
    def update_weights(self,deltas,dE_dW):
        for i in range(len(self.NN)):
            W,b = self.NN[i]
            self.prev_dE_dWs[i] = self.eta*dE_dW[i] + self.alpha*self.prev_dE_dWs[i]
            self.prev_deltas[i] = self.eta*deltas[i] + self.alpha*self.prev_deltas[i]
            W -= self.prev_dE_dWs[i]
            b -= self.prev_deltas[i]

            
    def train(self,X,y,epoch=100,errorcal=10):
        mse_l = []
        record = True
        mse = 0
        for i in range(1,epoch+1):
            if i%errorcal == 0: 
                record = True
                mse=0

            for x,t in zip(X,y):
                A = self.forward(x)
                d,gra = self.backward(A,t)
                self.update_weights(d,gra)
                if record: mse += MSE(A[-1],t)
            if record:
                mse_l.append(mse)
                record = False
        return mse_l
        
    def train_on_batch(self,X,y,epoch=100,errorcal=10):
        batch_size = len(X)
        mse_l = []
        record = True
        mse = 0
        for i in range(1,epoch+1):
            if i%errorcal == 0: 
                record = True
                mse=0
            delta_b = [0]*self.L
            dE_dW_b = [0]*self.L
            
            for x,t in zip(X,y):
                A = self.forward(x)
                d,gra = self.backward(A,t)
                for i in range(self.L):
                    delta_b[i] += d[i]
                    dE_dW_b[i] += gra[i]
                if record: mse += MSE(A[-1],t)
            
            for i in range(self.L):
                delta_b[i] /= batch_size
                dE_dW_b[i] /= batch_size
            self.update_weights(delta_b,dE_dW_b)

            if record:
                mse_l.append(mse)
                record = False
        return mse_l
            
    def predict_single(self,x):
        return self.forward(x)[-1]
    
    
    def predict(self,X):
        return np.array([self.forward(x)[-1] for x in X])
    
        
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt 

    bc = load_breast_cancer()
    np.random.seed(69420)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(bc.data,bc.target,test_size=0.3,shuffle=True)
    
    s01 = sigmoid(0.01) # lower beta to avoid overflows in exp
    model_bc = NeuralNetwork([30,12,1],[s01,"linear","sigmoid"],seed=8,eta=0.1,momentum=0.3)
    y_pred = model_bc.predict(X_test)
    y_pred = [np.round(y_) for y_ in y_pred]
    mse_l = model_bc.train(X_train,y_train,epoch=100)
    fig = plt.figure(figsize=(5,5))
    plt.plot(range(len(mse_l)),mse_l)
    plt.xlabel(f"Epoch")
    plt.ylabel("MSE")
    plt.plot()
    y_pred = model_bc.predict(X_test)
    y_pred = [np.round(y_) for y_ in y_pred]
    print(f"Accuracy of the model: {accuracy_score(y_test,y_pred)}")

