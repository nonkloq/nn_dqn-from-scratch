import numpy as np
import os
import json
import pickle


def save_network(network,netname):
    NN = network.NN
    if not os.path.exists(netname): os.mkdir(netname)
    meta = []
    i = 1
    for W,b in NN:
        wn = os.path.join(netname, f"Wl{i}.npy")
        bn = os.path.join(netname, f"bl{i}.npy")
        np.save(wn,W)
        np.save(bn,b)
        i+=1
        meta.append((wn,bn))

    jn = os.path.join(netname, "netstruct.json")
    with open(jn, "w") as f: json.dump(meta,f)

    cn = os.path.join(netname, f"{netname}.pkl")
    with open(cn, "wb") as f: pickle.dump(network, f)


def load_network(netname):
    cn = os.path.join(netname, f"{netname}.pkl")
    with open(cn, "rb") as f: loaded_network = pickle.load(f)
    jn = os.path.join(netname, "netstruct.json")
    with open(jn, "r") as f: meta= json.load(f)
    
    for i in range(loaded_network.L):
        wn,bn = meta[i]
        pW,pb =loaded_network.NN[i]
        del pW
        del pb
        
        loaded_network.NN[i] = (np.load(wn),np.load(bn))

    return loaded_network


if __name__ == "__main__":
    from nn import NeuralNetwork

    test_net = NeuralNetwork([2,3,1],["sigmoid","sigmoid"],eta=1)
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    test_net.train(X,y,100)
    y_pred = [round(y[0]) for y in test_net.predict(X)]
    print("test_net:",y_pred)

    save_network(test_net,"XOR")
    andv1 = load_network("XOR")
    
    y_pred = [round(y[0]) for y in andv1.predict(X)]
    print("andv1:",y_pred)
    print(f"Loaded class main attributes:\nArch: {andv1.arch}\nAlpha: {andv1.alpha}\nAF: {andv1.af}\neta: {andv1.eta}")

    for i in range(andv1.L):
        W1,b1 = test_net.NN[i]
        W2,b2 = andv1.NN[i]
        print(i+1,":",W1 == W2,b1 == b2)