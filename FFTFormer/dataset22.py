import pickle
import numpy as np
from numpy import linalg as la

def normalize_iq(X):
    n1=int(X.shape[0])
    n2=int(X.shape[1])
    X_norm = np.zeros_like(X)
    for i in range(n1):
        for j in range(n2):
            norm_num=1/la.norm(X[i,j,:],ord=np.inf)
            X_norm[i,j,:]=X[i,j,:]*norm_num
    return X_norm

def load_data(filename=r'/home/sp432cy/sp432cy/AMR/RML22/RML22.01A.pkl', factor_label=1):

    X, lbl = [], []
    train_idx, val_idx = [], []
    np.random.seed(2022)
    a=0
    zero_label_mods  = ['AM-DSB', 'AM-SSB', 'GFSK', 'CPFSK', 'WBFM']

    data = pickle.load(open(filename,'rb'),encoding='iso-8859-1')
    mods,snrs = [sorted(list(set([k[j] for k in data.keys()]))) for j in [0,1] ]
    for mod in mods:
        for snr in snrs:
            X.append(data[(mod,snr)])  
            for i in range(data[(mod,snr)].shape[0]):
                    if mod in zero_label_mods:
                        lbl.append((mod, snr, 0))  
                    else:
                        lbl.append((mod, snr, factor_label))
            train_idx+=list(np.random.choice(range(a*500,(a+1)*500), size=300, replace=False))
            val_idx+=list(np.random.choice(list(set(range(a*500,(a+1)*500))-set(train_idx)), size=100, replace=False))
            a+=1
    X = np.vstack(X)
    n_examples = X.shape[0]
    test_idx = list(set(range(0,n_examples))-set(train_idx)-set(val_idx))

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]
   
    factors = [0,1,2,3,4]

    def to_onehot1(yy):
        yy1 = np.zeros([len(yy), len(mods)])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1
    
    def to_onehot2(zz):
        zz1 = np.zeros([len(zz), len(factors)])
        zz1[np.arange(len(zz)), [factors.index(z) for z in zz]] = 1
        return zz1
    
    Y_train = to_onehot1(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_val=to_onehot1(list(map(lambda x: mods.index(lbl[x][0]), val_idx)))
    Y_test = to_onehot1(list(map(lambda x: mods.index(lbl[x][0]),test_idx)))

    Z_train = to_onehot2(list(map(lambda x: lbl[x][2], train_idx)))
    Z_val=to_onehot2(list(map(lambda x: lbl[x][2], val_idx)))
    Z_test = to_onehot2(list(map(lambda x: lbl[x][2], test_idx)))

    X_train = normalize_iq(X_train)
    X_val = normalize_iq(X_val)
    X_test = normalize_iq(X_test)

    X_train = X_train.swapaxes(2, 1)
    X_val = X_val.swapaxes(2, 1)
    X_test = X_test.swapaxes(2, 1)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_val.shape)
    print(Y_test.shape)
    print(Z_train.shape)
    print(Z_val.shape)
    print(Z_test.shape)

    return (mods,snrs,lbl),(X_train,Y_train,Z_train),(X_val,Y_val,Z_val),(X_test,Y_test,Z_test),(train_idx,val_idx,test_idx), X.shape[0]

def load_all_data():

    files = {
        1: r'/home/sp604cy/sp604cy/AMR/RML22/RML22_0.20_500_dict.pkl',
        2: r'/home/sp604cy/sp604cy/AMR/RML22/RML22_0.35_500_dict.pkl',
        3: r'/home/sp604cy/sp604cy/AMR/RML22/RML22_0.50_500_dict.pkl',
        4: r'/home/sp604cy/sp604cy/AMR/RML22/RML22_0.65_500_dict.pkl',
    }

    all_X_train, all_Y_train, all_Z_train  = [], [], []
    all_X_val, all_Y_val, all_Z_val= [], [], []
    all_X_test, all_Y_test, all_Z_test = [], [], []
    all_lbl = []
    all_train_idx, all_val_idx, all_test_idx = [], [], []

    offset = 0
    for factor_label, filename in files.items():
        print(f"Loading {filename}")
        (mods,snrs,lbl),(X_train,Y_train,Z_train),(X_val,Y_val,Z_val),(X_test,Y_test,Z_test),(train_idx,val_idx,test_idx), n= load_data(filename, factor_label)

        all_X_train.append(X_train)
        all_Y_train.append(Y_train)
        all_Z_train.append(Z_train)
        all_X_val.append(X_val)
        all_Y_val.append(Y_val)
        all_Z_val.append(Z_val)
        all_X_test.append(X_test)
        all_Y_test.append(Y_test)
        all_Z_test.append(Z_test)

        all_train_idx += [i + offset for i in train_idx]
        all_val_idx += [i + offset for i in val_idx]
        all_test_idx += [i + offset for i in test_idx]
        all_lbl += lbl

        offset += n

    X_train = np.concatenate(all_X_train, axis=0)
    Y_train = np.concatenate(all_Y_train, axis=0)
    Z_train = np.concatenate(all_Z_train, axis=0)
    X_val = np.concatenate(all_X_val, axis=0)
    Y_val = np.concatenate(all_Y_val, axis=0)
    Z_val = np.concatenate(all_Z_val, axis=0)
    X_test = np.concatenate(all_X_test, axis=0)
    Y_test = np.concatenate(all_Y_test, axis=0)
    Z_test = np.concatenate(all_Z_test, axis=0)

    print("Shapes:")
    print(f"  X_train: {X_train.shape}, Y_train: {Y_train.shape}, Z_train: {Z_train.shape}")
    print(f"  X_val:   {X_val.shape}, Y_val:   {Y_val.shape}, Z_val:   {Z_val.shape}")
    print(f"  X_test:  {X_test.shape}, Y_test:  {Y_test.shape}, Z_test:  {Z_test.shape}")

    return (mods, snrs, all_lbl), (X_train, Y_train,  Z_train), (X_val, Y_val, Z_val), (X_test, Y_test, Z_test), (all_train_idx, all_val_idx, all_test_idx)

if __name__ == '__main__':
    (mods, snrs, all_lbl), (X_train, Y_train,  Z_train), (X_val, Y_val, Z_val), (X_test, Y_test, Z_test), (all_train_idx, all_val_idx, all_test_idx) = load_all_data()