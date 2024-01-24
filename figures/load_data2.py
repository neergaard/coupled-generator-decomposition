import torch
import numpy as np

def load_data(data_pool='all',type='mmmsmc',preproc='FT_frob',as_numpy_array=False):
    print('Loading data...')
    modality_names = ["EEG", "MEG"]
    num_subs = 16

    X_train = X_train1 = X_train2 = {}
    X_test = X_test1 = X_test2 = {}
    for m in modality_names:
        if preproc=='FT_frob' or preproc=='split':
            data_train = torch.load("/dtu-compute/macaroni/projects/CGD/data/concatenatedData/X_" + m + "_" + preproc + "0.pt")
            data_test = torch.load("/dtu-compute/macaroni/projects/CGD/data/concatenatedData/X_" + m + "_" + preproc + "1.pt")
        elif preproc=='FT_l2':
            data_train = torch.load("/dtu-compute/macaroni/projects/CGD/data/concatenatedData/Xf_" + m + "_" + preproc + "0.pt")
            data_test = torch.load("/dtu-compute/macaroni/projects/CGD/data/concatenatedData/Xf_" + m + "_" + preproc + "1.pt")

        if data_pool == 'all':
            if type == 'mmmsmc':
                X_train = data_train
                X_test = data_test
            elif type == 'mmms':
                X_train[m] = torch.cat((data_train[:, 0], data_train[:, 1],data_train[:, 2],),dim=-1)
                X_test[m] = torch.cat((data_test[:, 0], data_test[:, 1],data_test[:, 2],),dim=-1)
            elif type == 'mm' or type == 'group':
                X_train[m] = torch.reshape(torch.cat((data_train[:, 0], data_train[:, 1],data_train[:, 2],),dim=-1),(num_subs*data_train.shape[-2],540))
                X_test[m] = torch.reshape(torch.cat((data_test[:, 0], data_test[:, 1],data_test[:, 2],),dim=-1),(num_subs*data_test.shape[-2],540))
        elif data_pool == 'half':
            if type == 'mmmsmc':
                X_train1[m] = data_train[:8]
                X_test1[m] = data_test[:8]
                X_train2[m] = data_train[8:]
                X_test2[m] = data_test[8:]
            elif type == 'mmms':
                X_train1[m] = torch.cat((data_train[:8, 0], data_train[:8, 1],data_train[:8, 2],),dim=-1)
                X_test1[m] = torch.cat((data_test[:8, 0], data_test[:8, 1],data_test[:8, 2],),dim=-1)
                X_train2[m] = torch.cat((data_train[8:, 0], data_train[8:, 1],data_train[8:, 2],),dim=-1)
                X_test2[m] = torch.cat((data_test[8:, 0], data_test[8:, 1],data_test[8:, 2],),dim=-1)
            elif type == 'mm' or type == 'group':
                X_train1[m] = torch.reshape(torch.cat((data_train[:8, 0], data_train[:8, 1],data_train[:8, 2],),dim=-1),(8*data_train.shape[-2],540))
                X_test1[m] = torch.reshape(torch.cat((data_test[:8, 0], data_test[:8, 1],data_test[:8, 2],),dim=-1),(8*data_test.shape[-2],540))
                X_train2[m] = torch.reshape(torch.cat((data_train[8:, 0], data_train[8:, 1],data_train[8:, 2],),dim=-1),(8*data_train.shape[-2],540))
                X_test2[m] = torch.reshape(torch.cat((data_test[8:, 0], data_test[8:, 1],data_test[8:, 2],),dim=-1),(8*data_test.shape[-2],540))
    if type == 'group':
        if data_pool == 'all':
            X_train['all'] = torch.cat((X_train['EEG'],X_train['MEG']),dim=-2)
            X_test['all'] = torch.cat((X_test['EEG'],X_test['MEG']),dim=-2)
            X_train.pop('EEG')
            X_train.pop('MEG')
            X_test.pop('EEG')
            X_test.pop('MEG')
        elif data_pool == 'half':
            X_train1['all'] = torch.cat((X_train1['EEG'],X_train1['MEG']),dim=-2)
            X_test1['all'] = torch.cat((X_test1['EEG'],X_test1['MEG']),dim=-2)
            X_train2['all'] = torch.cat((X_train2['EEG'],X_train2['MEG']),dim=-2)
            X_test2['all'] = torch.cat((X_test2['EEG'],X_test2['MEG']),dim=-2)
            X_train1.pop('EEG')
            X_train1.pop('MEG')
            X_train2.pop('EEG')
            X_train2.pop('MEG')
            X_test1.pop('EEG')
            X_test1.pop('MEG')
            X_test2.pop('EEG')
            X_test2.pop('MEG')
    if as_numpy_array:
        if data_pool == 'all':
            for key in X_train.keys():
                X_train[key] = np.array(X_train[key])
                X_test[key] = np.array(X_test[key])
        elif data_pool == 'half':
            for key in X_train1.keys():
                X_train1[key] = np.array(X_train1[key])
                X_train2[key] = np.array(X_train2[key])
                X_test1[key] = np.array(X_test1[key])
                X_test2[key] = np.array(X_test2[key])
    if data_pool == 'all':
        return X_train,X_test
    elif data_pool == 'half':
        return X_train1,X_train2,X_test1,X_test2