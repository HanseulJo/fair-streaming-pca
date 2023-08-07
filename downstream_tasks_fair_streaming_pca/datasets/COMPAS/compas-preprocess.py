"""
Created on 2021/08/25
@author: nicklee

(Description)
"""
import numpy as np
from aif360.datasets import CompasDataset

compas = CompasDataset(
    protected_attribute_names=['race'],
    privileged_classes=[['Caucasian']],
    # features_to_drop=['sex', 'c_charge_desc'],
    categorical_features=['age_cat', 'c_charge_degree',
                     'c_charge_desc', 'sex'],
)

# 10 different 70-30 train-test splits
for i in range(10):
    train, test = compas.split([0.7], shuffle=True, seed=i)

    # normalize each feature, using train!
    train_mean = np.mean(train.features, axis=0)
    train_std = np.std(train.features, axis=0)
    train_std[train_std == 0] = 1

    train.features = (train.features - train_mean) / train_std
    test.features = (test.features - train_mean) / train_std

    np.savetxt(f'train_{i}.csv',
               np.concatenate((train.features, train.labels.astype(int), train.protected_attributes.astype(int)), axis=1),
               fmt=",".join(["%.18e"]*train.features.shape[-1] + ["%d","%d"])
               )
    np.savetxt(f'test_{i}.csv',
               np.concatenate((test.features, test.labels.astype(int), test.protected_attributes.astype(int)), axis=1),
               fmt=",".join(["%.18e"]*test.features.shape[-1] + ["%d","%d"])
               )

np.savetxt('full.csv',
           np.concatenate((compas.features, compas.labels.astype(int), compas.protected_attributes.astype(int)), axis=1),
           fmt=",".join(["%.18e"]*test.features.shape[-1] + ["%d","%d"])
          )