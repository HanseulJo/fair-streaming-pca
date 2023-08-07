"""
Created on 2021/08/25
@author: nicklee
@modified-by: Hanseul Cho

(Description)
"""
import numpy as np
from aif360.datasets import AdultDataset



adult = AdultDataset(
    protected_attribute_names=['sex'],
    privileged_classes=[['Male']],
    # features_to_drop=['race'],
    categorical_features=['workclass', 'education',
                     'marital-status', 'occupation', 'relationship',
                     'race','native-country'],
)

# 10 different 70-30 train-test splits
for i in range(10):
    train, test = adult.split([0.7], shuffle=True, seed=i)
    
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
           np.concatenate((adult.features, adult.labels.astype(int), adult.protected_attributes.astype(int)), axis=1),
           fmt=",".join(["%.18e"]*test.features.shape[-1] + ["%d","%d"])
          )
