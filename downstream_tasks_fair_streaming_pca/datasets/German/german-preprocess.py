"""
Created on 2021/08/25
@author: nicklee

(Description)
"""
import numpy as np
from aif360.datasets import GermanDataset

# german = load_preproc_data_german(protected_attributes=['age'])
german = GermanDataset(
    protected_attribute_names=['age'],
    privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
    # features_to_drop=['sex', 'personal_status'],
    categorical_features=['status', 'credit_history', 'purpose',
                     'savings', 'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone',
                     'foreign_worker', 'sex'],

)

# 10 different 70-30 train-test splits
for i in range(10):
    train, test = german.split([0.7], shuffle=True, seed=i)

    # normalize each feature, using train!
    train_mean = np.mean(train.features, axis=0)
    train_std = np.std(train.features, axis=0)
    train_std[train_std == 0] = 1

    train.features = (train.features - train_mean) / train_std
    test.features = (test.features - train_mean) / train_std

    np.savetxt(f'train_{i}.csv',
               np.concatenate((train.features, train.labels.astype(int)-1, train.protected_attributes.astype(int)), axis=1),
               fmt=",".join(["%.18e"]*train.features.shape[-1] + ["%d","%d"])
               )
    np.savetxt(f'test_{i}.csv',
               np.concatenate((test.features, test.labels.astype(int)-1, test.protected_attributes.astype(int)), axis=1),
               fmt=",".join(["%.18e"]*test.features.shape[-1] + ["%d","%d"])
               )

np.savetxt('full.csv',
           np.concatenate((german.features, german.labels.astype(int)-1, german.protected_attributes.astype(int)), axis=1),
           fmt=",".join(["%.18e"]*test.features.shape[-1] + ["%d","%d"])
          )
