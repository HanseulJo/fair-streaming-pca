# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys

file_dir = os.path.abspath('.')
sys.path.append(file_dir)

import csv
import numpy as np
import scipy
from sklearn.svm import LinearSVC

from src.fair_pca.fair_PCA import apply_fair_PCA_to_dataset, \
    apply_fair_PCA_equalize_covariance_to_dataset, apply_fair_kernel_PCA_to_dataset

from fair_streaming_pca import FairPCA

embedding_dim_range = [2]
methods_to_run = [
    # 'Fair PCA',
    # 'Fair PCA-S (0.5)',
    # 'Fair PCA-S (0.85)',
    # 'Fair Kernel PCA',
    # 'INLP',
    # 'RLACE',
    # 'Fair Streaming PCA (Offline): mean, null',
    # 'Fair Streaming PCA (Offline): all, 2',
    # 'Fair Streaming PCA (Offline): all, 5',
    # 'Fair Streaming PCA (Offline): all, 10',
    # 'Fair Streaming PCA (Offline): all, 15',
    # 'Fair Streaming PCA (Offline): all, 25',
    # 'Fair Streaming PCA (Offline): all, 50',
    # 'Fair Streaming PCA (Iterative): mean, null',
    # 'Fair Streaming PCA (Iterative): all, 2',
    # 'Fair Streaming PCA (Iterative): all, 5',
    # 'Fair Streaming PCA (Iterative): all, 10',
    # 'Fair Streaming PCA (Iterative): all, 15',
    # 'Fair Streaming PCA (Iterative): all, 25',
    # 'Fair Streaming PCA (Iterative): all, 50',
]
dataset_names = [
    # 'COMPAS',
    # 'German',
    # 'Adult'
]

for method in methods_to_run:

    print(f'Running {method}')

    for embedding_dim in embedding_dim_range:

        print('Embedding dim = ' + str(embedding_dim))

        for name in dataset_names:
            for split in range(10):
                print('%s, repeat=%d' % (name, split))

                data = np.array([dPoint for dPoint in csv.reader(
                    open('./fair-manifold-pca/datasets/{}/train_{}.csv'.format(name, split),'r'))]).astype(float)
                
                X_train, Y_train, A_train = data[:, :-2], data[:, -2].astype(int), data[:, -1].astype(int)
                
                if method == 'Fair PCA':

                    pipe = apply_fair_PCA_to_dataset(
                        (X_train, Y_train, A_train), embedding_dim, standardize=False, fit_classifier=False)
                    
                    # Save transformation matrix
                    np.savetxt(
                        './fair-manifold-pca/uci/{}/FairPCA_V_{}_dim{}.csv'.format(
                            name, split, embedding_dim),
                        pipe['FairPCA'].transformation_matrix, delimiter=',')

                elif 'Fair PCA-S (' in method:
                    param = float(method.split('(')[1][:-1])  # 0.5, 0.85
                    assert method == f'Fair PCA-S ({param})'

                    pipe = apply_fair_PCA_equalize_covariance_to_dataset(
                        (X_train, Y_train, A_train), embedding_dim, fit_classifier=False, standardize=False,
                        nr_eigenvecs_cov_constraint=int(param * (data.shape[1] - 2)))

                    # Save transformation matrix
                    np.savetxt(
                        './fair-manifold-pca/uci/{}/FairPCA_S'.format(
                            name) + str(param).replace('.', '') + '_V_{}_dim{}.csv'.format(split, embedding_dim),
                        pipe['FairPCA'].transformation_matrix, delimiter=',')

                elif method == 'Fair Kernel PCA':
                    test_data = np.array([dPoint for dPoint in csv.reader(
                        open('./fair-manifold-pca/datasets/{}/test_{}.csv'.format(name, split),'r'))]).astype(float)

                    pipe = apply_fair_kernel_PCA_to_dataset(
                        (X_train, Y_train, A_train), embedding_dim, standardize=False, fit_classifier=False)

                    # Save transformed data
                    np.savetxt(
                        './fair-manifold-pca/uci/{}/FairKernelPCA_embedding_TRAIN_{}_dim{}.csv'.format(
                            name, split, embedding_dim),
                        pipe.just_transform(data[:, :-2])[:, :embedding_dim])
                    np.savetxt(
                        './fair-manifold-pca/uci/{}/FairKernelPCA_embedding_TEST_{}_dim{}.csv'.format(
                            name, split, embedding_dim),
                        pipe.just_transform(test_data[:, :-2])[:, :embedding_dim])

                elif method == 'INLP':

                    from nullspace_projection.src import debias

                    attr_clf = LinearSVC
                    params_svc = {'fit_intercept': False, 'class_weight': None, 'dual': False,
                                'random_state': 0}
                    params = params_svc
                    min_acc = 0
                    is_autoregressive = True
                    dropout_rate = 0

                    input_dim = data[:, :-2].shape[1]
                    dim_to_remove = input_dim - embedding_dim

                    P, _, _ = debias.get_debiasing_projection(attr_clf, params, dim_to_remove,
                                                            input_dim,
                                                            is_autoregressive, min_acc,
                                                            data[:, :-2], data[:, -1].astype(int),
                                                            data[:, :-2], data[:, -1].astype(int),
                                                            Y_train_main=None, Y_dev_main=None,
                                                            by_class=False, dropout_rate=dropout_rate)
                    try:
                        eigs, U = scipy.sparse.linalg.eigsh(P, k=embedding_dim, which='LM')
                    except scipy.sparse.linalg.ArpackNoConvergence as error:
                        eigs, U = np.linalg.eigh(P)
                        print(eigs)
                        print(error.eigenvalues)
                        U = U[:,-embedding_dim:]
                    # Save transformation matrix
                    np.savetxt(
                        './fair-manifold-pca/uci/{}/INLP_V_{}_dim{}.csv'.format(
                            name, split, embedding_dim),
                        U, delimiter=',')

                elif method == 'RLACE':

                    from rlace_icml import rlace

                    num_iters = 10000
                    input_dim = data[:, :-2].shape[1]
                    dim_to_remove = input_dim - embedding_dim

                    permutation = np.random.permutation(data.shape[0])
                    output = rlace.solve_adv_game(data[permutation, :-2],
                                                data[permutation, -1].astype(int),
                                                data[permutation, :-2],
                                                data[permutation, -1].astype(int),
                                                rank=dim_to_remove, device='cpu',
                                                out_iters=num_iters,
                                                )

                    eigs, U = scipy.sparse.linalg.eigsh(output['P'], k=embedding_dim, which='LM')

                    # Save transformation matrix
                    np.savetxt(
                        './fair-manifold-pca/uci/{}/RLACE_V_{}_dim{}.csv'.format(
                            name, split, embedding_dim),
                        U, delimiter=',')
        
                elif 'Fair Streaming PCA (Offline):' in method:
                    constraint, unfair_dim = method.split(':')[-1].split(',')
                    constraint = constraint.strip()
                    if constraint in ['covariance', 'all']:
                        unfair_dim = int(unfair_dim)
                    else: unfair_dim = ''

                    pca_model = FairPCA()
                    pca_model.fit_offline(
                        X_train, A_train, 
                        target_unfair_dim=unfair_dim,
                        target_pca_dim=embedding_dim,
                        constraint=constraint
                    )

                    # Save transformation matrix
                    dir_path = './fair-manifold-pca/uci/{}/FairStreamingPCA-offline-{}-{}_V_{}_dim{}.csv'.format(
                            name, constraint, unfair_dim, split, embedding_dim)
                    np.savetxt(dir_path, pca_model.V, delimiter=',')


                elif 'Fair Streaming PCA (Iterative):' in method:
                    constraint, unfair_dim = method.split(':')[-1].split(',')
                    constraint = constraint.strip()
                    if constraint in ['covariance', 'all']:
                        unfair_dim = int(unfair_dim)
                    else: unfair_dim = ''

                    pca_model = FairPCA()
                    pca_model.fit_streaming(
                        X_train, A_train, 
                        target_unfair_dim=unfair_dim,
                        target_pca_dim=embedding_dim,
                        n_iter_unfair=100,  # you can manipulate this
                        n_iter_pca=100,     # you can manipulate this
                        constraint=constraint,
                        seed=0,
                    )
                    # Save transformation matrix
                    dir_path = './fair-manifold-pca/uci/{}/FairStreamingPCA-iterative-{}-{}_V_{}_dim{}.csv'.format(
                            name, constraint, unfair_dim, split, embedding_dim)
                    np.savetxt(dir_path, pca_model.V, delimiter=',')