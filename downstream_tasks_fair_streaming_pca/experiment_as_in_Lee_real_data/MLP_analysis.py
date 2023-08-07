# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys

file_dir = os.path.abspath('.')
sys.path.append(file_dir)

import csv
import numpy as np
from sklearn.neural_network import MLPClassifier

embedding_dim_range = [2,10]
methods_to_run = [
    # 'PCA',
    'FPCA-0',
    'FPCA-0.1',
    # 'MBFPCA-3',
    # 'MBFPCA-6',
    # 'Fair PCA',
    # 'Fair PCA-S (0.5)',
    # 'Fair PCA-S (0.85)',
    # 'Fair Kernel PCA',
    # 'INLP',
    # 'RLACE',
    # 'PCA Samadi',
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
    'German',
    # 'Adult'
]

def make_directory(dir_path):
    try:
        os.mkdir(dir_path)
    except Exception:
        pass

def fairness_metric_neural_network(X, Y, Z):
    clf = MLPClassifier(hidden_layer_sizes=(10, 5), random_state=2023, max_iter=10000)
    clf.fit(X, Y)
    predictions = clf.predict(X)

    acc = np.mean(np.equal(predictions, Y))
    DP = np.abs(np.mean(predictions[Z == 1]) - np.mean(predictions[Z == 0]))

    return acc, DP


for method in methods_to_run:
    for embedding_dim in embedding_dim_range:
        print('Evaluating method ' + method + ' for dim=' + str(embedding_dim))

        accs = np.zeros((10, 3))
        DPs = np.zeros((10, 3))

        for counter1, dataset in enumerate(dataset_names):
            for counter2, split in enumerate(range(10)):

                data = np.array([dPoint for dPoint in csv.reader(
                    open('./fair-manifold-pca/datasets/{}/test_{}.csv'.format(dataset,
                                                                              split),
                         'r'))]).astype(float)

                X = data[:, :-2]
                Y = data[:, -2]
                Z = data[:, -1]

                if method == 'Fair Kernel PCA':
                    if dataset == 'Adult': continue
                    emb_test_data = np.loadtxt(
                        './fair-manifold-pca/uci/{}/FairKernelPCA_embedding_TEST_{}_dim{}.csv'.format(
                            dataset,
                            split, embedding_dim))

                elif method == 'PCA Samadi':
                    emb_test_data = np.loadtxt(
                        './fair-manifold-pca/uci/{}/PCA_Samadi_embedding_TEST_{}_dim{}.csv'.format(
                            dataset,
                            split, embedding_dim), delimiter=',')

                else:
                    if method.split("-")[0] == "FPCA":
                        param = {'0': '00', '0.1': '01'}[method.split("-")[1]]
                        proj_matrix = np.array([dPoint for dPoint in csv.reader(
                            # open('./fair-manifold-pca/uci/{}/{}_fpca_{}/FPCA_V_{}.csv'.format(
                            #     dataset,
                            #     embedding_dim, param, split), 'r'))]).astype(float)
                            open('./fair-manifold-pca/uci/{}/FPCA{}_V_{}_dim{}.csv'.format(
                                dataset, param, split, embedding_dim), 'r'))]).astype(float)

                    elif method.split("-")[0] == "MBFPCA":
                        tau = method.split("-")[1]
                        proj_matrix = np.array([dPoint for dPoint in csv.reader(
                            open(
                                './fair-manifold-pca/uci/{}/MBFPCA{}_V_{}_dim{}.csv'.format(
                                    dataset, tau, split, embedding_dim), 'r'))]).astype(float)

                        # if dataset == 'German':
                        #     proj_matrix = np.array([dPoint for dPoint in csv.reader(
                        #         open(
                        #             './fair-manifold-pca/uci/{}/{}_mbfpca_{}/STFPCA_V_{}.csv'.format(
                        #                 dataset, embedding_dim, tau, split), 'r'))]).astype(float)
                        # else:
                        #     if not (embedding_dim == 10 and tau == '3'):
                        #         proj_matrix = np.array([dPoint for dPoint in csv.reader(
                        #             open(
                        #                 './fair-manifold-pca/uci/{}/{}_stfpca_{}/STFPCA_V_{}.csv'.format(
                        #                     dataset, embedding_dim, tau, split), 'r'))]).astype(
                        #             float)
                        #     else:
                        #         proj_matrix = np.zeros((X.shape[1], embedding_dim))

                    elif method.split("(")[0] == "Fair PCA-S ":
                        param = {'0.5': '05', '0.85': '085'}[method.split("(")[1].split(")")[0]]
                        proj_matrix = np.array([dPoint for dPoint in csv.reader(
                            open(
                                './fair-manifold-pca/uci/{}/FairPCA_S{}_V_{}_dim{}.csv'.format(
                                    dataset, param,
                                    split, embedding_dim, ), 'r'))]).astype(float)
                        
                    elif 'Fair Streaming PCA (Offline):' in method:
                        constraint, unfair_dim = method.split(':')[-1].split(',')
                        constraint = constraint.strip()
                        if constraint in ['covariance', 'all']:
                            unfair_dim = int(unfair_dim)
                        else: unfair_dim = ''

                        proj_matrix = np.array([dPoint for dPoint in csv.reader(
                            open(
                                './fair-manifold-pca/uci/{}/FairStreamingPCA-offline-{}-{}_V_{}_dim{}.csv'.format(
                                dataset, constraint, unfair_dim, split, embedding_dim), 'r'))]).astype(float)

                    elif 'Fair Streaming PCA (Iterative):' in method:
                        constraint, unfair_dim = method.split(':')[-1].split(',')
                        constraint = constraint.strip()
                        if constraint in ['covariance', 'all']:
                            unfair_dim = int(unfair_dim)
                        else: unfair_dim = ''

                        proj_matrix = np.array([dPoint for dPoint in csv.reader(
                            open(
                                './fair-manifold-pca/uci/{}/FairStreamingPCA-iterative-{}-{}_V_{}_dim{}.csv'.format(
                                dataset, constraint, unfair_dim, split, embedding_dim), 'r'))]).astype(float)
                    
                    else:
                        proj_matrix = np.array([dPoint for dPoint in csv.reader(
                            open(
                                './fair-manifold-pca/uci/{}/{}_V_{}_dim{}.csv'.format(
                                    dataset, method.replace(" ", ""), split, embedding_dim),
                                'r'))]).astype(float)
                    
                    emb_test_data = np.matmul(X, proj_matrix)

                accs[counter2, counter1], DPs[counter2, counter1] = fairness_metric_neural_network(
                    emb_test_data, Y, Z)

        if method == "PCA":
            np.savetxt('./fair-manifold-pca/uci/StandardPCA/accs_neural_network_dim{}.csv'.format(
                embedding_dim),
                np.vstack((np.mean(accs, axis=0), np.std(accs, axis=0))), delimiter=',')
            np.savetxt('./fair-manifold-pca/uci/StandardPCA/DPs_neural_network_dim{}.csv'.format(
                embedding_dim),
                np.vstack((np.mean(DPs, axis=0), np.std(DPs, axis=0))), delimiter=',')
        
        elif method.split("-")[0] == "FPCA":
            param = {'0': '00', '0.1': '01'}[method.split("-")[1]]
            np.savetxt(
                './fair-manifold-pca/uci/FPCA{}/accs_neural_network_dim{}.csv'.format(param,
                                                                                      embedding_dim),
                np.vstack((np.mean(accs, axis=0), np.std(accs, axis=0))), delimiter=',')
            np.savetxt(
                './fair-manifold-pca/uci/FPCA{}/DPs_neural_network_dim{}.csv'.format(param,
                                                                                     embedding_dim),
                np.vstack((np.mean(DPs, axis=0), np.std(DPs, axis=0))), delimiter=',')

        elif method.split("-")[0] == "MBFPCA":
            tau = method.split("-")[1]
            np.savetxt(
                './fair-manifold-pca/uci/MBFPCA{}/accs_neural_network_dim{}.csv'.format(tau,
                                                                                        embedding_dim),
                np.vstack((np.mean(accs, axis=0), np.std(accs, axis=0))), delimiter=',')
            np.savetxt(
                './fair-manifold-pca/uci/MBFPCA{}/DPs_neural_network_dim{}.csv'.format(tau,
                                                                                       embedding_dim),
                np.vstack((np.mean(DPs, axis=0), np.std(DPs, axis=0))), delimiter=',')

        elif method.split("(")[0] == "Fair PCA-S ":
            param = {'0.5': '05', '0.85': '085'}[method.split("(")[1].split(")")[0]]
            np.savetxt(
                './fair-manifold-pca/uci/FairPCA-S{}/accs_neural_network_dim{}.csv'.format(param,
                                                                                           embedding_dim),
                np.vstack((np.mean(accs, axis=0), np.std(accs, axis=0))), delimiter=',')
            np.savetxt(
                './fair-manifold-pca/uci/FairPCA-S{}/DPs_neural_network_dim{}.csv'.format(param,
                                                                                          embedding_dim),
                np.vstack((np.mean(DPs, axis=0), np.std(DPs, axis=0))), delimiter=',')
            
        elif 'Fair Streaming PCA (Offline):' in method:
            constraint, unfair_dim = method.split(':')[-1].split(',')
            constraint = constraint.strip()
            if constraint in ['covariance', 'all']:
                unfair_dim = int(unfair_dim)
            else: unfair_dim = ''
            make_directory('./fair-manifold-pca/uci/FairStreamingPCA-offline-{}-{}'.format(constraint, unfair_dim))
            np.savetxt('./fair-manifold-pca/uci/FairStreamingPCA-offline-{}-{}/accs_neural_network_dim{}.csv'.format(
                constraint, unfair_dim, embedding_dim),
                np.vstack((np.mean(accs, axis=0), np.std(accs, axis=0))), delimiter=',')
            np.savetxt('./fair-manifold-pca/uci/FairStreamingPCA-offline-{}-{}/DPs_neural_network_dim{}.csv'.format(
                constraint, unfair_dim, embedding_dim),
                np.vstack((np.mean(DPs, axis=0), np.std(DPs, axis=0))), delimiter=',')
        
        elif 'Fair Streaming PCA (Iterative):' in method:
            constraint, unfair_dim = method.split(':')[-1].split(',')
            constraint = constraint.strip()
            if constraint in ['covariance', 'all']:
                unfair_dim = int(unfair_dim)
            else: unfair_dim = ''
            make_directory('./fair-manifold-pca/uci/FairStreamingPCA-iterative-{}-{}'.format(constraint, unfair_dim))
            np.savetxt('./fair-manifold-pca/uci/FairStreamingPCA-iterative-{}-{}/accs_neural_network_dim{}.csv'.format(
                constraint, unfair_dim, embedding_dim),
                np.vstack((np.mean(accs, axis=0), np.std(accs, axis=0))), delimiter=',')
            np.savetxt('./fair-manifold-pca/uci/FairStreamingPCA-iterative-{}-{}/DPs_neural_network_dim{}.csv'.format(
                constraint, unfair_dim, embedding_dim),
                np.vstack((np.mean(DPs, axis=0), np.std(DPs, axis=0))), delimiter=',')

        else:
            np.savetxt('./fair-manifold-pca/uci/{}/accs_neural_network_dim{}.csv'.format(
                method.replace(" ", ""), embedding_dim),
                np.vstack((np.mean(accs, axis=0), np.std(accs, axis=0))), delimiter=',')
            np.savetxt('./fair-manifold-pca/uci/{}/DPs_neural_network_dim{}.csv'.format(
                method.replace(" ", ""), embedding_dim),
                np.vstack((np.mean(DPs, axis=0), np.std(DPs, axis=0))), delimiter=',')
