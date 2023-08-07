# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys

file_dir = os.path.abspath('.')
sys.path.append(file_dir)

import numpy as np

embedding_dim_range = [2, 10]
methods_to_run = [
    'PCA',
    'FPCA-0.1',
    'FPCA-0',
    'MBFPCA-3',
    'MBFPCA-6',
    'Fair PCA',
    'Fair PCA-S (0.85)',
    'Fair PCA-S (0.5)',
    'Fair Kernel PCA',
    'INLP',
    'RLACE',
    'PCA Samadi',
    'Fair Streaming PCA (Offline): mean, null', 'Fair Streaming PCA (Iterative): mean, null',
    'Fair Streaming PCA (Offline): all, 2', 'Fair Streaming PCA (Iterative): all, 2',
    'Fair Streaming PCA (Offline): all, 5', 'Fair Streaming PCA (Iterative): all, 5',
    'Fair Streaming PCA (Offline): all, 10', 'Fair Streaming PCA (Iterative): all, 10',
    'Fair Streaming PCA (Offline): all, 15', 'Fair Streaming PCA (Iterative): all, 15',
    'Fair Streaming PCA (Offline): all, 25', 'Fair Streaming PCA (Iterative): all, 25',
    'Fair Streaming PCA (Offline): all, 50', 'Fair Streaming PCA (Iterative): all, 50',
]

    

dataset_names = [
    'COMPAS',
    'German',
    'Adult'
]

method_to_folder_dict = {
    'PCA': 'StandardPCA',
    'FPCA-0': 'FPCA00',
    'FPCA-0.1': 'FPCA01',
    'MBFPCA-3': 'MBFPCA3',
    'MBFPCA-6': 'MBFPCA6',
    'Fair PCA': 'FairPCA',
    'Fair PCA-S (0.5)': 'FairPCA-S05',
    'Fair PCA-S (0.85)': 'FairPCA-S085',
    'Fair Kernel PCA': 'FairKernelPCA',
    'INLP': 'INLP',
    'RLACE': 'RLACE',
    'PCA Samadi': 'PCASamadi',
    'Fair Streaming PCA (Offline): mean, null': 'FairStreamingPCA-offline-mean-',
    'Fair Streaming PCA (Offline): all, 2': 'FairStreamingPCA-offline-all-2',
    'Fair Streaming PCA (Offline): all, 5': 'FairStreamingPCA-offline-all-5',
    'Fair Streaming PCA (Offline): all, 10': 'FairStreamingPCA-offline-all-10',
    'Fair Streaming PCA (Offline): all, 15': 'FairStreamingPCA-offline-all-15',
    'Fair Streaming PCA (Offline): all, 25': 'FairStreamingPCA-offline-all-25',
    'Fair Streaming PCA (Offline): all, 50': 'FairStreamingPCA-offline-all-50',
    'Fair Streaming PCA (Iterative): mean, null': 'FairStreamingPCA-iterative-mean-',
    'Fair Streaming PCA (Iterative): all, 2': 'FairStreamingPCA-iterative-all-2',
    'Fair Streaming PCA (Iterative): all, 5': 'FairStreamingPCA-iterative-all-5',
    'Fair Streaming PCA (Iterative): all, 10': 'FairStreamingPCA-iterative-all-10',
    'Fair Streaming PCA (Iterative): all, 15': 'FairStreamingPCA-iterative-all-15',
    'Fair Streaming PCA (Iterative): all, 25': 'FairStreamingPCA-iterative-all-25',
    'Fair Streaming PCA (Iterative): all, 50': 'FairStreamingPCA-iterative-all-50',
    }

for counter1, dataset in enumerate(dataset_names):
    print('-----------------------------------------------------------------------------------')
    print('-----------------------------------------------------------------------------------')
    print(f'Dataset = {dataset}')
    for embedding_dim in embedding_dim_range:
        print('-----------------------------------------------------------------------------------')
        print(f'Embedding dim ($k$) = {embedding_dim}')
        print('-----------------------------------------------------------------------------------')
        for method in methods_to_run:
            method_name = method
            if 'Fair Streaming PCA' in method:
                b = method.split(':')[1][1:]
                if 'null' in b:
                    b = 'mean'
                else:
                    b = b[-2:]
                if 'Offline' in method:
                    method_name = f"\\textbf{{Ours}} (offline, {b})"
                else:
                    method_name = f"\\textbf{{Ours}} (FNPM, {b})" 

            if method == 'PCA':
                print(f"\\multirow{{18}}{{*}}{{{embedding_dim}}}", end=" ")

            # print(f'Method = {method_name}')
            print(f"& {method_name} &", end=" ")

            if (method == 'Fair Kernel PCA' and dataset == 'Adult'):
                print("\\multicolumn{8}{c}{Takes too long time} \\\\")
                continue

            if method[:4] == 'FPCA' and dataset != 'German':
                print("\\multicolumn{8}{c}{Memory Out} \\\\")
                continue

            if method == 'MBFPCA-6' and dataset == 'Adult' and embedding_dim == 10:
                print("\\multicolumn{8}{c}{Takes too long time} \\\\")
                continue 

            if method == 'MBFPCA-3' and dataset == 'Adult' and embedding_dim == 10:
                print("\\multicolumn{8}{c}{Takes too long time} \\\\")
                continue

            # if method == 'MBFPCA-3' and embedding_dim == 10 and not (dataset == 'German'):
            #     print('Data not available in the repository of Lee et al.')
            #     print('***************************************************************************')
            #     continue

            if method in ['Fair Kernel PCA', 'PCA Samadi']:
                var = (np.nan, np.nan)
                print(f"N/A &", end=" ")
            else:
                var = np.around(np.loadtxt(
                    './fair-manifold-pca/uci/{}/exp_vars_test_dim{}.csv'.format(
                        method_to_folder_dict[method],
                        embedding_dim), delimiter=',')[:, counter1], 2)
                # print(f'%Var($\\uparrow$) = {var[0]}$_{{({var[1]})}}$')
                print(f"{var[0]}$_{{({var[1]})}}$ &", end=" ")

            if method in ['Fair Kernel PCA', 'PCA Samadi']:
                mmd = (np.nan, np.nan)
                print(f"N/A &", end=" ")
            else:
                mmd = np.around(np.loadtxt(
                    './fair-manifold-pca/uci/{}/mmds_test_dim{}.csv'.format(
                        method_to_folder_dict[method],
                        embedding_dim), delimiter=',')[:, counter1], 3)
                # print(f'MMD$^2$($\\downarrow$) = {mmd[0]}$_{{({mmd[1]})}}$')
                print(f"{mmd[0]}$_{{({mmd[1]})}}$ &", end=" ")

            acc_kernel = np.around(100 * np.loadtxt(
                './fair-manifold-pca/uci/{}/accs_dim{}.csv'.format(method_to_folder_dict[method],
                                                                   embedding_dim), delimiter=',')[:,
                                         counter1], 2)
            # print(f'%Acc($\\uparrow$) - kernel SVM = {acc_kernel[0]}$_{{({acc_kernel[1]})}}$')
            print(f"{acc_kernel[0]}$_{{({acc_kernel[1]})}}$ &", end=" ")

            dp_kernel = np.around(np.loadtxt(
                './fair-manifold-pca/uci/{}/DPs_dim{}.csv'.format(method_to_folder_dict[method],
                                                                  embedding_dim), delimiter=',')[:,
                                  counter1], 2)
            # print(f'$\Delta_{{DP}}$($\\downarrow$) - kernel SVM = {dp_kernel[0]}$_{{({dp_kernel[1]})}}$')
            print(f"{dp_kernel[0]}$_{{({dp_kernel[1]})}}$ &", end=" ")

            acc_linear = np.around(100 * np.loadtxt(
                './fair-manifold-pca/uci/{}/accs_linear_dim{}.csv'.format(
                    method_to_folder_dict[method],
                    embedding_dim), delimiter=',')[:, counter1], 2)
            # print(f'%Acc($\\uparrow$) - linear SVM = {acc_linear[0]}$_{{({acc_linear[1]})}}$')
            print(f"{acc_linear[0]}$_{{({acc_linear[1]})}}$ &", end=" ")

            dp_linear = np.around(np.loadtxt(
                './fair-manifold-pca/uci/{}/DPs_linear_dim{}.csv'.format(
                    method_to_folder_dict[method],
                    embedding_dim), delimiter=',')[:, counter1], 2)
            # print(f'$\Delta_{{DP}}$($\\downarrow$) - linear SVM = {dp_linear[0]}$_{{({dp_linear[1]})}}$')
            print(f"{dp_linear[0]}$_{{({dp_linear[1]})}}$ &", end=" ")

            acc_mlp = np.around(100 * np.loadtxt(
                './fair-manifold-pca/uci/{}/accs_neural_network_dim{}.csv'.format(
                    method_to_folder_dict[method],
                    embedding_dim), delimiter=',')[:, counter1], 2)
            # print(f'%Acc($\\uparrow$) - MLP = {acc_mlp[0]}$_{{({acc_mlp[1]})}}$')
            print(f"{acc_mlp[0]}$_{{({acc_mlp[1]})}}$ &", end=" ")

            dp_mlp = np.around(np.loadtxt(
                './fair-manifold-pca/uci/{}/DPs_neural_network_dim{}.csv'.format(
                    method_to_folder_dict[method],
                    embedding_dim), delimiter=',')[:, counter1], 2)
            # print(f'$\Delta_{{DP}}$($\\downarrow$) - MLP = {dp_mlp[0]}$_{{({dp_mlp[1]})}}$')
            print(f"{dp_mlp[0]}$_{{({dp_mlp[1]})}}$ \\\\")

            # print('*******************************************************************************')

    print('\n\n')
