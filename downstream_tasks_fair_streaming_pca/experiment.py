import os.path as osp
import pandas as pd
from jax import numpy as jnp
from fair_streaming_pca import FairPCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from mmd import mmd_rbf, compute_bandwidth

embedding_dim_range = [2, 10]

dataset_names = [
    'German',
    'COMPAS',
    'Adult',
]

classifiers = dict(
    linear_svm = SVC,
    kernel_svm = SVC,
    mlp = MLPClassifier
)

clf_args = {
    'linear_svm': dict(kernel='linear'),
    'kernel_svm': dict(kernel='rbf'),
    'mlp': dict(hidden_layer_sizes=(10,5), random_state=999, max_iter=10000)
}

root_dir = 'datasets'

constraints = [
    ('vanilla', None),
    ('mean', None),
    ('all', 2),
    ('all', 5),
    ('all', 10),
    ('all', 25),
    ('all', 50)
]

for embedding_dim in embedding_dim_range:
    print('Embedding dim = ' + str(embedding_dim))

    for dataset_name in dataset_names:
        data_dir = osp.join(root_dir, dataset_name)
        df_results_offline = {
            "name" : [],
            "Explained Variance Ratio (train)": [],
            'Maximum Mean Discrepancy (train)': [],
            "Accuracy (linear_svm)": [],
            "Delta_DP (linear_svm)": [],
            "Accuracy (kernel_svm)": [],
            "Delta_DP (kernel_svm)": [],
            "Accuracy (mlp)": [],
            "Delta_DP (mlp)": []
        }
        df_results_streaming = {
            "name" : [],
            "Explained Variance Ratio (train)": [],
            'Maximum Mean Discrepancy (train)': [],
            "Accuracy (linear_svm)": [],
            "Delta_DP (linear_svm)": [],
            "Accuracy (kernel_svm)": [],
            "Delta_DP (kernel_svm)": [],
            "Accuracy (mlp)": [],
            "Delta_DP (mlp)": []
        }
        for split in range(10):
            print(f'{dataset_name}, repeat={split}')
            df_train = pd.read_csv(osp.join(data_dir, f'train_{split}.csv'), header=None)
            df_test = pd.read_csv(osp.join(data_dir, f'test_{split}.csv'), header=None)
            data_length = df_train.shape[1]

            X_train, Y_train, A_train = df_train[range(data_length-2)], df_train[data_length-2], df_train[data_length-1]
            X_test, Y_test, A_test = df_test[range(data_length-2)], df_test[data_length-2], df_test[data_length-1]
            Y_train, A_train, Y_test, A_test = Y_train.astype(int), A_train.astype(int), Y_test.astype(int), A_test.astype(int)
            sigma = compute_bandwidth(X_train, embedding_dim)
            
            for constraint, unfair_dim in constraints:
                ## Offline PCA with EigenDecomposition
                print()
                df_results_offline['name'].append(f"{constraint+('' if unfair_dim is None else str(unfair_dim))}_split{split}")
                print(df_results_offline['name'][-1] + '_offline')
                pca_model = FairPCA()
                pca_model.fit_offline(
                    X_train, A_train, 
                    target_unfair_dim=unfair_dim,
                    target_pca_dim=embedding_dim,
                    constraint=constraint
                )
                df_results_offline['Explained Variance Ratio (train)'].append(pca_model.explained_variance_ratio)
                print("ExpVarRatio (train) :", pca_model.explained_variance_ratio)
                X_train_projected = pca_model.transform_low_dim(X_train)
                X_test_projected = pca_model.transform_low_dim(X_test)
                mmd_01 = mmd_rbf(X_train_projected[A_train==0], X_train_projected[A_train==1])
                df_results_offline['Maximum Mean Discrepancy (train)'].append(mmd_01)
                print("MMD (train) :", mmd_01)

                for classifier_name, classifier_class in classifiers.items():
                    print(classifier_name)
                    classifier = classifier_class(**clf_args[classifier_name])
                    classifier.fit(X_train_projected, Y_train)
                    y_test_pred = classifier.predict(X_test_projected)
                    acc = (y_test_pred == Y_test).mean()
                    dp = abs(y_test_pred[A_test==1].mean() - y_test_pred[A_test==0].mean())
                    df_results_offline[f'Accuracy ({classifier_name})'].append(acc)
                    df_results_offline[f'Delta_DP ({classifier_name})'].append(dp)
                    print("Accuracy (test) :", acc)
                    print("Delta_DP (test) :", dp)

                ## Iterative PCA with Power Method
                print()
                df_results_streaming['name'].append(f"{constraint+('' if unfair_dim is None else str(unfair_dim))}_split{split}")
                print(df_results_streaming['name'][-1] + '_streaming')
                pca_model = FairPCA()
                pca_model.fit_streaming(
                    X_train, A_train, 
                    target_unfair_dim=unfair_dim,
                    target_pca_dim=embedding_dim,
                    n_iter_unfair=10,
                    n_iter_pca=10,
                    constraint=constraint,
                    seed=0,
                )
                df_results_streaming['Explained Variance Ratio (train)'].append(pca_model.explained_variance_ratio)
                print("ExpVarRatio (train) :", pca_model.explained_variance_ratio)
                X_train_projected = pca_model.transform_low_dim(X_train)
                X_test_projected = pca_model.transform_low_dim(X_test)
                mmd_01 = mmd_rbf(X_train_projected[A_train==0], X_train_projected[A_train==1])
                df_results_streaming['Maximum Mean Discrepancy (train)'].append(mmd_01)
                print("MMD (train) :", mmd_01)
                
                for classifier_name, classifier_class in classifiers.items():
                    print(classifier_name)
                    classifier = classifier_class(**clf_args[classifier_name])
                    classifier.fit(X_train_projected, Y_train)
                    y_test_pred = classifier.predict(X_test_projected)
                    acc = (y_test_pred == Y_test).mean()
                    dp = abs(y_test_pred[A_test==1].mean() - y_test_pred[A_test==0].mean())
                    df_results_streaming[f'Accuracy ({classifier_name})'].append(acc)
                    df_results_streaming[f'Delta_DP ({classifier_name})'].append(dp)
                    print("Accuracy (test) :", acc)
                    print("Delta_DP (test) :", dp)
            pd.DataFrame.from_dict(df_results_offline).sort_values(by=['name']).to_csv(f"{dataset_name}_offline_dim{embedding_dim}.csv", index=False)
            pd.DataFrame.from_dict(df_results_streaming).sort_values(by=['name']).to_csv(f"{dataset_name}_streaming_dim{embedding_dim}.csv", index=False)
            print()
