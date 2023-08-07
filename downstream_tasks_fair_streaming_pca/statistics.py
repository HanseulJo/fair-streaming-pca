import pandas as pd
from itertools import product

embedding_dim_range = [2, 10]
dataset_names = [
    'German',
    'COMPAS',
    'Adult',
]
methods = [
    'offline',
    'streaming'
]
constraints = [
    ('vanilla', None),
    ('mean', None),
    ('all', 2),
    ('all', 5),
    ('all', 10),
    ('all', 25),
    ('all', 50)
]

stat_dict = {
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
for dataset_name, dim, method in product(dataset_names, embedding_dim_range, methods):
    df = pd.read_csv(f'{dataset_name}_{method}_dim{dim}.csv')
    
    for constraint, unfair_dim in constraints:
        name_ = constraint + ('' if unfair_dim is None else str(unfair_dim))
        data = df[df['name'].str.contains(name_)].drop(columns='name')
        means, stds = data.mean(0).to_numpy(), data.std(0).to_numpy()
        means[[0,2,4,6]] *= 100
        stds[[0,2,4,6]] *= 100
        expvar_m, mmd_m, acc_lin_m, dp_lin_m, acc_ker_m, dp_ker_m, acc_mlp_m, dp_mlp_m = means
        expvar_s, mmd_s, acc_lin_s, dp_lin_s, acc_ker_s, dp_ker_s, acc_mlp_s, dp_mlp_s = stds
        stat_dict['name'].append(f"{dataset_name}_{method}_dim{dim}_{name_}")
        stat_dict['Explained Variance Ratio (train)'].append(f"{expvar_m:.2f} ({expvar_s:.2f})")
        stat_dict['Maximum Mean Discrepancy (train)'].append(f"{mmd_m:.3f} ({mmd_s:.3f})")
        stat_dict['Accuracy (linear_svm)'].append(f"{acc_lin_m:.2f} ({acc_lin_s:.2f})")
        stat_dict['Delta_DP (linear_svm)'].append(f"{dp_lin_m:.2f} ({dp_lin_s:.2f})")
        stat_dict['Accuracy (kernel_svm)'].append(f"{acc_ker_m:.2f} ({acc_ker_s:.2f})")
        stat_dict['Delta_DP (kernel_svm)'].append(f"{dp_ker_m:.2f} ({dp_ker_s:.2f})")
        stat_dict['Accuracy (mlp)'].append(f"{acc_mlp_m:.2f} ({acc_mlp_s:.2f})")
        stat_dict['Delta_DP (mlp)'].append(f"{dp_mlp_m:.2f} ({dp_mlp_s:.2f})")

pd.DataFrame.from_dict(stat_dict).to_csv('statistics.csv', index=False)

