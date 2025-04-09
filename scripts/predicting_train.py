import pandas as pd
import numpy as np
from pathlib import Path
import torch
from lightning import pytorch as pl

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from chemprop import data, featurizers, nn, utils
from chemprop.models import multi_prod

input_path = "../data/dataset3.csv"
smiles_columns = ["SMILES 1", "SMILES 2"]
target_columns = ["Pressure EXP (bar)"]
features_columns = ["Liquid Fraction 1 (mol/mol)", "Temperature (°C)", "Liquid Fraction 2 (mol/mol)"]
mole_fraction_columns = ["Liquid Fraction 1 (mol/mol)", "Liquid Fraction 2 (mol/mol)"]

df_input = pd.read_csv(input_path)

smiss = df_input.loc[:, smiles_columns].values
ys = df_input.loc[:, target_columns].values

extra_mol_features = df_input.loc[:, features_columns]
extra_mol_features[mole_fraction_columns] = extra_mol_features[mole_fraction_columns] * 0.4
extra_mol_features['Temperature (°C)'] = (extra_mol_features['Temperature (°C)'] + 273.15) / 373.15 * 0.1
for column in mole_fraction_columns:
    extra_mol_features[column] = extra_mol_features[column] * extra_mol_features['Temperature (°C)']

extra_mol_features = extra_mol_features.drop(columns=["Temperature (°C)"])
extra_features_descriptors = extra_mol_features.values

all_data = []

solute_data = [
    data.MoleculeDatapoint(
        mol=utils.make_mol(smiss[i, 0], keep_h=True, add_h=False),
        x_d=extra_features_descriptors[i]
    )
    for i in range(len(smiss))
]
all_data.append(solute_data)

for col_idx in range(1, len(smiles_columns)):
    solvent_data = [
        data.MoleculeDatapoint(
            mol=utils.make_mol(smiss[i, col_idx], keep_h=False, add_h=False),
            x_d=extra_features_descriptors[i]
        )
        for i in range(len(smiss))
    ]
    all_data.append(solvent_data)

component_to_split_by = 1
mols = [d.mol for d in all_data[component_to_split_by]]

n_splits = 3
n_repeats = 3
random_state = 42

k_train_indices, k_val_indices, k_test_indices = [], [], []

k_train_data, k_val_data, k_test_data = [], [], []

for i in range(n_splits * n_repeats):
    train_idx, val_idx, test_idx = data.make_split_indices(
        mols,
        "random",
        (0.8, 0.1, 0.1),
        seed=random_state + i
    )
    
    k_train_indices.append(np.array(train_idx).flatten())
    k_val_indices.append(np.array(val_idx).flatten())
    k_test_indices.append(np.array(test_idx).flatten())

    train_data_i, val_data_i, test_data_i = data.split_data_by_indices(
        all_data,
        train_idx,
        val_idx,
        test_idx
    )
    
    k_train_data.append(train_data_i)
    k_val_data.append(val_data_i)
    k_test_data.append(test_data_i)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

num_folds = len(k_test_data)
ckpt_dir = Path("../checkpoints")

all_test_results = []
fold_test_metrics = []

all_train_results = []
fold_train_metrics = []

for fold_idx in range(num_folds):
    rep = fold_idx // n_splits + 1
    fold = fold_idx % n_splits + 1

    print(f"\nfold = {fold}, rep = {rep}\n")

    pattern = f"last-fold={fold}-rep={rep}-epoch=*.ckpt"
    matches = list(ckpt_dir.glob(pattern))
    if not matches:
        print(f"Checkpoint non trovato per fold={fold}, rep={rep}")
        continue
    
    checkpoint_path = matches[0]  

    mpnn_cls = multi_prod.MulticomponentMPNN_mod
    mcmpnn = mpnn_cls.load_from_checkpoint(checkpoint_path)

    train_data_fold  = k_train_data[fold_idx]

    train_datasets = [data.MoleculeDataset(train_data_fold[0][i], featurizer) for i in range(len(smiles_columns))]

    train_mcdset = data.MulticomponentDataset(train_datasets)
    train_loader = data.build_dataloader(train_mcdset, shuffle=False)

    with torch.inference_mode():
        trainer = pl.Trainer(
            logger=None,
            enable_progress_bar=True,
            accelerator="cpu",
            devices=1
    )
    train_preds = trainer.predict(mcmpnn, train_loader)

    train_preds = np.concatenate(train_preds, axis=0)
    train_indices = k_train_indices[fold_idx]

    train_results_df = df_input.iloc[train_indices].copy()
    train_results_df["Pressure PRED (bar)"] = train_preds
    train_results_df["Pressure EXP (bar)"] = ys[train_indices]

    train_results_df["OriginalIndex"] = train_indices
    train_results_df["Fold"] = fold
    train_results_df["Replicate"] = rep

    val_data_fold  = k_val_data[fold_idx]

    val_datasets = [data.MoleculeDataset(val_data_fold[0][i], featurizer) for i in range(len(smiles_columns))]

    val_mcdset = data.MulticomponentDataset(val_datasets)
    val_loader = data.build_dataloader(val_mcdset, shuffle=False)
    val_preds = trainer.predict(mcmpnn, val_loader)

    val_preds = np.concatenate(val_preds, axis=0)
    val_indices = k_val_indices[fold_idx]

    val_results_df = df_input.iloc[val_indices].copy()
    val_results_df["Pressure PRED (bar)"] = val_preds
    val_results_df["Pressure EXP (bar)"] = ys[val_indices]

    val_results_df["OriginalIndex"] = val_indices
    val_results_df["Fold"] = fold
    val_results_df["Replicate"] = rep

    train_results_df = pd.concat([train_results_df, val_results_df], ignore_index=True)
    all_train_results.append(train_results_df)

    test_data_fold  = k_test_data[fold_idx]

    test_datasets = [data.MoleculeDataset(test_data_fold[0][i], featurizer) for i in range(len(smiles_columns))]

    test_mcdset = data.MulticomponentDataset(test_datasets)
    test_loader = data.build_dataloader(test_mcdset, shuffle=False)
    test_preds = trainer.predict(mcmpnn, test_loader)

    test_preds = np.concatenate(test_preds, axis=0)
    test_indices = k_test_indices[fold_idx]

    test_results_df = df_input.iloc[test_indices].copy()
    test_results_df["Pressure PRED (bar)"] = test_preds
    test_results_df["Pressure EXP (bar)"] = ys[test_indices]

    test_results_df["OriginalIndex"] = test_indices
    test_results_df["Fold"] = fold
    test_results_df["Replicate"] = rep
    all_test_results.append(test_results_df)

final_train_results = pd.concat(all_train_results, ignore_index=True)
final_test_results = pd.concat(all_test_results, ignore_index=True)

train_results_output_path = "../results/all_results_train_ft2.csv"
final_train_results.to_csv(train_results_output_path, index=False)
print(f"Saving results in: {train_results_output_path}")

test_results_output_path = "../results/all_results_test_ft2.csv"
final_test_results.to_csv(test_results_output_path, index=False)
print(f"Saving results in: {test_results_output_path}")

df_train_1 = pd.read_csv("../results/all_results_train_ft1.csv")
df_test_2 = pd.read_csv("../results/all_results_test_ft2.csv")

df_train_2 = pd.read_csv("../results/all_results_train_ft2.csv")
df_test_1 = pd.read_csv("../results/all_results_test_ft1.csv")

df_train_2 = df_train_2[df_train_2["CAS-RN 1"] == "7783-06-4"]
df_test_2 = df_test_2[df_test_2["CAS-RN 1"] == "7783-06-4"]

df_train = pd.concat([df_train_1, df_train_2], ignore_index=True)
df_test = pd.concat([df_test_1, df_test_2], ignore_index=True)

df_train.to_csv("../results/all_results_train.csv", index=False)
df_test.to_csv("../results/all_results_test.csv", index=False)

def calculate_metrics(group):
    y_true = group["Pressure EXP (bar)"]
    y_pred = group["Pressure PRED (bar)"]
    
    return pd.Series({
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MBE': (y_pred - y_true).mean(),  # Mean Bias Error
        'R2': r2_score(y_true, y_pred)
    })

metrics_train = df_train.groupby(['Fold', 'Replicate']).apply(calculate_metrics, include_groups=False).reset_index()
metrics_test = df_test.groupby(['Fold', 'Replicate']).apply(calculate_metrics, include_groups=False).reset_index()

metrics_train = metrics_train[['Fold', 'Replicate', 'MAE', 'RMSE', 'MBE', 'R2']]
metrics_test = metrics_test[['Fold', 'Replicate', 'MAE', 'RMSE', 'MBE', 'R2']]

metrics_train_output_path = "../results/fold_metrics_train.csv"
metrics_train.to_csv(metrics_train_output_path, index=False)
print(f"Saving metrics in: {metrics_train_output_path}")

metrics_test_output_path = "../results/fold_metrics_test.csv"
metrics_test.to_csv(metrics_test_output_path, index=False)
print(f"Saving metrics in: {metrics_test_output_path}")

summary_metrics_train = metrics_train[['MAE', 'RMSE', 'MBE', 'R2']].agg(['mean', 'std'])
summary_metrics_test = metrics_test[['MAE', 'RMSE', 'MBE', 'R2']].agg(['mean', 'std'])

summary_metrics_train['Set'] = 'Train'
summary_metrics_test['Set'] = 'Test'
summary_metrics = pd.concat([summary_metrics_train, summary_metrics_test])

summary_metrics_output_path = "../results/summary_metrics.csv"
summary_metrics.to_csv(summary_metrics_output_path)
print(f"Saving summary metrics in: {summary_metrics_output_path}")

# final_train_results = pd.read_csv("../results/all_results_train.csv")
# final_test_results = pd.read_csv("../results/all_results_test.csv")

# grouping_columns = [
#     'Component 1', 'CAS-RN 1', 'Component 2', 'CAS-RN 2', 
#     'Liquid Fraction 1 (mol/mol)', 'Temperature (°C)', 
#     'Liquid Fraction 2 (mol/mol)', 'Pressure EXP (bar)', 
#     'SMILES 1', 'SMILES 2'
# ]

# final_train_results = final_train_results.groupby(grouping_columns, as_index=False)['Pressure PRED (bar)'].mean()
# final_test_results = final_test_results.groupby(grouping_columns, as_index=False)['Pressure PRED (bar)'].mean()

# train_results_output_path = "../results/all_results_train_mean.csv"
# final_train_results.to_csv(train_results_output_path, index=False)
# print(f"Saving results in: {train_results_output_path}")

# test_results_output_path = "../results/all_results_test_mean.csv"
# final_test_results.to_csv(test_results_output_path, index=False)
# print(f"Saving results in: {test_results_output_path}")