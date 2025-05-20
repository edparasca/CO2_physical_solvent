import pandas as pd
from lightning import pytorch as pl
from sklearn.model_selection import RepeatedKFold, train_test_split

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from chemprop import data, featurizers, nn, utils
from chemprop.models import multi_prod

num_workers = 0
input_path = "../data/dataset1.csv"
smiles_columns = ["SMILES 1", "SMILES 2"]
target_columns = ["Pressure COSMO (bar)"]
features_columns = ["Liquid Fraction 1 (mol/mol)", "Temperature (°C)", "Liquid Fraction 2 (mol/mol)"]
mole_fraction_columns = ["Liquid Fraction 1 (mol/mol)", "Liquid Fraction 2 (mol/mol)"]

df_input = pd.read_csv(input_path)

smiss = df_input.loc[:, smiles_columns].values
ys = df_input.loc[:, target_columns].values

extra_mol_features = df_input.loc[:, features_columns]
extra_mol_features[mole_fraction_columns] = extra_mol_features[mole_fraction_columns] * 0.85
extra_mol_features['Temperature (°C)'] = (extra_mol_features['Temperature (°C)'] + 273.15) / 373.15 * 0.3
for column in mole_fraction_columns:
    extra_mol_features[column] = extra_mol_features[column] * extra_mol_features['Temperature (°C)']

extra_mol_features = extra_mol_features.drop(columns=["Temperature (°C)"])
extra_features_descriptors = extra_mol_features.values

all_data = []

solute_data = [
    data.MoleculeDatapoint(
        mol=utils.make_mol(smiss[i, 0], keep_h=False, add_h=False),
        y=ys[i],
        x_d=extra_features_descriptors[i]
    )
    for i in range(len(smiss))
]
all_data.append(solute_data)

for col_idx in range(1, len(smiles_columns)):
    solvent_data = [
        data.MoleculeDatapoint(
            mol=utils.make_mol(smiss[i, col_idx], keep_h=False, add_h=False),
            y=ys[i],
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

rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

k_train_indices, k_val_indices, k_test_indices = [], [], []
for train_idx, test_idx in rkf.split(mols):
    train_sub_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,          
        random_state=random_state
    )
    k_train_indices.append(train_sub_idx)
    k_val_indices.append(val_idx)
    k_test_indices.append(test_idx)

k_train_data, k_val_data, k_test_data = data.split_data_by_indices(
    all_data, k_train_indices, k_val_indices, k_test_indices
)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

num_folds = len(k_train_data)

all_results = []

for fold_idx in range(num_folds):
    rep = fold_idx // n_splits + 1
    fold = fold_idx % n_splits + 1

    print(f"\nfold = {fold}, rep = {rep}\n")

    train_data_fold = k_train_data[fold_idx]
    val_data_fold   = k_val_data[fold_idx]
    test_data_fold  = k_test_data[fold_idx]

    train_datasets = [
        data.MoleculeDataset(train_data_fold[c], featurizer)
        for c in range(len(train_data_fold))
    ]
    val_datasets = [
        data.MoleculeDataset(val_data_fold[c], featurizer)
        for c in range(len(val_data_fold))
    ]
    test_datasets = [
        data.MoleculeDataset(test_data_fold[c], featurizer)
        for c in range(len(test_data_fold))
    ]

    train_mcdset = data.MulticomponentDataset(train_datasets)
    scaler = train_mcdset.normalize_targets()

    val_mcdset = data.MulticomponentDataset(val_datasets)
    val_mcdset.normalize_targets(scaler)

    test_mcdset = data.MulticomponentDataset(test_datasets)

    train_mcdset.cache = True
    val_mcdset.cache   = True

    depth = 6
    dropout = 0.09
    ffn_hidden_dim = 1000
    ffn_num_layers = 2
    message_hidden_dim = 900
    batch_size = 32

    init_lr = 6.8e-4
    max_lr = 2.6e-4
    final_lr = 7.5e-5
    warmup_epochs = 2

    train_loader = data.build_dataloader(train_mcdset, batch_size=batch_size, num_workers=num_workers)
    val_loader = data.build_dataloader(val_mcdset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_mcdset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    mcmp = nn.MulticomponentMessagePassing(
        blocks=[nn.BondMessagePassing(d_h=message_hidden_dim, depth=depth, dropout=dropout) for _ in range(len(smiles_columns))],
        n_components=len(smiles_columns)
    )

    agg = nn.MeanAggregation()

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn_input_dim = mcmp.output_dim

    ffn = nn.RegressionFFN(
        input_dim=ffn_input_dim,
        output_transform=output_transform,
        hidden_dim=ffn_hidden_dim, 
        n_layers=ffn_num_layers
    )

    metric_list = [nn.metrics.MAE(), nn.metrics.MSE(), nn.metrics.R2Score()]
    mcmpnn = multi_prod.MulticomponentMPNN_mod(
        mcmp,
        agg,
        ffn,
        batch_norm=False,
        metrics=metric_list,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
        warmup_epochs=warmup_epochs,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",         
        patience=5,        
        verbose=False
    )

    checkpointing = ModelCheckpoint(
        dirpath="../checkpoints",
        filename=f"best-fold={fold}-rep={rep}-" + "{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_last=False,
    )

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="cpu",
        devices=1,
        max_epochs=30, 
        callbacks=[early_stopping, checkpointing]
    )

    trainer.fit(mcmpnn, train_loader, val_loader)
    results = trainer.test(mcmpnn, test_loader)
    all_results.append(results)