import pandas as pd
from pathlib import Path
from lightning import pytorch as pl
from sklearn.preprocessing import StandardScaler

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from chemprop import data, featurizers, nn, utils
from chemprop.models import multi_prod

num_workers = 0
input_path = "../data/dataset2.csv"
smiles_columns = ["SMILES 1", "SMILES 2"]
target_columns = ["Pressure EXP (bar)"]
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


k_train_data, k_val_data, k_test_data = [], [], []

for i in range(n_splits * n_repeats):
    train_idx, val_idx, test_idx = data.make_split_indices(
        mols,
        "random",
        (0.8, 0.1, 0.1),
        seed=random_state + i
    )
    
    # Ottieni i dati splittati di questa fold
    train_data_i, val_data_i, test_data_i = data.split_data_by_indices(
        all_data,
        train_idx,
        val_idx,
        test_idx
    )
    
    # Accumula
    k_train_data.append(train_data_i)
    k_val_data.append(val_data_i)
    k_test_data.append(test_data_i)

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

num_folds = len(k_train_data)
ckpt_dir = Path("../checkpoints")

all_results = []

for fold_idx in range(num_folds):
    rep = fold_idx // n_splits + 1
    fold = fold_idx % n_splits + 1

    print(f"\nfold = {fold}, rep = {rep}\n")

    pattern = f"best-fold={fold}-rep={rep}-epoch=*.ckpt"
    matches = list(ckpt_dir.glob(pattern))
    if not matches:
        print(f"Checkpoint non trovato per fold={fold}, rep={rep}")
        continue
    
    checkpoint_path = matches[0]  

    mpnn_cls = multi_prod.MulticomponentMPNN_mod
    mcmpnn = mpnn_cls.load_from_checkpoint(checkpoint_path)

    scaler = StandardScaler()
    scaler.mean_ = mcmpnn.predictor.output_transform.mean.numpy()
    scaler.scale_ = mcmpnn.predictor.output_transform.scale.numpy()

    train_data_fold = k_train_data[fold_idx]
    val_data_fold   = k_val_data[fold_idx]
    test_data_fold  = k_test_data[fold_idx]

    train_datasets = [data.MoleculeDataset(train_data_fold[0][i], featurizer) for i in range(len(smiles_columns))]
    val_datasets = [data.MoleculeDataset(val_data_fold[0][i], featurizer) for i in range(len(smiles_columns))]
    test_datasets = [data.MoleculeDataset(test_data_fold[0][i], featurizer) for i in range(len(smiles_columns))]

    train_mcdset = data.MulticomponentDataset(train_datasets)
    train_mcdset.normalize_targets(scaler)

    val_mcdset = data.MulticomponentDataset(val_datasets)
    val_mcdset.normalize_targets(scaler)

    test_mcdset = data.MulticomponentDataset(test_datasets)

    train_mcdset.cache = True
    val_mcdset.cache   = True


    train_loader = data.build_dataloader(train_mcdset, num_workers=num_workers)
    val_loader = data.build_dataloader(val_mcdset, num_workers=num_workers, shuffle=False)
    test_loader = data.build_dataloader(test_mcdset, num_workers=num_workers, shuffle=False)

    blocks_to_freeze = [0, 1]
    for i in blocks_to_freeze:
        mp_block = mcmpnn.message_passing.blocks[i]
        mp_block.apply(lambda module: module.requires_grad_(False))
    mp_block.eval()
    mcmpnn.bn.apply(lambda module: module.requires_grad_(False))
    mcmpnn.bn.eval()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",         
        patience=5,        
        verbose=False
    )

    checkpointing = ModelCheckpoint(
        dirpath="../checkpoints",
        filename=f"fine-fold={fold}-rep={rep}-" + "{epoch}-{val_loss:.2f}",
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