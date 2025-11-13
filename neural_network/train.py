import argparse
import os
os.chdir("..")
from datetime import datetime
from os.path import dirname, join, realpath

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import numpy as np

from datamodules import DataModule
from networks import ExposureResNet
import yaml


############# PARAMETERS #############

SCRIPT_DIR = dirname(realpath(__file__))
SAVE_PATH = join(SCRIPT_DIR, "..", "output", "training")

parser = argparse.ArgumentParser(description="Train model for audio prediction from terrain patches.")
parser.add_argument("--folds", type=int, help="Number of folds for cross-validation.", default=1)
parser.add_argument("--project", type=str, help="WandB project name.", default="End_to_end_AE")
args = parser.parse_args()

PROJECT_NAME = args.project
NUM_FOLDS = args.folds  # Number of folds for cross-validation
DATA_FOLDERS = [
    '/home/alienware/Desktop/tmp/training_trajs/'
]
BATCH_SIZE = 2
NUM_EPOCHS = 50
PATIENCE = 20  # Set to NUM_EPOCHS to disable early stopping
LEARNING_RATE = 0.0005
INPUT_SIZE = (640, 480)   # Size of the input images
OUTPUT_SIZE = 1  # Size of the output (number of classes or regression output)

MODEL_TYPE = "ResNet"
DATA_SPLIT = 10  # Number of parts to split the dataset into test, validation     
SEED = 42  # Seed for the random split

######################################

if __name__ == "__main__":

    # Check if CUDA is available
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Weights & Biases
    os.environ["WANDB_SILENT"] = "true"
    tags = [MODEL_TYPE]

    # Load the dataset
    datamodule = DataModule(
        input_size=INPUT_SIZE,
        data_folders=DATA_FOLDERS,
        batch_size=BATCH_SIZE,
        split_ratio=DATA_SPLIT,
        split_seed=SEED,
    )

    # Create a save folder matching the date and time
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = join(SAVE_PATH, time_str)
    os.makedirs(save_folder, exist_ok=True)

    # Save details to a YAML file
    details = {
        "data_folder": DATA_FOLDERS,
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model_type": MODEL_TYPE,
        "data_split": DATA_SPLIT,
        "seed": SEED,
    }

    with open(join(save_folder, "details.yaml"), "w") as file:
        yaml.dump(details, file)

    calib = np.loadtxt("DROID-SLAM/calib/belair.txt", delimiter=" ")
    # Start cross-validation
    for k in range(args.folds):
        if args.folds > 1:
            print("\n--------------------------------")
            print(f"  Fold {k + 1} / {args.folds}")
            print("--------------------------------\n")
            fold_save_folder = join(save_folder, f"fold_{k+1}")
        else:
            fold_save_folder = save_folder
        print(f"Saving to {fold_save_folder}")

        # Create an instance of the selected model
        if MODEL_TYPE == "ResNet":
            model = ExposureResNet(output_size=OUTPUT_SIZE, lr=LEARNING_RATE, calib=calib, n_images=3)
        else:
            raise ValueError("Invalid model type.")

        # Define logger
        # logger = WandbLogger(
        #     log_model=False,
        #     project=args.project,
        #     save_dir=fold_save_folder,
        #     name=f"{time_str}_fold_{k+1}",
        #     tags=tags,
        # )

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=fold_save_folder, filename="model"),
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(leave=True),
        ]

        trainer = Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="gpu",
            # logger=logger,
            log_every_n_steps=1,
            enable_checkpointing=True,
            callbacks=callbacks,
            precision="32-true",
        )

        # Train the model
        datamodule.k = k
        trainer.fit(model, datamodule)

        # Save the model as a TorchScript file
        best_checkpoint = torch.load(join(fold_save_folder, "model.ckpt"), weights_only=True)
        model.load_state_dict(best_checkpoint["state_dict"])
        model.to_torchscript(join(fold_save_folder, "model.pt"))

        wandb.finish()