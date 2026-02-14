"""
Training Script for Cloud Hole Detection - Azure ML Compatible

This script is designed to run in Azure ML Studio with:
- Command-line arguments for hyperparameters
- Azure ML logging integration
- Model registration to Azure ML
- Handles mounted/downloaded datasets from Azure

Usage:
    # Local
    python train_azure.py --data_path data/gold/datasets/v1.0_baseline

    # Azure ML
    python train_azure.py --data_path ${{inputs.dataset}}
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from torchvision.models import resnet18
from src.data.datasets import CloudHoleDataset

try:
    from azureml.core import Run

    AZUREML_AVAILABLE = True
except ImportError:
    AZUREML_AVAILABLE = False
    print("Warning: azureml-core not installed. Running in local mode.")


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    patience,
    min_delta,
    save_dir,
    run=None,
):
    """Complete training loop with early stopping"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_roc_auc": [],
    }

    best_val_f1 = 0.0
    epochs_without_improvement = 0
    best_epoch = 0

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Patience: {patience}")
    print(f"Min Delta: {min_delta}")
    print("=" * 80 + "\n")

    for epoch in range(epochs):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, run
        )

        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, run)

        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])

        # Check for improvement
        # FIXED: More explicit improvement check
        current_val_f1 = val_metrics["f1"]
        improvement = current_val_f1 - best_val_f1

        if improvement > min_delta:
            # Significant improvement detected
            best_val_f1 = current_val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "history": history,
                },
                save_dir / "best_model.pth",
            )

            print(
                f"""
                Best model saved (F1: {best_val_f1:.4f}
                improvement: +{improvement:.4f})
            """
            )
        else:
            epochs_without_improvement += 1
            print(f"""
                No improvement (current F1: {current_val_f1:.4f},
                best F1: {best_val_f1:.4f}, delta: {improvement:.4f})
            """)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "history": history,
                },
                save_dir / f"checkpoint_epoch_{epoch+1}.pth",
            )
            print(f"✓ Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")

        print(f"Epochs without improvement: {epochs_without_improvement}/{patience}\n")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f'\n{"="*80}')
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
            print(f'{"="*80}\n')
            break

    # Save final model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        save_dir / "final_model.pth",
    )

    # Save training history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n✓ Training complete!")
    print(f'✓ Best model saved to: {save_dir / "best_model.pth"}')
    print(f'✓ Best validation F1: {best_val_f1:.4f} (epoch {best_epoch+1})')

    return history


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Cloud Hole Detection Model"
    )

    # Data paths
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to gold dataset (e.g., data/gold/datasets/v1.0_baseline)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for models and logs",
    )

    # Hyperparameters
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Minimum improvement for early stopping",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )

    # Model configuration
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone and train only classification head",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use data augmentation during training"
    )

    return parser.parse_args()


def train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        epoch,
        run=None
):
    """Train model for one epoch"""
    model.train()

    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Calculate metrics
    avg_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    if len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs)
    else:
        roc_auc = 0.0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }

    # Log to Azure ML
    if run and AZUREML_AVAILABLE:
        run.log("train_loss", avg_loss)
        run.log("train_accuracy", accuracy)
        run.log("train_f1", f1)
        run.log("train_roc_auc", roc_auc)

    print(
        f"\nTrain Metrics - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
        f"F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
    )

    return metrics


def validate(model, val_loader, criterion, device, epoch, run=None):
    """Validate model on validation set"""
    model.eval()

    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []

    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Calculate metrics
    avg_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    if len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs)
    else:
        roc_auc = 0.0

    cm = confusion_matrix(all_labels, all_predictions)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
    }

    # Log to Azure ML
    if run and AZUREML_AVAILABLE:
        run.log("val_loss", avg_loss)
        run.log("val_accuracy", accuracy)
        run.log("val_f1", f1)
        run.log("val_roc_auc", roc_auc)

    print(
        f"\nVal Metrics - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, "
        f"F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
    )
    print(f"Confusion Matrix:\n{cm}")

    return metrics


def test(model, test_loader, criterion, device, save_path=None, run=None):
    """Test model on test set"""
    model.eval()

    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probs = []

    print("\n" + "=" * 80)
    print("TESTING MODEL")
    print("=" * 80)

    pbar = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Calculate metrics
    avg_loss = running_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    if len(np.unique(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs)
    else:
        roc_auc = 0.0

    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Loss:        {avg_loss:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                 0      1")
    print(f"Actual  0     {tn:4d}  {fp:4d}")
    print(f"        1     {fn:4d}  {tp:4d}")
    print("=" * 80 + "\n")

    # Log to Azure ML
    if run and AZUREML_AVAILABLE:
        run.log("test_loss", avg_loss)
        run.log("test_accuracy", accuracy)
        run.log("test_precision", precision)
        run.log("test_recall", recall)
        run.log("test_f1", f1)
        run.log("test_roc_auc", roc_auc)
        run.log("test_specificity", specificity)

    # Save results
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        metrics_to_save = {k: v for k, v in metrics.items()}
        with open(save_path, "w") as f:
            json.dump(metrics_to_save, f, indent=2)

        print(f"✓ Test results saved to: {save_path}")

    return metrics


def early_stopping(val_loss, best_val_loss, counter, patience):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")

    return best_val_loss, counter


def main():
    """Main training function"""

    # Parse arguments
    args = parse_args()

    # Get Azure ML run context (if available)
    if AZUREML_AVAILABLE:
        run = Run.get_context()
        print("Running in Azure ML")
    else:
        run = None
        print("Running locally")

    print("\n" + "=" * 80)
    print("CLOUD HOLE DETECTION - TRAINING")
    print("=" * 80 + "\n")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Create output directories
    output_dir = Path(args.output_dir)
    model_save_path = output_dir / "models"
    logs_path = output_dir / "logs"
    model_save_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading datasets...\n")

    data_path = Path(args.data_path)

    train_dataset = CloudHoleDataset(
        gold_zarr_path=str(data_path / "train" / "data.zarr"),
        augment=args.augment,
        pretrained=True,
        model="resnet18",
    )

    val_dataset = CloudHoleDataset(
        gold_zarr_path=str(data_path / "validation" / "data.zarr"),
        augment=False,
        pretrained=True,
        model="resnet18",
    )

    test_dataset = CloudHoleDataset(
        gold_zarr_path=str(data_path / "test" / "data.zarr"),
        augment=False,
        pretrained=True,
        model="resnet18",
    )

    print("\n✓ Datasets loaded")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples\n")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Create model
    print("Creating model...\n")
    model = resnet18(pretrained=True)

    if args.freeze_backbone:
        # Freeze all layers
        print("Freezing backbone layers...")
        for param in model.parameters():
            param.requires_grad = False

        # Replace and unfreeze classification head
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        print("✓ Backbone frozen")
        print("✓ Classification head (fc layer) unfrozen\n")

        # Verify trainable parameters
        trainable_params = sum(
            p.numel() for p in model.parameters() 
            if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%\n")

        # Optimizer - only trainable parameters
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.learning_rate
        )
    else:
        # Train entire model
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        print("✓ Training entire model (no frozen layers)\n")

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Log hyperparameters to Azure ML
    if run and AZUREML_AVAILABLE:
        run.log("batch_size", args.batch_size)
        run.log("learning_rate", args.learning_rate)
        run.log("epochs", args.epochs)
        run.log("freeze_backbone", args.freeze_backbone)
        run.log("augment", args.augment)

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        save_dir=model_save_path,
        run=run,
    )

    # Load best model for testing
    print("\nLoading best model for testing...")
    checkpoint = torch.load(model_save_path / "best_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f'✓ Loaded best model from epoch {checkpoint["epoch"]+1}')

    # Test
    test_metrics = test(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        save_path=logs_path / "test_results.json",
        run=run,
    )

    # Register model to Azure ML (if available)
    if run and AZUREML_AVAILABLE and hasattr(run, "experiment"):
        print("\nRegistering model to Azure ML...")
        try:
            run.upload_file(
                name="outputs/best_model.pth",
                path_or_stream=str(model_save_path / "best_model.pth"),
            )

            # Register the model
            run.register_model(
                model_name="cloud_hole_detector",
                model_path="outputs/best_model.pth",
                tags={
                    "test_f1": test_metrics["f1"],
                    "test_accuracy": test_metrics["accuracy"],
                    "freeze_backbone": args.freeze_backbone,
                },
                properties={
                    "test_f1": test_metrics["f1"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_roc_auc": test_metrics["roc_auc"],
                },
            )
            print("✓ Model registered to Azure ML")
        except Exception as e:
            print(f"⚠ Could not register model: {e}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best model: {model_save_path}/best_model.pth")
    print(f"Test results: {logs_path}/test_results.json")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
