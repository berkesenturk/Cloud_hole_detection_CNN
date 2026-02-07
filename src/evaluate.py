import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit
from src.training.train import train_model, validate_model
from src.utils import early_stopping, plot_training_curves


def validation_of_pretrained_model(
    batch_size,
    scheduler,
    num_epochs,
    num_splits,
    patience,
    model,
    dataset,
    criterion,
    optimizer
):
    """
    Description:
        Perform time-series cross-validation on the pretrained model
        using the given parameters.

    Parameters:
        batch_size: int
            The batch size for training.
        learning_rate: float
            The learning rate for training.
        num_epochs: int
            The number of epochs to train for.
        num_splits: int
            The number of splits for time-series cross-validation.
        patience: int
            The number of epochs to wait before stopping if no
            improvement.

    """
    tscv = TimeSeriesSplit(n_splits=num_splits)
    fold_metrics = []
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(
        tscv.split(range(len(dataset)))
    ):
        print(f"Fold {fold + 1}/{num_splits}")

        print("\tTRAIN indices:", train_idx)
        print("\tVALIDATION indices:", val_idx)

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False
        )
        # train_labels = [train_subset.dataset.targets[i]
        #                 for i in train_subset.indices]
        # val_labels = [val_subset.dataset.targets[i]
        #               for i in val_subset.indices]

        best_val_loss = float('inf')
        counter = 0

        # plot_class_distribution(train_labels,
        #                         dataset_name="Training Set")
        # plot_class_distribution(val_labels,
        #                         dataset_name="Validation Set")

        for epoch in range(num_epochs):
            train_loss, train_accuracy = train_model(
                model, train_loader, criterion, optimizer
            )
            val_loss, val_accuracy = validate_model(
                model, val_loader, criterion
            )

            scheduler.step(val_loss)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(
                f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, "
                f"Accuracy: {val_accuracy:.4f}%"
            )

            best_val_loss, counter = early_stopping(
                val_loss, best_val_loss, counter, patience
            )

            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")

                plot_training_curves(
                    train_losses, val_losses,
                    train_accuracies, val_accuracies, fold
                )
                break

            # Save the best model
            if val_loss == best_val_loss:
                torch.save({
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, f'best_model_fold_{fold + 1}.pth')

        plot_training_curves(
            train_losses, val_losses,
            train_accuracies, val_accuracies, fold
        )

        fold_metrics.append((val_loss, val_accuracy))

    avg_val_loss = np.mean([metrics[0] for metrics in fold_metrics])
    avg_val_accuracy = np.mean([metrics[1] for metrics in fold_metrics])
    folds = range(1, num_splits + 1)
    val_loss_per_fold = [metrics[0] for metrics in fold_metrics]
    val_acc_per_fold = [metrics[1] for metrics in fold_metrics]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(folds, val_loss_per_fold, color='skyblue')
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Across Folds")

    plt.subplot(1, 2, 2)
    plt.bar(folds, val_acc_per_fold, color='lightgreen')
    plt.xlabel("Fold")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy Across Folds")

    plt.tight_layout()
    plt.show()

    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%")

    return (train_losses, val_losses, train_accuracies,
            val_accuracies, fold_metrics, model)
