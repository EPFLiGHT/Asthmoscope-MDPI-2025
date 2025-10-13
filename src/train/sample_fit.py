import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import hmean
from sklearn.metrics import classification_report, roc_auc_score

from src.train.mixup import do_mixup

THRESHOLD_VALUES = [0.1, 0.3, 0.45, 0.5, 0.55, 0.7, 0.9]


def plot_classification_metrics_wrt_thresholds(threshold_values, sensitivities,
                                               specificities, f1_scores, accuracies):
    fig, ax = plt.subplots()
    ax.set_title("Classification metrics wrt different thresholds")

    ax.plot(threshold_values, f1_scores, 'x-', label='F1-score')
    ax.plot(threshold_values, sensitivities, 'x-', label='sensitivity')
    ax.plot(threshold_values, specificities, 'x-', label='specificity')
    ax.plot(threshold_values, accuracies, 'x-', label='accuracy')
    ax.legend(bbox_to_anchor=(1,1))
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metrics score")
    plt.tight_layout()

    return fig, ax


def classification_metrics(samples_df, verbose=True, threshold_values=THRESHOLD_VALUES):
    y_true = samples_df['label']
    output_preds = samples_df['output']

    # 1. AUCs
    ## Overall AUC
    metrics_dict = {"auc_score": roc_auc_score(y_true, output_preds)}

    ## AUC for each position
    for position in samples_df["position"].unique():
        pos_df = samples_df.query("position == @position")
        metrics_dict[f"auc_position_{position}"] = roc_auc_score(pos_df['label'],
                                                                 pos_df['output'])

    ## AUC for each stethoscope
    for stethoscope in samples_df["stethoscope"].unique():
        stetho_df = samples_df.query("stethoscope == @stethoscope")
        metrics_dict[f"auc_stetho_{stethoscope}"] = roc_auc_score(stetho_df['label'],
                                                      stetho_df['output'])

    if verbose:
        # Print AUC results
        print(f"----------")
        print(f"AUC values")
        print(f"----------")
        for metric_name, value in metrics_dict.items():
            print(f"{metric_name:14}: {value:.4f}")

    # 2. Classification metrics for different thresholds
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    for threshold_value in threshold_values:
        y_pred = (output_preds >= threshold_value).astype(int)
        classification_results_dict = classification_report(y_true, y_pred,
                                                                    output_dict=True,
                                                                    zero_division=0)
        sensitivities.append(classification_results_dict['1']['recall'])
        specificities.append(classification_results_dict['0']['recall'])
        f1_scores.append(classification_results_dict['1']['f1-score'])
        accuracies.append(classification_results_dict['accuracy'])

    # Get a plot of the classification metrics
    metrics_wrt_thresholds_fig, ax = plot_classification_metrics_wrt_thresholds(
        threshold_values, sensitivities, specificities, f1_scores, accuracies)
    plt.close()

    # Record the threshold and metrics corresponding to the best F1-score
    best_index = np.argmax(f1_scores)
    metrics_dict.update({
        "threshold": threshold_values[best_index],
        "f1_score": f1_scores[best_index],
        "sensitivity": sensitivities[best_index],
        "specificity": specificities[best_index],
        "accuracy": accuracies[best_index],
    })

    # Record figures
    figures_dict = {'metrics_wrt_thresholds_plot': metrics_wrt_thresholds_fig}

    if verbose:
        # Print classification results for best threshold
        print(f"----------------------------------------------------------------")
        print(f"Classification results for best threshold according to F1-score:")
        print(f"----------------------------------------------------------------")
        print(f"threshold:   {metrics_dict['threshold']}")
        print(f"f1_score:    {metrics_dict['f1_score']:.3f}")
        print(f"sensitivity: {metrics_dict['sensitivity']:.3f}")
        print(f"specificity: {metrics_dict['specificity']:.3f}")
        print(f"accuracy:    {metrics_dict['accuracy']:.3f}")

    return metrics_dict, figures_dict


def train_epoch(model, train_loader, optimizer, criterion, epoch, device, mixup_augmenter=None, scheduler=None, log_interval=10):
    model.train()

    total_batch_loss = 0

    for batch_idx, batch_dict in enumerate(train_loader):  # added
        data = batch_dict["data"].to(device)
        target = batch_dict["target"].to(device)

        optimizer.zero_grad()

        # ➡ Forward pass
        if mixup_augmenter is not None:
            mixup_lambda = mixup_augmenter.get_lambda(batch_size=len(data))

            output_dict = model(data, mixup_lambda=mixup_lambda)
            target = do_mixup(target, mixup_lambda)
        else:
            output_dict = model(data)

        output = output_dict["diagnosis_output"]
        loss = criterion(output, target)
        total_batch_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        if scheduler is not None:  # added
            scheduler.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {:3d} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss = total_batch_loss / len(train_loader)
    return train_loss


def evaluate(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0
    example_spect = []

    val_samples_df = val_loader.dataset.samples_df
    val_samples_df["output"] = 0.

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(val_loader):  # added
            # Load the input features and labels from the val dataset
            sample_idx = batch_dict["sample_idx"]
            data = batch_dict["data"].to(device)
            target = batch_dict["target"].to(device)

            # Make predictions: Pass image data from val dataset, make predictions about class image belongs to
            output_dict = model(data)
            output = output_dict["diagnosis_output"]

            # Compute the loss sum up batch loss
            batch_size = data.shape[0]
            val_loss += (batch_size * criterion(output, target).item())

            # Add sample predictions
            val_samples_df.loc[
                sample_idx.cpu().numpy(), "output"] = output.cpu().numpy()[0, 0]

            # WandB – Log images in your val dataset automatically
            example_spect.append(torch.unsqueeze(data[0], 0))

    # position_df = position_df.reset_index()
    metrics, figures_dict = classification_metrics(val_samples_df, verbose=True)
    val_loss = val_loss / len(val_samples_df)
    metrics["Validation Loss"] = val_loss

    return metrics, figures_dict, example_spect

