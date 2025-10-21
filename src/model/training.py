import os
from itertools import chain
from typing import Tuple, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.losses import loss_from_task_config
from util.config import RunConfig, MLConfig
from util.torch_utils import init_weights, TorchStandardScaler


def train(model: nn.Module,
          data_loader: DataLoader,
          y_std: torch.Tensor,
          loss_fn: nn.Module,
          optimizer: torch.optim.Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs one training iteration on a given model and return loss and MAE on the training set after the iteration.

    :param model: The model to train.
    :param data_loader: The training data loader.
    :param y_std: The standard deviation used for scaling the target variable to properly compute MAE.
    :param loss_fn: The loss function to use.
    :param optimizer: The optimizer to use for the optimization.
    :return: Tuple of tensors containing loss and MAE after the training step.
    """
    model.train()
    for X, y in data_loader:
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred[0], y, pred[1])
        loss.backward()
        optimizer.step()

    return evaluate(model, data_loader, y_std, loss_fn)


def evaluate(model: nn.Module,
             test_loader: DataLoader,
             y_std: torch.Tensor,
             loss_fn: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates a model with a given test set and returns the loss and MAE.

    :param model: The model to evaluate.
    :param test_loader: The test data to evaluate.
    :param y_std: The standard deviation used for scaling the target variable to properly compute MAE.
    :param loss_fn: The loss function to use.
    :return: Tuple of tensors containing loss and MAE.
    """
    num_batches = len(test_loader)
    running_loss = torch.zeros(1, device=y_std.device)
    running_mae = torch.zeros_like(y_std, device=y_std.device)

    with torch.inference_mode():
        model.eval()
        for X, y in test_loader:
            pred = model(X)
            running_loss += loss_fn(pred[0], y, pred[1])
            running_mae += torch.mean(torch.abs(pred[0] - y) * y_std, dim=0)

    running_loss /= num_batches
    running_mae /= num_batches
    return running_loss, running_mae


def train_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, y_scaler: TorchStandardScaler,
               criterion: nn.Module, optimizer: torch.optim.Optimizer, task_config: MLConfig,
               writer: Optional[SummaryWriter] = None):
    """
    Trains a given model with the given data according to a run and task config.

    :param model: PyTorch model to train.
    :param train_loader: DataLoader containing the training data.
    :param test_loader: DataLoader containing the test data.
    :param y_scaler: TorchStandardScaler used to scale the target variable(s).
    :param criterion: Loss to use for training.
    :param optimizer: Optimizer to use for training.
    :param task_config: TaskConfig defining model name and other training details.
    :param writer: Optional SummaryWriter for TensorBoard usage.
    """
    for epoch in tqdm(range(task_config.epochs)):
        train_loss, train_mae = train(model, train_loader, y_scaler.std, criterion, optimizer)

        if writer is not None:
            writer.add_scalar(f"{task_config.model_name}/loss_train", train_loss.item(), epoch)
            for i in range(train_mae.shape[0]):
                writer.add_scalar(f"{task_config.model_name}/mae_train_{i}", train_mae[i].item(), epoch)

            test_loss, test_mae = evaluate(model, test_loader, y_scaler.std, criterion)
            writer.add_scalar(f"{task_config.model_name}/loss_test", test_loss.item(), epoch)
            for i in range(test_mae.shape[0]):
                writer.add_scalar(f"{task_config.model_name}/mae_test_{i}", test_mae[i].item(), epoch)

            if hasattr(criterion, "sigma"):
                for i in range(criterion.sigma.shape[0]):
                    writer.add_scalar(f"{task_config.model_name}/sigma_{i}", criterion.sigma[i].item(), epoch)


def load_or_train_model(workdir: str, model: nn.Module, run_config: RunConfig, task_config: MLConfig,
                        train_loader: DataLoader, test_loader: DataLoader, y_scaler: TorchStandardScaler,
                        writer: Optional[SummaryWriter] = None) -> nn.Module:
    """
    Checks for existence of a "model.pt" file in a given working directory and loads the parameters into the given
    PyTorch model if it exists. Otherwise, the given model is trained by calling train_loop and then saved to
    "model.pt" in the working directory.

    Task config is used to create the loss and optimizer. Other parameters are passed on to train_loop.

    :param workdir: Working directory to use.
    :param model: PyTorch model to load the state dict for, or train.
    :param run_config: RunConfig to use for training.
    :param task_config: TaskConfig to use for training.
    :param train_loader: Training data used for training.
    :param test_loader: Test data used for evaluation.
    :param y_scaler: TorchStandardScaler that was used to scale the target variable.
    :param writer: Optional SummaryWriter for TensorBoard usage during training.
    :return: Trained model.
    """
    model_path = os.path.join(workdir, f"model.pt")
    if os.path.exists(model_path):
        print("Model already exists, skipping training")
        model.load_state_dict(torch.load(model_path, map_location=run_config.device))
    else:
        model.apply(init_weights)

        criterion = loss_from_task_config(task_config)
        optimizer = torch.optim.Adam(list(chain(model.parameters(), criterion.parameters())),
                                     lr=task_config.learning_rate, weight_decay=task_config.weight_decay)

        train_loop(model, train_loader, test_loader, y_scaler, criterion, optimizer, task_config, writer=writer)

        torch.save(model.state_dict(), model_path)

    return model
