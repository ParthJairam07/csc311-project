import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch
import matplotlib.pyplot as plt

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)


def load_data(base_path="./data"):
    """Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """

        out = torch.sigmoid(self.g(inputs))   # encode
        out = torch.sigmoid(self.h(out))      # decode

        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.
    

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_costs = []
    valid_accs = []

    for epoch in range(0, num_epoch):
        train_loss = 0.0

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[nan_mask] = output[nan_mask]

            loss = torch.sum((output - target) ** 2.0) + 0.5 * lamb * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_costs.append(train_loss)
        valid_accs.append(valid_acc)
        print(
            "Epoch: {} \tTraining Cost: {:.6f}\t " "Valid Acc: {}".format(
                epoch, train_loss, valid_acc
            )
        )
    return train_costs, valid_accs


def evaluate(model, train_data, valid_data):
    """Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    num_question = train_matrix.shape[1]
    k_values = [10, 50, 100, 200, 500]

    # Set model hyperparameters.
    model = None

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 50
    lamb = 0.0

    k_star, best_acc = None, -1
    best_model, best_hist = None, None
    for k in k_values:
        model = AutoEncoder(num_question, k)
        train_costs, valid_accs = train(
            model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch
        )
        acc = evaluate(model, zero_train_matrix, valid_data)
        print(f"k={k}, valid acc={acc}")
        if acc > best_acc:
            k_star, best_acc = k, acc
            best_model = model
            best_hist = (train_costs, valid_accs)

    print(f"\nSelected k* = {k_star} (validation accuracy = {best_acc:.4f})")

    # part (d)
    train_costs, valid_accs = best_hist
    epochs = range(1, num_epoch + 1)

    plt.switch_backend("Agg")
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train_costs, "b-o", markersize=3, label="Training cost")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training cost (sum of squared error)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax2 = ax1.twinx()
    ax2.plot(epochs, valid_accs, "r-s", markersize=3, label="Validation accuracy")
    ax2.set_ylabel("Validation accuracy", color="r")
    ax2.tick_params(axis="y", labelcolor="r")

    plt.title(f"Training cost & validation accuracy vs epoch (k*={k_star}, lr={lr})")
    fig.tight_layout()
    plt.savefig("nn_k_star_curves.png", dpi=150)
    print("Saved plot to nn_k_star_curves.png")

  
    test_acc = evaluate(best_model, zero_train_matrix, test_data)
    print(f"Final Test Accuracy (k*={k_star}): {test_acc:.4f}")


    # part(e)
    lambda_values = [0.001, 0.01, 0.1, 1.0]

    best_lamb, best_reg_acc = None, -1
    best_reg_model = None
    for lamb in lambda_values:
        model = AutoEncoder(num_question, k_star)
        train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, num_epoch)
        acc = evaluate(model, zero_train_matrix, valid_data)
        print(f"lambda={lamb}, valid acc={acc:.4f}")
        if acc > best_reg_acc:
            best_lamb, best_reg_acc = lamb, acc
            best_reg_model = model

    reg_test_acc = evaluate(best_reg_model, zero_train_matrix, test_data)
    print(f"\nOptimal lambda value* = {best_lamb}")
    print(f"Regularized Validation Accuracy: {best_reg_acc:.4f}")
    print(f"Regularized Test Accuracy: {reg_test_acc:.4f}")


if __name__ == "__main__":
    main()