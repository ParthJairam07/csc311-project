from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    evaluate,
)
from item_response import irt, sigmoid
import numpy as np


def bootstrap_sample(data):
    """Sample N points with replacement to build one bagging resample."""
    n = len(data["is_correct"])
    idx = np.random.choice(n, n, replace=True)
    # Index all three lists with the SAME idx so rows stay aligned.
    return {
        "user_id": [data["user_id"][i] for i in idx],
        "question_id": [data["question_id"][i] for i in idx],
        "is_correct": [data["is_correct"][i] for i in idx],
    }


def train_models(train_data, val_data, lr, iterations, num_models=3):
    """Train num_models IRT models, each on its own bootstrap resample."""
    models = []
    for m in range(num_models):
        boot_data = bootstrap_sample(train_data)
        # irt returns (theta, beta, val_acc_lst, train_nll_lst, val_nll_lst).
        theta, beta, _, _, _ = irt(boot_data, val_data, lr, iterations)
        models.append((theta, beta))
        print("Finished training base model {}".format(m + 1))
    return models


def predict_probs(data, theta, beta):
    """Predicted probability of correct for each row, from one model."""
    probs = []
    for i in range(len(data["question_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]

        if u < len(theta) and q < len(beta):
            probs.append(sigmoid(theta[u] - beta[q]))
        else:
            probs.append(0.5)
    return np.array(probs)


def ensemble_predict(data, models):
    """Average the predicted probabilities across all base models."""
    prob_sum = np.zeros(len(data["is_correct"]))
    for theta, beta in models:
        prob_sum += predict_probs(data, theta, beta)
    return prob_sum / len(models)


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    # Seed so the bootstrap resamples (and results) are reproducible.
    np.random.seed(311)
    lr = 0.0025
    iterations = 49

    models = train_models(train_data, val_data, lr, iterations, num_models=3)

    val_probs = ensemble_predict(val_data, models)
    test_probs = ensemble_predict(test_data, models)

    val_acc = evaluate(val_data, val_probs)
    test_acc = evaluate(test_data, test_probs)

    print("Ensemble validation accuracy: {:.4f}".format(val_acc))
    print("Ensemble test accuracy: {:.4f}".format(test_acc))


if __name__ == "__main__":
    main()