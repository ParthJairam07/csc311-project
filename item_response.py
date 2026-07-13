from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Apply sigmoid function."""
    return 1/(1 + np.exp(-x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.0
    for i in range(len(data["is_correct"])):
        u_id = data["user_id"][i]
        q_id = data["question_id"][i]
        is_c = data["is_correct"][i]
        x = theta[u_id] - beta[q_id]

        log_lklihood += is_c*x - np.logaddexp(0, x)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    users = np.asarray(data["user_id"], dtype=int)
    questions = np.asarray(data["question_id"], dtype=int)
    correct = np.asarray(data["is_correct"], dtype=float)
    # theta
    x = theta[users] - beta[questions]
    probs = sigmoid(x)

    theta_grad = np.zeros_like(theta, dtype=float)
    np.add.at(theta_grad, users, correct-probs)
    theta = theta + lr*theta_grad

    #beta
    x = theta[users] - beta[questions]
    probs = sigmoid(x)

    beta_grad = np.zeros_like(beta, dtype=float)
    np.add.at(beta_grad, questions, probs-correct)
    beta = beta + lr*beta_grad

    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    num_users = max(max(data["user_id"]), max(val_data["user_id"]))+1
    num_questions = max(max(data["question_id"]), max(val_data["question_id"]))+1
    theta = np.zeros(num_users)
    beta = np.zeros(num_questions)

    val_acc_lst = []
    train_nll_lst = []
    val_nll_lst = []

    for i in range(iterations):
        #alternating gradient descent
        theta, beta = update_theta_beta(data,lr,theta, beta)

        val_acc = evaluate(val_data, theta, beta)
        train_nll = neg_log_likelihood(data, theta, beta)
        val_nll = neg_log_likelihood(val_data, theta, beta)

        val_acc_lst.append(val_acc)
        train_nll_lst.append(train_nll)
        val_nll_lst.append(val_nll)

        print(
            "Iteration: {} \t Validation accuracy: {:.4f} \t Train nll: {:.4f} \t Validation nll: {:.4f}".format(
                i+1,
                val_acc,
                train_nll,
                val_nll
            )
        )

    return theta, beta, val_acc_lst, train_nll_lst, val_nll_lst


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) / len(data["is_correct"])


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################                                                            #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.0025
    iterations = 49
    theta, beta, val_acc_lst, train_nll_lst, val_nll_lst = irt(
        train_data,
        val_data,
        lr=lr,
        iterations=iterations
      )

    final_val_acc = evaluate(val_data, theta, beta)
    final_test_acc= evaluate(test_data, theta, beta)
    print("Learning rate: ", lr)
    print("Number of iterations: ", iterations)
    print("Final validation accuracy: ",final_val_acc)
    print("Final test accuracy: ", final_test_acc)

    num_iterations = range(1, iterations+1)
    plt.plot(
        num_iterations,
        train_nll_lst,
        label="Training nll"
    )
    plt.plot(
        num_iterations,
        val_nll_lst,
        label="Validation nll"
    )
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("IRT Training and Validation NLL")
    plt.legend()
    plt.show()


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################                                                         #
    # Implement part (d)                                                #
    #####################################################################
    easy_q = np.argmin(beta)
    hard_q = np.argmax(beta)

    sorted_qs = np.argsort(beta)
    med_q = sorted_qs[len(sorted_qs)//2]

    qs = [easy_q, med_q, hard_q]

    theta_vals = np.linspace(
        np.min(theta)-1,
        np.max(theta)+1,
        250
    )

    for q in qs:
        probs = sigmoid(theta_vals - beta[q])
        plt.plot(
            theta_vals,
            probs,
            label="Question {} (beta={:.2f})".format(q, beta[q])
        )

    plt.xlabel(r"$\theta$ (Student Ability)")
    plt.ylabel("Probability of Correct Response")
    plt.title("Probability of Correct Response vs Student Ability")
    plt.legend()
    plt.tight_layout()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
