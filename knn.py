import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # Transpose the matrix so that each row is a question. This way,
    # KNNImputer finds similar questions instead of similar students.
    mat = nbrs.fit_transform(matrix.T)
    # Transpose back so the matrix is (num_students, num_questions) again.
    mat = mat.T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_values = [1, 6, 11, 16, 21, 26]

    # ---------- User-based KNN ----------
    user_accs = []
    for k in k_values:
        print("User-based KNN with k = {}".format(k))
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        user_accs.append(acc)

    # ---------- Item-based KNN ----------
    item_accs = []
    for k in k_values:
        print("Item-based KNN with k = {}".format(k))
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        item_accs.append(acc)

    # ---------- Plot validation accuracy vs k ----------
    plt.plot(k_values, user_accs, marker="o", label="User-based")
    plt.plot(k_values, item_accs, marker="o", label="Item-based")
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title("KNN Validation Accuracy vs k")
    plt.legend()
    plt.savefig("knn_accuracy.png")
    plt.show()

    # ---------- Pick the best k and report test accuracy ----------
    best_k_user = k_values[user_accs.index(max(user_accs))]
    best_k_item = k_values[item_accs.index(max(item_accs))]

    print("Best k for user-based KNN: {}".format(best_k_user))
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    print("Test Accuracy (user-based, k = {}): {}".format(best_k_user, test_acc_user))

    print("Best k for item-based KNN: {}".format(best_k_item))
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)
    print("Test Accuracy (item-based, k = {}): {}".format(best_k_item, test_acc_item))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
