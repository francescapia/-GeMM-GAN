import math
from typing import Union

import numpy as np
import torch
import sklearn 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from typing import Optional
from functools import partial
import math
import ot
SEED = 42

def compute_distribution_distances(pred: torch.Tensor, true: Union[torch.Tensor, list]):
    """
    Computes distances between predicted and true distributions.

    Args:
        pred (torch.Tensor): Predicted tensor of shape [batch, times, dims].
        true (Union[torch.Tensor, list]): True tensor of shape [batch, times, dims] or list of tensors of length times.

    Returns:
        dict: Dictionary containing the computed distribution distances.
    """
    min_size = min(pred.shape[0], true.shape[0])
    
    names = [
        "1-Wasserstein",
        "2-Wasserstein",
        "Linear_MMD",
        "Poly_MMD"
    ]
    dists = []
    to_return = []
    w1 = wasserstein(pred, true, power=1)
    w2 = wasserstein(pred, true, power=2)
    pred_4_mmd = pred[:min_size]
    true_4_mmd = true[:min_size]
    mmd_linear = linear_mmd2(pred_4_mmd, true_4_mmd).item()
    mmd_poly = poly_mmd2(pred_4_mmd, true_4_mmd).item()
    dists.append((w1, w2, mmd_linear, mmd_poly))

    to_return.extend(np.array(dists).mean(axis=0))
    return dict(zip(names, to_return))


def compute_pairwise_distance(data_x, data_y=None):
    """
    Computes pairwise distances between two datasets.

    Args:
        data_x (np.ndarray): Array of shape [N, feature_dim].
        data_y (np.ndarray, optional): Array of shape [N, feature_dim]. Defaults to None.

    Returns:
        np.ndarray: Array of shape [N, N] containing pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='l1', n_jobs=8)
    return dists


def get_kth_value(unsorted, k, axis=-1):
    """
    Gets the k-th smallest value along the specified axis.

    Args:
        unsorted (np.ndarray): Unsorted array of any dimensionality.
        k (int): The k-th index.

    Returns:
        np.ndarray: Array containing k-th smallest values along the specified axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Computes distances to the k-th nearest neighbours.

    Args:
        input_features (np.ndarray): Array of shape [N, feature_dim].
        nearest_k (int): The number of nearest neighbours.

    Returns:
        np.ndarray: Distances to the k-th nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features (np.ndarray): Array of real features of shape [N, feature_dim].
        fake_features (np.ndarray): Array of fake features of shape [N, feature_dim].
        nearest_k (int): Number of nearest neighbours.

    Returns:
        dict: Dictionary containing precision, recall, density, and coverage.
    """
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()
    
    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)


# def compute_knn_real_fake(X_real, X_fake, n_neighbors=5):
#     """
#     Computes F1 score using k-nearest neighbours classifier for real and fake data.

#     Args:
#         X_real (np.ndarray): Array of real features.
#         X_fake (np.ndarray): Array of fake features.
#         n_neighbors (int, optional): Number of neighbours. Defaults to 5.

#     Returns:
#         float: F1 score.
#     """
#     X = np.concatenate((X_real, X_fake), axis=0)
#     y = np.concatenate((np.ones(len(X_real)), np.zeros(len(X_fake))), axis=0)

#     # Initialize KNN classifier
#     # knn = RandomForestClassifier()
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors)

#     # Train the classifier
#     knn.fit(X, y)

#     # Evaluate the classifier
#     y_pred = knn.predict(X)
#     auc = f1_score(y, y_pred, average="macro")
#     return auc

def compute_knn_real_fake(X_real, X_fake, 
                            X_real_test, X_fake_test, n_neighbors=5):
    """
    Computes F1 score using k-nearest neighbours classifier for real and fake data.

    Args:
        X_real (np.ndarray): Array of real features.
        X_fake (np.ndarray): Array of fake features.
        n_neighbors (int, optional): Number of neighbours. Defaults to 5.

    Returns:
        float: F1 score.
    """
    X = np.concatenate((X_real, X_fake), axis=0)
    y = np.concatenate((np.ones(len(X_real)), np.zeros(len(X_fake))), axis=0)
    X_test = np.concatenate((X_real_test, X_fake_test), axis=0)
    y_test = np.concatenate((np.ones(len(X_real_test)), np.zeros(len(X_fake_test))), axis=0)
    # Initialize KNN classifier
    # knn = RandomForestClassifier()
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the classifier
    knn.fit(X, y)

    # Evaluate the classifier
    y_pred = knn.predict(X_test)
    auc = f1_score(y_test, y_pred, average="macro")
    auc_2 = accuracy_score(y_test, y_pred)
    return [auc, auc_2]

model_args = {
            "random_state": SEED,
            "n_jobs": -1,
            "max_iter": 10000,
            "penalty": 'l2',
        }

def compute_logistic_real_fake(X_real, X_fake, 
                            X_real_test, X_fake_test, n_neighbors=5):
    """
    Computes F1 score using k-nearest neighbours classifier for real and fake data.

    Args:
        X_real (np.ndarray): Array of real features.
        X_fake (np.ndarray): Array of fake features.
        n_neighbors (int, optional): Number of neighbours. Defaults to 5.

    Returns:
        float: F1 score.
    """
    X = np.concatenate((X_real, X_fake), axis=0)
    y = np.concatenate((np.ones(len(X_real)), np.zeros(len(X_fake))), axis=0)
    X_test = np.concatenate((X_real_test, X_fake_test), axis=0)
    y_test = np.concatenate((np.ones(len(X_real_test)), np.zeros(len(X_fake_test))), axis=0)
    # Initialize KNN classifier
    # knn = RandomForestClassifier()
    knn = LogisticRegression(**model_args)

    # Train the classifier
    knn.fit(X, y)

    # Evaluate the classifier
    y_pred = knn.predict(X_test)
    auc = f1_score(y_test, y_pred, average="macro")
    auc_2 = accuracy_score(y_test, y_pred)
    return [auc, auc_2]


def compute_random_forest_real_fake(X_real, X_fake, 
                            X_real_test, X_fake_test, n_neighbors=5):
    """
    Computes F1 score using k-nearest neighbours classifier for real and fake data.

    Args:
        X_real (np.ndarray): Array of real features.
        X_fake (np.ndarray): Array of fake features.
        n_neighbors (int, optional): Number of neighbours. Defaults to 5.

    Returns:
        float: F1 score.
    """
    X = np.concatenate((X_real, X_fake), axis=0)
    y = np.concatenate((np.ones(len(X_real)), np.zeros(len(X_fake))), axis=0)
    X_test = np.concatenate((X_real_test, X_fake_test), axis=0)
    y_test = np.concatenate((np.ones(len(X_real_test)), np.zeros(len(X_fake_test))), axis=0)
    # Initialize KNN classifier
    knn = RandomForestClassifier()
    #knn = LogisticRegression(**model_args)

    # Train the classifier
    knn.fit(X, y)

    # Evaluate the classifier
    y_pred = knn.predict(X_test)
    auc = f1_score(y_test, y_pred, average="macro")
    auc_2 = accuracy_score(y_test, y_pred)
    return [auc, auc_2]

def train_knn_real_data(adata_real, category_field, use_pca, n_neighbors=5):
    """
    Trains a k-nearest neighbours classifier on real data.

    Args:
        adata_real (AnnData): Annotated Data object containing real data.
        category_field (str): The category field to be used as the target variable.
        use_pca (bool): Whether to use PCA-transformed data.
        n_neighbors (int, optional): Number of neighbours. Defaults to 5.

    Returns:
        KNeighborsClassifier: Trained KNN classifier.
    """
    if not use_pca:
        X = adata_real.X  # Features
    else:
        X = adata_real.obsm["X_pca"] 
        
    y = adata_real.obs[category_field]  # Target variable

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # You can adjust the number of neighbors
    # knn = RandomForestClassifier()    

    # Fit the classifier to the training data
    knn.fit(X, y)
    return knn


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """
    Compute the Wasserstein distance between two distributions.

    Args:
        x0 (torch.Tensor): The first distribution.
        x1 (torch.Tensor): The second distribution.
        method (Optional[str], optional): The method for computing Wasserstein distance.
            Options are "exact", "sinkhorn". Defaults to None.
        reg (float, optional): Regularization parameter for the Sinkhorn method. Defaults to 0.05.
        power (int, optional): Power for the distance computation, can be 1 or 2. Defaults to 2.
        **kwargs: Additional keyword arguments.

    Raises:
        ValueError: If an unknown method is provided.

    Returns:
        float: The computed Wasserstein distance.
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = ot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(ot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = ot.unif(x0.shape[0]), ot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret



min_var_est = 1e-8


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert X.size(0) == Y.size(0)
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(
        K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased
    )
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)  # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)  # (m,)
        diag_Y = torch.diag(K_YY)  # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)  # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX**2).sum() - sum_diag2_X  # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY**2).sum() - sum_diag2_Y  # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum = (K_XY**2).sum()  # \| K_{XY} \|_F^2

    if biased:
        mmd2 = (
            (Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m)
        )
    else:
        mmd2 = Kt_XX_sum / (m * (m - 1)) + Kt_YY_sum / (m * (m - 1)) - 2.0 * K_XY_sum / (m * m)

    var_est = (
        2.0
        / (m**2 * (m - 1.0) ** 2)
        * (
            2 * Kt_XX_sums.dot(Kt_XX_sums)
            - Kt_XX_2_sum
            + 2 * Kt_YY_sums.dot(Kt_YY_sums)
            - Kt_YY_2_sum
        )
        - (4.0 * m - 6.0) / (m**3 * (m - 1.0) ** 3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0
        * (m - 2.0)
        / (m**3 * (m - 1.0) ** 2)
        * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0 * (m - 3.0) / (m**3 * (m - 1.0) ** 2) * (K_XY_2_sum)
        - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0
        / (m**3 * (m - 1.0))
        * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0)
        )
    )
    return mmd2, var_est