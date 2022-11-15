import numpy as np
import csv


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Function from the given helpers.py
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def MSE_loss(tx, w, y):
    """Compute the loss by mean square error

    Args:
        tx : features
        w : weights
        y : labels

    Returns:
        MSE loss
    """
    err = y - np.dot(tx, w)
    loss = np.mean(np.square(err)) / 2
    return loss


def MSE_gradient(tx, w, y):
    """Compute gradient for MSE loss function

    Args:
        tx : features
        w : weights
        y : labels

    Returns:
        gradient
    """
    err = y - np.dot(tx, w)
    grad = -np.dot(tx.T, err) / len(err)
    return grad


def sigmoid(x):
    """Sigmoid function

    Args:
        x : variable on which apply the function

    Returns:
        sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def log_loss(y, y_hat):
    """Compute loss for logistic regression

    Args:
        y_hat : predictions
        y : true labels

    Returns:
        loss value
    """
    loss = (-1 / len(y)) * (y.T.dot(np.log(y_hat)) + (1 - y).T.dot(np.log(1 - y_hat)))
    return np.squeeze(loss)


def log_gradient(tx, w, y, reg=False, lambda_=0.1):
    """Compute gradient for logistic regression

    Args:
        tx : features
        w : weights
        y : labels
        reg (bool, optional): If true, add regularization. Defaults to False.
        lambda_ (int, optional): hyperparameters for regularization. Defaults to 0.

    Returns:
        gradient, loss value
    """
    y_hat = sigmoid(tx.dot(w))

    # Compute gradient with regularization
    if reg:
        return 1 / len(y) * np.dot(tx.T, (y_hat - y)) + 2 * lambda_ * w

    return 1 / len(y) * np.dot(tx.T, (y_hat - y))


def load_split_data(path, one_hot=True):
    """Load data and split according to the jet number

    Args:
        path (string): path to dataset
        one_hot (Bool): select label encoding (0,1 or -1,1). Default True (0,1)

    Returns:
        numpy array: the 3 dataset corresponding to different pri_jet with INDEX and PREDICTION as first and second columns
    """
    # Load all data from .csv file
    names = np.genfromtxt(path, delimiter=",", dtype=str, max_rows=1)

    if one_hot:
        conv = lambda x: 0.0 if (x == b"b") else 1.0
    else:
        conv = lambda x: -1.0 if (x == b"b") else 1.0

    y = np.expand_dims(
        np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            dtype=str,
            usecols=[1],
            converters={1: conv},
        ),
        axis=1,
    )
    idx = np.expand_dims(
        np.genfromtxt(path, delimiter=",", skip_header=1, usecols=[0]), axis=1
    )
    x = np.genfromtxt(path, delimiter=",", skip_header=1)[:, 2:]

    # Reconcatenante data
    data = np.concatenate((idx, np.concatenate((y, x), axis=1)), axis=1)

    # Convert undefined values (-999.) to nan
    data[data == -999.0] = float("nan")

    # Get index of PRI_jet_num
    jet_idx = np.where(names == "PRI_jet_num")[0][0]

    # Separate dataset according to PRI_jet_num
    data_0 = np.delete(data[data[:, jet_idx] == 0, :], jet_idx, axis=1)
    data_1 = np.delete(data[data[:, jet_idx] == 1, :], jet_idx, axis=1)
    data_2 = np.delete(data[data[:, jet_idx] == 2, :], jet_idx, axis=1)
    data_3 = np.delete(data[data[:, jet_idx] == 3, :], jet_idx, axis=1)

    # Concatenate data with PRI_jet_num = 2 or 3 as no features is undefined for both
    data_2_3 = np.concatenate((data_2, data_3), axis=0)

    # Remove features where all values are undefined
    data_0 = data_0[:, ~np.isnan(data_0).all(axis=0)]
    data_1 = data_1[:, ~np.isnan(data_1).all(axis=0)]

    # Remove last features as all values are 0
    data_0 = np.delete(data_0, -1, axis=1)

    return data_0, data_1, data_2_3


def std_data(data):

    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)  # standardize
    return data


def min_max_scale_data(data):

    data = data / np.max(data, axis=0)  # standardize
    return data


def nan_to_mean(data):
    """Replace all nan value by the mean of the corresponding column

    Args:
        data (numpy array): dataset with undefined values (no colummns should be all nan)

    Returns:
        dataset with nan values replaced
    """
    # Compute mean for each columns
    means = np.nanmean(data, axis=0)
    # Get index of nan values
    idx = np.where(np.isnan(data))

    data[idx] = np.take(means, idx[1])

    return data


def arcsinh_transform(data):
    """
    Compute the arcsinh of each datapoints

    Args:
        data (numpy array): Data to be transformed

    Returns:
        numpy array: transformed data
    """
    return np.arcsinh(data)


def process_data(data_0, data_1, data_2_3, deg=[2]):
    """Data processing pipeline

    Args:
        data_0 (numpy array): Dataset 0
        data_1 (numpy array): Dataset 1
        data_2_3 (numpy array): Dataset 2,3
        deg (numpy array): List of degree for the polynomial expansion

    Returns:
        numpy array: 3 processed dataset with INDEX and PREDICTION as first and second columns
    """
    data_0 = nan_to_mean(data_0)
    data_1 = nan_to_mean(data_1)
    data_2_3 = nan_to_mean(data_2_3)

    data_0 = poly_expansion(data_0, deg)
    data_1 = poly_expansion(data_1, deg)
    data_2_3 = poly_expansion(data_2_3, deg)

    data_0[:, 2:] = std_data(arcsinh_transform(data_0[:, 2:]))
    data_1[:, 2:] = std_data(arcsinh_transform(data_1[:, 2:]))
    data_2_3[:, 2:] = std_data(arcsinh_transform(data_2_3[:, 2:]))

    return data_0, data_1, data_2_3


def split_val_train(data, seed):
    np.random.seed(seed)
    split_idx = int(0.9 * data.shape[0])
    np.random.shuffle(data)
    train_data = data[:split_idx, :]
    val_data = data[split_idx:, :]
    return train_data, val_data


def poly_expansion(data, deg=[2]):
    """Generate a polynomial expansion of data

    Args:
        data (numpy array): dataset to be expanded
        deg (numpy array, optional): list of degree to add. Defaults to [2].

    Returns:
        dataset concatenated with the selected degree of polynomial expansion
    """
    data_poly = data.copy()
    for d in deg:
        data_poly = np.concatenate((data_poly, np.power(data[:, 2:], d)), axis=1)
    return data_poly


def count_params(model):
    """
    Counts the number of parameters in a model.

    Args:
        model (class NN)

    Returns:
        int: Nb of parameters
    """
    N_params = 0
    for i in range(len(model.architecture)):
        N_params += model.params[i]["W"].shape[0] * model.params[i]["W"].shape[1]
        N_params += model.params[i]["b"].shape[0]
    return N_params


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
