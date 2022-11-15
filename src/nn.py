import numpy as np


class NN:
    """
    Numpy implementation of a simple feed-forward neural network.
    """

    def __init__(self, architecture):
        """
        Args:
            architecture (list): List of dict specifiying the architecture of the network.
                                 Each element in the list describes one layer, its input/output sizes and its activation function.
        """
        self.architecture = architecture
        self.params = []  # to store the weights of the model
        self.memory = (
            []
        )  # to store information from the forward pass used in the backward pass and the weights' update
        self.gradients = (
            []
        )  # to store the gradients from the backward pass used in the weights' update

        self.seed = 42

    def init_weights(self):
        """
        Initializes the parameters of the NN.
        Samples each weight from an uniform distribution between [-.1,.1]
        """

        np.random.seed(self.seed)

        for i, layer in enumerate(self.architecture):
            N = layer["input_dim"] * layer["output_dim"]

            W = np.random.uniform(
                -0.1, 0.1, size=(layer["output_dim"], layer["input_dim"])
            )

            b = np.zeros(layer["output_dim"])

            self.params.append({"W": W, "b": b})

    def forward(self, x, eval=False):
        """
        Performs a single forward pass.

        Args:
            x (np.array): Samples on which the forward pass will be performed
            eval (bool, optional): Whether put the model in evaluation mode, meaning no information will be used for the backward pass.
                                   Defaults to False.

        Returns:
            np.array: Output of the model.
        """

        np.random.seed(self.seed)

        A = x
        for i, params in enumerate(self.params):  # goes through the layers one by one
            A_prev = A
            w = params["W"]
            b = params["b"]

            Z = A_prev.dot(w.T) + b  # compute the output of the layer

            activation = self.architecture[i]["activation"]

            # apply the layer-specific activation function on the layer's output
            if activation == "relu":
                A = relu(Z)
            elif activation == "sigmoid":
                A = sigmoid(Z)
            elif activation == "softmax":
                A = softmax(Z)

            if not eval:
                self.memory.append(
                    {"input": A_prev, "output": Z}
                )  # store information for backward pass and update

        return A

    def backward(self, pred, true):
        """
        Performs a single backward pass.

        Args:
            pred (np.array): Output of the model (logits).
            true (np.array): Ground truth labels.
        """

        np.random.seed(self.seed)

        n_sample = len(true)

        # computes gradient of the last layer
        if self.architecture[-1]["activation"] == "sigmoid":
            dA = -(np.divide(true, pred) - np.divide(1 - true, 1 - pred)) / n_sample

        if self.architecture[-1]["activation"] == "softmax":
            dA = pred
            dA[range(n_sample), true] -= 1
            dA /= n_sample

        for i, layer in reversed(
            list(enumerate(self.architecture))
        ):  # goes trough the layers one by one, starting from last layer and backward.

            # retrieve layer-specific info from memory
            A_prev = self.memory[i]["input"]
            Z = self.memory[i]["output"]
            W = self.params[i]["W"]
            activation_f = self.architecture[i]["activation"]

            # compute gradients
            if activation_f == "softmax":
                dW = A_prev.T.dot(dA)
                db = dA.sum(axis=0)
                dA = dA.dot(W)
            elif activation_f == "relu":
                dZ = d_relu(dA, Z)
                dW = A_prev.T.dot(dZ)
                db = dZ.sum(axis=0)
                dA = dZ.dot(W)

            self.gradients.append({"dW": dW, "db": db})  # store gradients for update

    def update(self, lr=0.1):
        """
        Performs a single update step.

        Args:
            lr (float, optional): Learning rate. Defaults to .1.
        """
        for i, layer in enumerate(self.params):  # goes trought layers one by one
            # update weights
            layer["W"] -= lr * list(reversed(self.gradients))[i]["dW"].T
            layer["b"] -= lr * list(reversed(self.gradients))[i]["db"]

        # clear memory, since we don't want to keep past information in later iterations.
        self.memory.clear()
        self.gradients.clear()

    def train(
        self, t_data, t_y, v_data, v_y, max_epochs=10001, dec_speed=200, eval=True
    ):
        """
        Performs model's training.

        Args:
            t_data (np.array): Training data
            t_y (np.array): Training labels
            v_data (np.array)): Validation data
            v_y (np.array): Validation labels
            max_epochs (int, optional): # of training iteration to perform. Defaults to 10001.
            dec_speed (int, optional): Decreasing speed of the learning rate. Defaults to 200.

        Returns:
            Tuple of lists: Train losses, validation losses, train accuracies, validation accuracies. All for every training epoch.
        """

        np.random.seed(self.seed)

        train_l = []
        train_a = []
        val_l = []
        val_a = []

        for epoch in range(1, max_epochs):

            if eval & (epoch % 50 == 0):
                # performs validation step
                v_out = self.forward(v_data, eval=True)
                v_y_pred = np.argmax(v_out, axis=1)
                v_loss = cross_entropy_loss(v_out, v_y)
                v_acc = np.mean(v_y_pred == v_y)

                # log validation informations
                val_l.append(v_loss)
                val_a.append(v_acc)
                print(f"\nval loss = {v_loss}, \nval acc = {v_acc}\n")

            # performs training step
            lr = dec_lr(epoch, dec_speed)  # compute the lr for the current epoch
            out = self.forward(t_data)
            y_pred = np.argmax(out, axis=1)

            # log training informations
            train_loss = cross_entropy_loss(out, t_y)
            train_acc = np.mean(y_pred == t_y)
            train_l.append(train_loss)
            train_a.append(train_acc)

            self.backward(out.copy(), t_y)
            print("epoch", epoch, "lr", lr)
            print("train_loss", train_loss, "train_acc", train_acc)

            self.update()

        print("\n-----training done-----\n")
        return train_l, train_a, val_l, val_a


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(dA, z):
    sig_z = sigmoid(z)

    return dA * sig_z * (1 - sig_z)


def relu(x):
    return np.maximum(0, x)


def d_relu(dA, z):

    dZ = dA.copy()
    dZ[z <= 0] = 0
    return dZ


def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def cross_entropy_loss(pred, true):
    """
    Compute the negative log likelihood loss.

    Args:
        pred (np.array): Model predictions (logits).
        true (np.array): Ground truth labels.

    Returns:
        float: loss value
    """

    N = len(true)
    return -np.log(pred[range(N), true]).sum() / N


def dec_lr(n, n_, lr_min=1e-5, lr_max=0.5):
    """
    Compute the exponential decreasing learning rate.

    Args:
        n (int): Current epoch iteration.
        n_ (int): Deacreasing speed (the higher the slower the lr will deacrease).
        lr_min (float, optional): Mininum learning rate. Defaults to 1e-5.
        lr_max (float, optional): Starting learning rate. Defaults to .5.

    Returns:
        float: Learning rate.
    """
    return max(lr_min, lr_max * np.exp(-n / n_))
