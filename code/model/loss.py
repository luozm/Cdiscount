"""
Loss functions

"""
from keras import backend as K


def darc1(alpha):
    """DARC1 loss function

    The implementation for loss function DARC1,
    which is employed by paper: "Generalization in Deep Learning".

    Please set the model's output without "Softmax"
    """
    def loss(y_true, y_pred):
        """
        :param y_true: true labels
        :param y_pred: predict labels

        :return: regularized loss value
        """
        y_pred_softmax = K.softmax(y_pred)
        xentropy = K.categorical_crossentropy(y_true, y_pred_softmax)
        reg = K.max(K.sum(K.abs(y_pred), axis=0))
        return xentropy+alpha*reg
    return loss


def sparse_darc1(alpha):
    """Sparse DARC1 loss function

    The implementation for loss function DARC1 (sparse version),
    which is employed by paper: "Generalization in Deep Learning".

    Please set the model's output without "Softmax"
    """
    def loss(y_true, y_pred):
        """
        :param y_true: true sparse labels
        :param y_pred: predict labels

        :return: regularized loss value
        """
        y_pred_softmax = K.softmax(y_pred)
        xentropy = K.sparse_categorical_crossentropy(y_true, y_pred_softmax)
        reg = K.max(K.sum(K.abs(y_pred), axis=0))
        return xentropy + alpha * reg
    return loss
