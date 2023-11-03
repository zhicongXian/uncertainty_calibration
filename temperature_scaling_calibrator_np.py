from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.calibration import calibration_curve
from utils.reliability_diagrams import *
from scipy.optimize import minimize


def ece_score(probabilities, accuracy, confidence):
    n_bins = len(accuracy)
    n = len(probabilities)
    h = np.histogram(a=probabilities, range=(0, 1), bins=n_bins)[0]

    ece = 0
    for m in np.arange(n_bins):
        ece = ece + (h[m] / n) * np.abs(accuracy[m] - confidence[m])

    return ece


def numpy_invert_sigmoid(x: np.ndarray):
    return -np.log((1 / (x + np.finfo(np.float64).eps)) - 1)


def numpy_invert_softmax(x: np.ndarray, c=1):
    """

    Parameters
    ----------
    x :
    c :

    Returns
    -------

    """

    return np.log(x + np.finfo(np.float64).eps) + c


def bce(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Binary cross entropy loss
    Parameters
    ----------
    logits :
    labels :

    Returns
    -------

    """
    predictions = 1 / (1 + np.exp(-logits))
    loss = np.mean(-np.sum(labels * np.log(predictions + np.finfo(np.float64).eps) + (1 - labels) * np.log(
        1 - predictions + np.finfo(np.float64).eps), axis=-1))

    return loss


def cce(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Categorical Cross Entropy Loss
    Parameters
    ----------
    logits :
    labels :

    Returns
    -------

    """
    # 1. check input label format:
    if np.ndim(labels) >= 2 and np.max(labels) <= 1 and np.min(labels) >= 0:
        # data already in one-hot-encoding:
        one_hot_targets = labels
    elif np.ndim(labels) == 2:
        # label vector is a 2d array, transform to one-hot-encoding:
        enc = OneHotEncoder()
        one_hot_targets = enc.fit_transform(labels)
    else:
        raise ValueError(f"Unknown labels format {labels}")

    # 2. loss calculation
    predictions = softmax(logits)
    loss = np.mean(-np.sum(one_hot_targets * np.log(predictions + np.finfo(np.float64).eps), axis=-1))

    return loss


class TemperatureScalingCalibrator(BaseEstimator):
    """

    """

    def __init__(self):
        self._temperature = 1.5  # initial value to be optimized

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """
        Learns the logistic regression weights.

        Parameters
        ----------
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        y_true : 1d ndarray
            Binary true targets.

        Returns
        -------
        self
        """
        self.fit_predict(y_prob, y_true)
        return self

    def fit_predict(self, y_prob: np.ndarray, y_true: np.ndarray, verbose=True) -> np.ndarray:
        """
        Chain the .fit and .predict step together.

        Parameters
        ----------
        verbose :
        y_prob : 1d ndarray
            Raw probability/score of the positive class.

        y_true : 1d ndarray
            Binary true targets.

        Returns
        -------
        y_calibrated_prob : 1d ndarray
            Calibrated probability.
        """

        # 1. check input data format

        if np.ndim(y_prob) == 1:
            y_prob = y_prob[:, None]
        if np.ndim(y_true) == 1:
            y_true = y_true[:, None]

        # 2. define objective function
        nb_class = None
        binary_class = None
        if y_true.shape[-1] == 1:
            if len(set(np.squeeze(y_true).tolist())) <= 2:
                binary_class = True
                nb_class = 2
            else:
                binary_class = False
                nb_class = len(set(np.squeeze(y_true).tolist()))


        elif y_true.shape[-1] >= 2 and np.max(y_true) <= 1 and np.min(y_true) >= 0:
            print("labels are one-hot encoding")
            if y_true.shape[-1] == 2:
                binary_class = True
                nb_class = 2
            else:
                binary_class = False
                nb_class = y_true.shape[-1]
        else:
            raise ValueError(f"Unknown label format {y_true}")

        # reverse logistic
        if binary_class:
            logits = numpy_invert_sigmoid(y_prob)
        else:
            logits = numpy_invert_softmax(y_prob)

        def objective(temperature):

            scaled_logits = logits / temperature

            if binary_class:
                loss = bce(scaled_logits, y_true)
            else:
                loss = cce(scaled_logits, y_true)
            return loss

        t0 = self._temperature
        res = minimize(objective, np.asarray([t0]).astype(np.float128), method='Nelder-Mead')
        print(res)
        self._temperature = res.x[0]

        # output the fitting results

        scaled_logits = logits / self._temperature

        calibrated_y_prob = 1 / (1 + np.exp(-scaled_logits))

        # Reliability diagrams group predictions into discrete bins and plot the expected accuracy on the y-axis
        # and average classifier confisdence on the x-axis, the diagonal line in the diagram is called calibration curve
        # ideally, all points with (predicted_mean_in_each_bin, fraction_of_positive_labels_in_each_bin) should line
        # on the calibration curve.
        for i in range(nb_class):
            acc_before, prob_before = calibration_curve(y_prob=y_prob[:, i],
                                                        y_true=y_true[:, i],
                                                        n_bins=10)

            acc_after, prob_after = calibration_curve(y_prob=calibrated_y_prob[:, i],
                                                      y_true=y_true[:, i],
                                                      n_bins=10)

            y_pred = (y_prob > 0.5).astype(np.int64)

            if verbose:
                fig1 = reliability_diagram(y_true[:, i], y_pred[:, i], y_prob[:, i], num_bins=10, draw_ece=True,
                                           draw_bin_importance="alpha", draw_averages=True,
                                           title="previous reliability diagram", figsize=(6, 6), dpi=100,
                                           return_fig=True)
                fig2 = reliability_diagram(y_true[:, i], y_pred[:, i], calibrated_y_prob[:, i], num_bins=10,
                                           draw_ece=True,
                                           draw_bin_importance="alpha", draw_averages=True,
                                           title="after calibration reliability diagram", figsize=(6, 6), dpi=100,
                                           return_fig=True)
                # expected calibration error
                ece_before = ece_score(y_prob, acc_before, prob_before)
                print("ece score for before calibration: ", ece_before)

                ece_after = ece_score(calibrated_y_prob, acc_after, prob_after)
                print("ece score for after calibration: ", ece_after)

        print(f"Final temperature is: {self._temperature}")

        # return the scaled results

        return calibrated_y_prob

    def predict(self, y_prob: np.ndarray) -> np.ndarray:

        logits = numpy_invert_sigmoid(y_prob)
        scaled_logits = logits / self._temperature
        calibrated_y_prob = 1 / (1 + np.exp(-scaled_logits))

        return calibrated_y_prob
