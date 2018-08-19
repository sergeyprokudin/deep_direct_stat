import numpy as np
import joblib


def load_idiap(data_path,
               val_split=0.5,
               canonical_split=True,
               verbose=0):
    """ Load, preprocess and perform val-test split for IDIAP headpose dataset
        You can download

    Parameters
    ----------
    data_path: str
        path to joblib pickle containing IDIAP data
    val_split: float
        ratio of test data that will be used for validation
    canonical_split: bool
        whether to perform canonical split used to get results for the paper

    Returns
    -------
        xtr: array of shape [n_samples, 75, 75, 3]
            images
        ptr: array of shape [n_samples, 1]
            pan angles (in radians) for head pose
        ttr: array of shape [n_samples, 1]
            tilt angles (in radians) for head pose
        rtr: array of shape [n_samples, 1]
            roll angles (in radians) for head pose
        names_tr: list of lenth n_samples
            list containing image names

        xval, pval, tval, rval, names_val - same for validation part
        xte, pte, tte, rte, names_te - same for test part

    """

    (xtr, ptr, ttr, rtr, names_tr), (xvalte, pvalte, tvalte, rvalte, names_valte) = joblib.load(data_path)

    # [channels, height, width] -> [height, width, channels]
    xtr = xtr.transpose([0, 2, 3, 1])
    xvalte = xvalte.transpose([0, 2, 3, 1])

    n_valtest_images = xvalte.shape[0]

    if canonical_split:
        val_split = 0.5
        np.random.seed(13)

    val_size = int(n_valtest_images * val_split)
    rix = np.random.choice(n_valtest_images, n_valtest_images, replace=False)

    np.random.seed(None)

    val_ix = rix[0:val_size]
    te_ix = rix[val_size:]

    xval = xvalte[val_ix]
    pval = pvalte[val_ix]
    tval = tvalte[val_ix]
    rval = rvalte[val_ix]
    names_val = [names_valte[ix] for ix in val_ix]

    xte = xvalte[te_ix]
    pte = pvalte[te_ix]
    tte = tvalte[te_ix]
    rte = rvalte[te_ix]
    names_te = [names_valte[ix] for ix in te_ix]

    return (xtr, ptr, ttr, rtr, names_tr), (xval, pval, tval, rval, names_val), (xte, pte, tte, rte, names_te)


def load_idiap_part(data_path,
                    data_part,
                    val_split=0.5,
                    canonical_split=True):

    (xtr, ptr_rad, ttr_rad, rtr_rad, names_tr),\
    (xval, pval_rad, tval_rad, rval_rad, names_val),\
    (xte, pte_rad, tte_rad, rte_rad, names_te) = load_idiap('data//IDIAP.pkl',
                                                            val_split=val_split,
                                                            canonical_split=canonical_split)

    if data_part == 'pan':
        return (xtr, ptr_rad), (xval, pval_rad), (xte, pte_rad)
    elif data_part == 'tilt':
        return (xtr, ttr_rad), (xval, tval_rad), (xte, tte_rad)
    elif data_part == 'roll':
        return (xtr, rtr_rad), (xval, rval_rad), (xte, rte_rad)
    else:
        raise ValueError("net_output should be 'pan', 'tilt' or 'roll'")
