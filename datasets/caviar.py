import pickle
import gzip
import numpy as np


def load_caviar(data_path,
                val_split=0.5,
                canonical_split=True,
                verbose=0):

    (xtr, ytr_deg, *info_tr), (xvalte, yvalte_deg, *info_valte) = pickle.load(gzip.open(data_path, 'rb'))

    def _parse_info(info):
        parsed_info = {}
        parsed_info['x_coord'] = info[0]
        parsed_info['y_coord'] = info[1]
        parsed_info['size'] = info[2]
        parsed_info['image_name'] = np.asarray(info[3])
        return parsed_info

    info_tr = _parse_info(info_tr)
    info_valte = _parse_info(info_valte)

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
    yval_deg = yvalte_deg[val_ix]
    info_val = _parse_info([info_valte[key][val_ix] for key in info_valte.keys()])

    xte = xvalte[te_ix]
    yte_deg = yvalte_deg[te_ix]
    info_te = _parse_info([info_valte[key][te_ix] for key in info_valte.keys()])

    return (xtr, ytr_deg, info_tr), (xval, yval_deg, info_val), (xte, yte_deg, info_te)

