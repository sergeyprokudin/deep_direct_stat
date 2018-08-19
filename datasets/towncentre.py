import numpy as np
import pickle, gzip


def split_dataset(X, y, img_names, split=0.1):
    itr, ival, ite, trs, vals, tes = [], [], [], set(), set(), set()
    for i, name in enumerate(img_names):
        # Extract the person's ID.
        pid = int(name.split('_')[1])

        # Decide where to put that person.
        if pid in trs:
            itr.append(i)
        elif pid in vals:
            ival.append(i)
        elif pid in tes:
            ite.append(i)
        else:
            rid = np.random.rand()
            if rid < 0.8:
                itr.append(i)
                trs.add(pid)
            elif (rid >= 0.8) and (rid < 0.9):
                ival.append(i)
                vals.add(pid)
            else:
                ite.append(i)
                tes.add(pid)
    return itr, ival, ite


def prepare_data(x, y):
    x, y = x.astype(np.float) / 255, y.astype(np.float)
    x = x.transpose([0, 2, 3, 1])  # [channels, height, width] -> [height, width, channels]
    # y = y.reshape(-1,1)
    return x, y


def load_towncentre(data_path,
                    val_test_split=0.1,
                    canonical_split=True,
                    verbose=1):

    x, y, img_names = pickle.load(gzip.open(data_path, 'rb'))

    img_names = np.asarray(img_names)

    x, y = prepare_data(x, y)
    if canonical_split:
        val_test_split = 0.1
        np.random.seed(13)
    person_ids = np.asarray([int(name.split('_')[1]) for name in img_names])
    unique_pid_set = np.unique(person_ids)
    rands = np.random.rand(unique_pid_set.shape[0])

    np.random.seed(None)

    train_pids = unique_pid_set[rands < 1-val_test_split*2]
    val_pids = unique_pid_set[(rands >= 1-val_test_split*2) & (rands < 1-val_test_split)]
    test_pids = unique_pid_set[rands > 1-val_test_split]

    ixtr = np.where(np.in1d(person_ids, train_pids))[0]
    ixval = np.where(np.in1d(person_ids, val_pids))[0]
    ixte = np.where(np.in1d(person_ids, test_pids))[0]

    xtr, ytr, img_names_tr = x[ixtr], y[ixtr], img_names[ixtr]
    xval, yval, img_names_val = x[ixval], y[ixval], img_names[ixval]
    xte, yte, img_names_te = x[ixte], y[ixte], img_names[ixte]

    if verbose:
        print("Number of train samples: %s" % xtr.shape[0])
        print("Number of validation samples: %s" % xval.shape[0])
        print("Number of test samples: %s" % xte.shape[0])

    return (xtr, ytr, img_names_tr), (xval, yval, img_names_val), (xte, yte, img_names_te)


def aug_data(x, y_deg, n_times=2, randomize_labels=True):
    n_points = y_deg.shape[0]
    x_aug = np.tile(x, [n_times, 1, 1, 1])
    y_deg_aug = np.tile(y_deg, [n_times])
    if randomize_labels:
        y_deg_aug[0:n_points] = y_deg
        y_deg_aug[n_points:n_points*2] = y_deg - 90
        # y_deg_aug = np.random.randint(0, 359, y_deg_aug.shape[0]).astype('float')
        # y_deg_aug[0:y_deg.shape[0]] = y_deg
    return x_aug, y_deg_aug