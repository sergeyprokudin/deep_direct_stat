import sys
import os
from os.path import dirname
from datasets.towncentre import load_towncentre
from models.infinite_mixture import BiternionMixture
import datetime
from utils import angles
import numpy as np

def log_step(mess):
    dtstr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(' '.join([dtstr, mess]))


def convert_and_pad_angle_data(y):
    """ In case there is only 1 component, we augment data with fake tilt and roll components
    """
    y_pan_bit = angles.deg2bit(y)
    y_tilt_bit = angles.deg2bit(np.zeros_like(y))
    y_roll_bit = angles.deg2bit(np.zeros_like(y))

    return np.hstack([y_pan_bit, y_tilt_bit, y_roll_bit])

def main():
    project_dir = dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(project_dir, 'logs')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_step("loading dataset")
    data_path = os.path.join(project_dir, 'data/TownCentre.pkl.gz')
    (xtr, ytr, img_names_tr), (xval, yval, img_names_val), (xte, yte, img_names_te) = \
        load_towncentre(data_path, canonical_split=True, verbose=1)

    ytr = convert_and_pad_angle_data(ytr)
    yval = convert_and_pad_angle_data(yval)
    yte = convert_and_pad_angle_data(yte)

    log_step("defining the model..")
    model = BiternionMixture(z_size=2, input_shape=xtr.shape[1:],
                         backbone_cnn='mobilenet', backbone_weights=None, debug=False,
                         hlayer_size=512, n_samples=2)

    log_step("training started")
    ckpt_path = '../logs/towncentre.h5'

    model.fit(xtr, ytr, validation_data=[xval, yval], ckpt_path=ckpt_path, epochs=50, patience=5)

    log_step("training finished. loading weights..")
    model.model.load_weights(ckpt_path)

    log_step("evaluating on train set..")
    model.evaluate(xtr, ytr)

    log_step("evaluating on validation set..")
    model.evaluate(xval, yval)

    log_step("evaluating on test set..")
    model.evaluate(xte, yte)

    log_step("all done. Model checkpoint: %s" % ckpt_path)


if __name__ == '__main__':
    main()