import sys
import os
from os.path import dirname
from datasets import pascal3d
from models.infinite_mixture import BiternionMixture
import datetime


def log_step(mess):
    dtstr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(' '.join([dtstr, mess]))


def main():
    cls = sys.argv[1]  # if cls is None, all classes will be loaded
    project_dir = dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(project_dir, 'logs')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_step("loading dataset")
    pascaldb_path = os.path.join(project_dir, 'data/pascal3d+_imagenet_train_test.h5')
    x_train, y_train, x_val, y_val, x_test, y_test = pascal3d.load_pascal_data(pascaldb_path, cls=cls)

    log_step("defining the model..")
    model = BiternionMixture(z_size=2, backbone_cnn='inception', hlayer_size=512, n_samples=50)
    ckpt_path = os.path.join(log_dir, '%s.h5' % cls)

    log_step("training on class :%s" % cls)
    model.fit(x_train, y_train, validation_data=[x_val, y_val], ckpt_path=ckpt_path, epochs=50,
              patience=5, batch_size=32)

    log_step("training finished. loading weights..")
    model.model.load_weights(ckpt_path)

    log_step("evaluating on test set..")
    model.evaluate(x_test, y_test)

    log_step("saving predictions for Matlab eval..")
    save_dir = os.path.join(log_dir, 'vp_test_results')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    model.save_detections_for_official_eval(x_test, os.path.join(save_dir, '%s_pred_view.txt' % cls))
    log_step("all done. Model checkpoint: %s" % ckpt_path)


if __name__ == '__main__':
    main()