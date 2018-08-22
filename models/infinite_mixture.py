import tensorflow as tf
import keras
import numpy as np
import os
from scipy import stats

from scipy.misc import imresize

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Lambda, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet169
from keras.applications.mobilenetv2 import MobileNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.custom_keras_callbacks import ModelCheckpointEveryNBatch


from utils.losses import maad_from_deg, maximum_expected_utility
from utils.losses import cosine_loss_tf, von_mises_log_likelihood_tf
from utils.losses import von_mises_log_likelihood_np
from utils.angles import bit2deg, rad2bit, bit2rad

P_UNIFORM = 0.15916927


class BiternionMixture:

    def __init__(self,
                 input_shape=[224, 224, 3],
                 debug=False,
                 backbone_cnn='inception',
                 backbone_weights='imagenet',
                 learning_rate=1.0e-4,
                 z_size=8,
                 n_samples=5,
                 hlayer_size=512,
                 noise_std=1.0,
                 gammas=[1.0e-1, 1.0e-1, 1.0e-1]):

        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.set_gammas(gammas)
        self.hlayer_size = hlayer_size
        self.z_size = z_size
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.n_sample_outputs = 9
        self.backbone_weights = backbone_weights

        if debug:
            x_in = Input(shape=input_shape)
            x = Flatten(input_shape=input_shape)(x_in)
            x = Dense(128, activation='relu')(x)
        else:
            if backbone_cnn == 'inception':
                backbone_model = InceptionResNetV2(weights=self.backbone_weights, include_top=False,
                                                   input_shape=input_shape)
            elif backbone_cnn == 'densenet':
                backbone_model = DenseNet169(weights=self.backbone_weights, include_top=False,
                                             input_shape=input_shape)
            elif backbone_cnn == 'mobilenet':
                backbone_model = MobileNetV2(weights=self.backbone_weights, include_top=False,
                                             input_shape=input_shape)

            x = backbone_model.output
            x = GlobalAveragePooling2D()(x)

        x = Dense(z_size*16, activation='relu', input_shape=[1])(x)
        x = Dense(z_size*4, activation='relu', input_shape=[1])(x)
        x = Dense(z_size, activation='relu', input_shape=[1])(x)

        az_mean, az_kappa = self.decoder_seq("azimuth")
        el_mean, el_kappa = self.decoder_seq("elevation")
        ti_mean, ti_kappa = self.decoder_seq("tilt")

        z_lst = []
        x_z_lst = []
        x_z_decoded_lst = []

        for k in range(0, n_samples):
            z_lst.append(Lambda(self._sample_z)(x))
            x_z_lst.append(concatenate([x, z_lst[k]]))
            kth_preds = concatenate([az_mean(x_z_lst[k]), az_kappa(x_z_lst[k]),
                                     el_mean(x_z_lst[k]), el_kappa(x_z_lst[k]),
                                     ti_mean(x_z_lst[k]), ti_kappa(x_z_lst[k])])
            x_z_decoded_lst.append(kth_preds)

        y_pred = concatenate(x_z_decoded_lst)

        if debug:
            self.model = Model(x_in, y_pred, name='bi')
        else:
            self.model = Model(backbone_model.input, y_pred, name='BiternionInception')

        opt = Adam(lr=learning_rate)
        self.model.compile(optimizer=opt, loss=self._mc_loss)

    def _mc_loss(self, y_target, y_pred):

        az_target, el_target, ti_target = self.unpack_target(y_target)

        sample_az_likelihoods = []
        sample_el_likelihoods = []
        sample_ti_likelihoods = []

        n_feat = 9

        for sid in range(0, self.n_samples):

            az_mean, az_kappa, el_mean, el_kappa, ti_mean, ti_kappa = \
                self.unpack_sample_preds(y_pred[:, sid * n_feat:sid * n_feat + n_feat])

            sample_az_likelihoods.append(K.exp(von_mises_log_likelihood_tf(az_target,
                                                                           az_mean,
                                                                           az_kappa)))

            sample_el_likelihoods.append(K.exp(von_mises_log_likelihood_tf(el_target,
                                                                           el_mean,
                                                                           el_kappa)))

            sample_ti_likelihoods.append(K.exp(von_mises_log_likelihood_tf(ti_target,
                                                                           ti_mean,
                                                                           ti_kappa)))
        az_likelihood = -K.log(P_UNIFORM * self.az_gamma +
                               (1 - self.az_gamma) * K.mean(concatenate(sample_az_likelihoods), axis=1))

        el_likelihood = -K.log(P_UNIFORM * self.el_gamma +
                               (1 - self.el_gamma) * K.mean(concatenate(sample_el_likelihoods), axis=1))

        ti_likelihood = -K.log(P_UNIFORM * self.ti_gamma +
                               (1 - self.ti_gamma) * K.mean(concatenate(sample_ti_likelihoods), axis=1))

        return az_likelihood+el_likelihood+ti_likelihood

    def _sample_z(self, x):
        return K.random_normal(shape=K.shape(x), mean=0., stddev=self.noise_std)

    def decoder_seq(self, name):

        decoder_seq = Sequential(name='decoder_%s'%name)

        decoder_seq.add(Dense(self.hlayer_size, activation='relu', input_shape=[self.z_size*2]))
        decoder_seq.add(Dense(self.hlayer_size, activation='relu'))

        decoder_mean = Sequential()
        decoder_mean.add(decoder_seq)
        decoder_mean.add(Dense(128, activation='relu'))
        decoder_mean.add(Dense(2, activation='linear'))
        decoder_mean.add(Lambda(lambda x: K.l2_normalize(x, axis=1), name='%s_mean' % name))

        decoder_kappa = Sequential()
        decoder_kappa.add(decoder_seq)
        decoder_kappa.add(Dense(128, activation='relu'))
        decoder_kappa.add((Dense(1,  activation='linear')))
        decoder_kappa.add(Lambda(lambda x: K.abs(x), name='%s_kappa' % name))

        return decoder_mean, decoder_kappa

    def set_gammas(self, gammas):
        self.az_gamma = gammas[0]
        self.el_gamma = gammas[1]
        self.ti_gamma = gammas[2]

    def unpack_sample_preds(self, y_pred):

        az_mean = y_pred[:, 0:2]
        az_kappa =  y_pred[:, 2:3]

        el_mean = y_pred[:, 3:5]
        el_kappa = y_pred[:, 5:6]

        ti_mean = y_pred[:, 6:8]
        ti_kappa = y_pred[:, 8:9]

        return az_mean, az_kappa, el_mean, el_kappa, ti_mean, ti_kappa

    def unpack_all_preds(self, y_pred):

        az_means = []
        az_kappas = []
        el_means = []
        el_kappas = []
        ti_means = []
        ti_kappas = []

        n_feat = 9

        for sid in range(0, self.n_samples):
            az_mean, az_kappa, el_mean, el_kappa, ti_mean, ti_kappa = \
                self.unpack_sample_preds(y_pred[:, sid * n_feat:sid * n_feat + n_feat])

            az_means.append(az_mean)
            az_kappas.append(az_kappa)
            el_means.append(el_mean)
            el_kappas.append(el_kappa)
            ti_means.append(ti_mean)
            ti_kappas.append(ti_kappa)

        return az_means, az_kappas, el_means, el_kappas, ti_means, ti_kappas

    def unpack_target(self, y_target):

        az_target = y_target[:, 0:2]
        el_target = y_target[:, 2:4]
        ti_target = y_target[:, 4:6]

        return az_target, el_target, ti_target

    def fit(self, x, y, validation_data, ckpt_path, epochs=1, batch_size=32, patience=5):

        early_stop_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')
        model_ckpt = ModelCheckpoint(ckpt_path, monitor='val_loss', verbose=1,
                                     save_best_only=True, save_weights_only=True)

        self.model.fit(x, y, validation_data=validation_data,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[early_stop_cb, model_ckpt])

        self.model.load_weights(ckpt_path)

    def predict(self, x):
        """ Predict orientation angles (azimuth, elevation, tilt) from images, in degrees
        """
        y_pred = self.model.predict(np.asarray(x))

        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = \
            self.unpack_all_preds(y_pred)
        az_preds_deg, el_preds_deg, ti_preds_deg = self.convert_to_deg_preds(y_pred)

        return az_preds_deg, el_preds_deg, ti_preds_deg

    def log_likelihood(self, y_true_bit, y_preds_bit, kappa_preds, gamma, angle='', verbose=1):

        likelihoods = np.hstack([np.exp(von_mises_log_likelihood_np(y_true_bit, y_preds_bit[sid], kappa_preds[sid]))
                       for sid in range(0, self.n_samples)])

        vm_lls = np.log(P_UNIFORM*gamma +
                        (1-gamma)*np.mean(likelihoods, axis=1))
        vm_ll_mean = np.mean(vm_lls)
        vm_ll_sem = stats.sem(vm_lls)
        if verbose:
            print("Log-likelihood %s : %2.2f+-%2.2fSE" % (angle, vm_ll_mean, vm_ll_sem))

        return vm_lls, vm_ll_mean, vm_ll_sem

    def maad(self, y_true_deg, y_pred_deg, angle='', verbose=1):
        """ Compute Mean Absolute Angular Deviation between ground truth and predictions (in degrees)
        """
        aads = maad_from_deg(y_true_deg, y_pred_deg)
        maad = np.mean(aads)
        sem = stats.sem(aads)
        if verbose:
            print("MAAD %s : %2.2f+-%2.2fSE" % (angle, maad, sem))
        return aads, maad, sem

    def convert_to_deg_preds(self, y_pred):

        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = self.unpack_all_preds(y_pred)

        az_preds_deg_lst = np.vstack([bit2deg(az_preds_bit[sid]) for sid in range(0, self.n_samples)]).T
        az_preds_deg = maximum_expected_utility(az_preds_deg_lst)
        el_preds_deg_lst = np.vstack([bit2deg(el_preds_bit[sid]) for sid in range(0, self.n_samples)]).T
        el_preds_deg = maximum_expected_utility(el_preds_deg_lst)
        ti_preds_deg_lst = np.vstack([bit2deg(ti_preds_bit[sid]) for sid in range(0, self.n_samples)]).T
        ti_preds_deg = maximum_expected_utility(ti_preds_deg_lst)

        return az_preds_deg, el_preds_deg, ti_preds_deg

    def evaluate(self, x, y_true, verbose=1, return_full=False):

        y_pred = self.model.predict(np.asarray(x))

        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = \
            self.unpack_all_preds(y_pred)
        az_preds_deg, el_preds_deg, ti_preds_deg = self.convert_to_deg_preds(y_pred)

        az_true_bit, el_true_bit, ti_true_bit = self.unpack_target(y_true)
        az_true_deg = bit2deg(az_true_bit)
        el_true_deg = bit2deg(el_true_bit)
        ti_true_deg = bit2deg(ti_true_bit)

        az_aads, az_maad, az_sem = self.maad(az_true_deg, az_preds_deg, 'azimuth', verbose=verbose)
        el_aads, el_maad, el_sem = self.maad(el_true_deg, el_preds_deg, 'elevation', verbose=verbose)
        ti_aads, ti_maad, ti_sem = self.maad(ti_true_deg, ti_preds_deg, 'tilt', verbose=verbose)

        az_lls, az_ll_mean, az_ll_sem = self.log_likelihood(az_true_bit, az_preds_bit, az_preds_kappa, self.az_gamma,
                                                               'azimuth', verbose=verbose)
        el_lls, el_ll_mean, el_ll_sem = self.log_likelihood(el_true_bit, el_preds_bit, el_preds_kappa, self.el_gamma,
                                                               'elevation', verbose=verbose)
        ti_lls, el_ll_mean, ti_ll_sem = self.log_likelihood(ti_true_bit, ti_preds_bit, ti_preds_kappa, self.ti_gamma,
                                                               'tilt', verbose=verbose)

        lls = az_lls + el_lls + ti_lls
        ll_mean = np.mean(lls)
        ll_sem = stats.sem(lls)

        maad_mean = np.mean([az_maad, el_maad, ti_maad])
        print("MAAD TOTAL: %2.2f+-%2.2fSE" % (maad_mean, az_sem))
        print("Log-likelihood TOTAL: %2.2f+-%2.2fSE" % (ll_mean, ll_sem))

        if return_full:
            return maad_mean, ll_mean, ll_sem, lls
        else:
            return maad_mean, ll_mean, ll_sem

    def save_detections_for_official_eval(self, x, save_path):

        # det path example: '/home/sprokudin/RenderForCNN/view_estimation/vp_test_results/aeroplane_pred_view.txt'

        y_pred = self.model.predict(np.asarray(x))
        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = \
            self.unpack_sample_preds(y_pred)
        az_preds_deg, el_preds_deg, ti_preds_deg = self.convert_to_deg_preds(y_pred)

        y_pred = np.vstack([az_preds_deg, el_preds_deg, ti_preds_deg]).T

        np.savetxt(save_path, y_pred, delimiter=' ', fmt='%i')
        print("evaluation data saved to %s" % save_path)

        return

    def train_finetune_eval(self, x_train, y_train, x_val, y_val, x_test, y_test,
                            ckpt_path, batch_size=32, patience=10, epochs=200):

        self.fit(x_train, y_train, [x_val, y_val], epochs=epochs,
                 ckpt_path=ckpt_path, patience=patience, batch_size=batch_size)

        print("EVALUATING ON TRAIN")
        train_maad, train_ll, train_ll_sem = self.evaluate(x_train, y_train)
        print("EVALUATING ON VALIDAITON")
        val_maad, val_ll, val_ll_sem = self.evaluate(x_val, y_val)
        print("EVALUATING ON TEST")
        test_maad, test_ll, test_ll_sem = self.evaluate(x_test, y_test)

        return train_maad, train_ll, val_maad, val_ll, test_maad, test_ll

    def pdf(self, x, gamma=1.0e-1, angle='azimuth', step=0.01):
        """

        :param x: input images
        :param gamma: weight of a default uniform distribution added to mixture
        :param angle: azimuth, elevation or tilt
        :param step: step of pdf
        :return: points at (0, 2pi) and corresponding pdf values
        """
        vals = np.arange(0, 2*np.pi, step)

        n_images = x.shape[0]
        x_vals_tiled = np.ones(n_images)

        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = \
            self.unpack_all_preds(self.model.predict(x))

        if angle == 'azimuth':
            mu_preds_bit = az_preds_bit
            kappa_preds = az_preds_kappa
            gamma = self.az_gamma
        elif angle == 'elevation':
            mu_preds_bit = el_preds_bit
            kappa_preds = el_preds_kappa
            gamma = self.el_gamma
        elif angle == 'tilt':
            mu_preds_bit = ti_preds_bit
            kappa_preds = ti_preds_kappa
            gamma = self.ti_gamma

        pdf_vals = np.zeros([n_images, len(vals)])

        for xid, xval in enumerate(vals):
            x_bit = rad2bit(x_vals_tiled*xval)
            pdf_vals[:, xid] = np.exp(self.log_likelihood(x_bit,
                                                          mu_preds_bit, kappa_preds, gamma, angle=angle, verbose=0)[0])

        return vals, pdf_vals

    def plot_pdf(self, vals, pdf_vals, ax=None, target=None, predicted=None, step=1.0e-2):

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        x = np.arange(0, 2*np.pi, step)
        xticks = [0., .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]
        xticks_labels = ["$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks_labels)
        ax.plot(vals, pdf_vals, label='pdf')
        # mu = np.sum(pdf_vals*vals*step)
        # ax.axvline(mu, c='blue', label='mean')
        if target is not None:
            ax.axvline(target, c='orange', label='ground truth')

        if predicted is not None:
            ax.axvline(predicted, c='darkblue', label='predicted value')

        ax.set_xlim((0, 2*np.pi))
        ax.set_ylim(0, 1.0)
        ax.legend(loc=4)

        return

    def visualize_detections(self, x, y_true=None, kappa=1.0):

        import matplotlib.pyplot as plt

        n_images = x.shape[0]

        y_pred = self.model.predict(x)
        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = \
            self.unpack_sample_preds(y_pred)
        az_preds_deg, el_preds_deg, ti_preds_deg = self.convert_to_deg_preds(y_pred)
        az_preds_rad = np.deg2rad(az_preds_deg)

        if y_true is not None:
            az_true_bit, el_true_bit, ti_true_bit = self.unpack_target(y_true)
            az_true_rad = bit2rad(az_true_bit)

        xvals, pdf_vals = self.pdf(x, gamma=self.az_gamma)

        for i in range(0, n_images):
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(x[i])
            if y_true is not None:
                self.plot_pdf(xvals, pdf_vals[i], target=az_true_rad[i], predicted=az_preds_rad[i], ax=axs[1])
            else:
                self.plot_pdf(xvals, pdf_vals[i], ax=axs[1])
            fig.show()

        return

    def make_halo(self, img, standard_size=[224, 224], black_canvas=False):
        img_halo = np.copy(img)
        lx, ly = img.shape[0:2]
        X, Y = np.ogrid[0:lx, 0:ly]
        mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
        if black_canvas:
            img_halo[mask] = 0
        else:
            img_halo[mask] = 255
        img_halo = imresize(img_halo, size=standard_size)
        return img_halo

    def frame_image(self, img, frame_width, black_canvas=False):
        b = frame_width # border size in pixel
        ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
        if black_canvas:
            framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]), dtype='uint8')*255
        else:
            framed_img = np.ones((b+ny+b, b+nx+b, img.shape[2]), dtype='uint8')*255
        for i in range(0, 3):
            framed_img[b:-b, b:-b,i] = img[:,:,i]
        return framed_img

    def plot_pdf_circle(self, img, xvals, pdf, ypred_rad=None, ytrue_rad=None, show_legend=True,
                        theta_zero_location='E', show_ticks=True, pdf_scaler=15.0, pdf_color='green',
                        pred_color='blue', fontsize=10):

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 4))

        ax_img = fig.add_subplot(1, 1, 1, frameon=False)
        ax_pdf = fig.add_subplot(1, 1, 1, projection='polar')

        ax_img.axis("off")

        img_halo = self.frame_image(self.make_halo(img, standard_size=[224, 224]), frame_width=90)

        ax_img.imshow(img_halo)

        ax_pdf.axvline(0, ymin=0.54, color='white', linewidth=3, linestyle='dashed')

        ax_pdf.set_yticks([])
        ax_pdf.set_xticks(([]))
        if show_ticks:
            ax_pdf.set_xticklabels(["0°", "%d°" % np.rad2deg(ypred_rad)], fontsize=fontsize)
            if (ypred_rad is not None) and (ytrue_rad is not None):
                ax_pdf.set_xticks(([0, ypred_rad, ytrue_rad]))
                ax_pdf.set_xticklabels(["0°", "%d°" % np.rad2deg(ypred_rad), "%d°" % np.rad2deg(ytrue_rad)],
                                       fontsize=fontsize)
            elif ytrue_rad is not None:
                ax_pdf.set_xticks(([0, ytrue_rad]))
                ax_pdf.set_xticklabels(["0°", "%d°" % np.rad2deg(ytrue_rad)], fontsize=fontsize)
            elif ypred_rad is not None:
                ax_pdf.set_xticks(([0, ypred_rad]))
                ax_pdf.set_xticklabels(["0°", "%d°" % np.rad2deg(ypred_rad)], fontsize=fontsize)
        else:
            ax_pdf.set_xticklabels([])
        ax_pdf.set_ylim(0, 20)
        ax_pdf.patch.set_alpha(0.1)
        ax_pdf.set_theta_zero_location(theta_zero_location)
        margin = 10.2
        border = 0.8

        ax_pdf.fill_between(xvals, np.ones(xvals.shape[0])*(margin+border), pdf*pdf_scaler+margin+border,
                            color=pdf_color, alpha=0.5, label='$p_{\\theta}(\phi | \mathbf{x})$')
        if ytrue_rad is not None:
            ax_pdf.axvline(ytrue_rad, ymin=0.54, color='orange', linewidth=4, label='ground truth')
        if ypred_rad is not None:
            ax_pdf.axvline(ypred_rad, ls='dashed', ymin=0.54, color=pred_color, linewidth=4, label='prediction')
        if show_legend:
            ax_pdf.legend(fontsize=fontsize, loc=1, framealpha=1.0)

        return fig

    def visualize_detections_on_circle(self, x, y_true=None, show_legend=True, save_figs=False, save_path=None):

        n_images = x.shape[0]

        if save_figs:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        y_pred = self.model.predict(x)
        az_preds_bit, az_preds_kappa, el_preds_bit, el_preds_kappa, ti_preds_bit, ti_preds_kappa = \
            self.unpack_sample_preds(y_pred)
        az_preds_deg, el_preds_deg, ti_preds_deg = self.convert_to_deg_preds(y_pred)
        az_preds_rad = np.deg2rad(az_preds_deg)

        if y_true is not None:
            az_true_bit, el_true_bit, ti_true_bit = self.unpack_target(y_true)
            az_true_rad = bit2rad(az_true_bit)
        else:
            az_true_rad = list(None for i in range(0, n_images))

        xvals, pdf_vals = self.pdf(x, gamma=self.az_gamma)

        for i in range(0, n_images):
            if i > 0:
                show_legend = False

            fig = self.plot_pdf_circle(x[i], xvals, pdf_vals[i],
                                       ypred_rad=az_preds_rad[i],
                                       ytrue_rad=az_true_rad[i],
                                       theta_zero_location='N',
                                       show_ticks=True,
                                       pdf_color='blue',
                                       pred_color='blue',
                                       show_legend=show_legend)

            if save_figs:
                fig_save_path = os.path.join(save_path, 'frame_%d.png' % i)
                print("saving frame detections to %s" % fig_save_path)
                fig.savefig(fig_save_path)

            fig.show()

        return