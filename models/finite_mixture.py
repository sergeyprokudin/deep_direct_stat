import tensorflow as tf
import keras
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate

from utils.angles import deg2bit, bit2deg_multi, rad2bit, bit2deg
from utils.losses import maad_from_deg, von_mises_log_likelihood_np, von_mises_log_likelihood_tf
from scipy.stats import sem
from utils.sampling import sample_von_mises_mixture_multi
from utils.losses import maximum_expected_utility

N_BITERNION_OUTPUT = 2


def vgg_model(n_outputs=1, final_layer=False, l2_normalize_final=False,
              image_height=50, image_width=50,
              conv_dropout_val=0.2, fc_dropout_val=0.5, fc_layer_size=512):

    model = Sequential()

    model.add(Conv2D(24, kernel_size=(3, 3),
                     activation=None,
                     input_shape=[image_height, image_width, 3]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(24, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(48, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(48, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(conv_dropout_val))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(fc_dropout_val))

    if final_layer:
        model.add(Dense(n_outputs, activation=None))
        if l2_normalize_final:
            model.add(Lambda(lambda x: K.l2_normalize(x, axis=1)))

    return model


class BiternionVGGMixture:

    def __init__(self,
                 image_height=50,
                 image_width=50,
                 n_channels=3,
                 **kwargs):

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.hyp_params = kwargs
        self.n_components = kwargs.get('n_components', 8)
        self.learning_rate = kwargs.get('learning_rate', 1.0e-3)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1.0e-7)
        self.conv_dropout = kwargs.get('conv_dropout', 0.2)
        self.fc_dropout = kwargs.get('fc_dropout', 0.5)
        self.vgg_fc_layer_size = kwargs.get('vgg_fc_layer_size', 512)
        self.mix_fc_layer_size = kwargs.get('mix_fc_layer_size', 512)

        self.X = Input(shape=[image_height, image_width, 3])

        vgg_x = vgg_model(final_layer=False,
                          image_height=self.image_height,
                          image_width=self.image_width,
                          conv_dropout_val=self.conv_dropout,
                          fc_dropout_val=self.fc_dropout,
                          fc_layer_size=self.vgg_fc_layer_size)(self.X)

        mu_preds = []
        for i in range(0, self.n_components):
            mu_pred = Dense(N_BITERNION_OUTPUT)(Dense(self.mix_fc_layer_size)(vgg_x))
            mu_pred_normalized = Lambda(lambda x: K.l2_normalize(x, axis=1))(mu_pred)
            # mu_pred_norm_reshaped = Lambda(lambda x: K.reshape(x, [-1, 1, N_BITERNION_OUTPUT]))(mu_pred_normalized)
            mu_preds.append(mu_pred_normalized)

        self.mu_preds = concatenate(mu_preds)

        self.kappa_preds = Lambda(lambda x: K.abs(x))(Dense(self.n_components)(Dense(self.mix_fc_layer_size)(vgg_x)))
        # kappa_preds = Lambda(lambda x: K.reshape(x, [-1, self.n_components, 1]))(kappa_preds)

        self.component_probs = Lambda(lambda x: K.softmax(x))(Dense(self.n_components)(Dense(self.mix_fc_layer_size)(vgg_x)))
        # self.component_probs = Lambda(lambda x: K.reshape(x, [-1, self.n_components, 1]))(component_probs)

        self.y_pred = concatenate([self.mu_preds, self.kappa_preds, self.component_probs])

        self.model = Model(inputs=self.X, outputs=self.y_pred)

        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate,
                                               beta_1=self.beta1,
                                               beta_2=self.beta2,
                                               epsilon=self.epsilon)

        self.model.compile(optimizer=self.optimizer, loss=self._neg_mean_vmm_loglikelihood_tf)

    def fit(self, train_data, val_data, n_epochs, batch_size, callbacks=None):

        xtr, ytr_bit,  = train_data
        xval, yval_bit = val_data

        self.model.fit(xtr, ytr_bit,
                       batch_size=batch_size,
                       epochs=n_epochs,
                       validation_data=(xval, yval_bit),
                       callbacks=callbacks)

        return

    def save_weights(self, path):

        self.model.save_weights(path)

    def load_weights(self, path):

        self.model.load_weights(path)

    def parse_output_tf(self, y_preds):

        mu_preds = K.reshape(y_preds[:, 0:self.n_components*N_BITERNION_OUTPUT],
                             [-1, self.n_components, N_BITERNION_OUTPUT])

        kappa_ptr = self.n_components*N_BITERNION_OUTPUT
        kappa_preds = K.reshape(y_preds[:, kappa_ptr:kappa_ptr+self.n_components], [-1, self.n_components, 1])

        cprobs_ptr = kappa_ptr + self.n_components
        component_probs = K.reshape(y_preds[:, cprobs_ptr:cprobs_ptr+self.n_components], [-1, self.n_components])

        return mu_preds, kappa_preds, component_probs

    def parse_output_np(self, y_preds):

        mu_preds = np.reshape(y_preds[:, 0:self.n_components*N_BITERNION_OUTPUT],
                             [-1, self.n_components, N_BITERNION_OUTPUT])

        kappa_ptr = self.n_components*N_BITERNION_OUTPUT
        kappa_preds = np.reshape(y_preds[:, kappa_ptr:kappa_ptr+self.n_components], [-1, self.n_components, 1])

        cprobs_ptr = kappa_ptr + self.n_components
        component_probs = np.reshape(y_preds[:, cprobs_ptr:cprobs_ptr+self.n_components], [-1, self.n_components])

        return mu_preds, kappa_preds, component_probs

    def pdf(self, x, x_vals):
        """ Compute probability density function on a circle given images

        Parameters
        ----------
        x: numpy array of shape [n_images, image_width, image_height, n_channels]
            angles in biternion (cos, sin) representation that will be used to compute likelihood

        x_vals: numpy array of shape [n_points]
            angles (in rads) at which pdf values were computed

        Returns
        -------

        pdfs: numpy array of shape [n_images, n_components, n_points]
            array containing pdf values for each CVAE sample on circle [0, 2pi] for each values

        acc_pdf: numpy array of shape [n_images, n_points]
            array containing accumulated pdf value on circle [0, 2pi] for each values
        """

        n_images = x.shape[0]

        x_vals_tiled = np.ones(n_images)

        preds = self.model.predict(x)

        mu_preds, kappa_preds, component_probs = self.parse_output_np(preds)

        component_probs = np.tile(component_probs.reshape([n_images, self.n_components, 1]), [1, 1, len(x_vals)])

        vm_pdfs = np.zeros([n_images, self.n_components, len(x_vals)])

        for xid, xval in enumerate(x_vals):

            for cid in range(0, self.n_components):

                x_bit = rad2bit(x_vals_tiled*xval)

                vm_pdfs[:, cid, xid] = np.exp(np.squeeze(von_mises_log_likelihood_np(x_bit,
                                                                          mu_preds[:, cid, :],
                                                                          kappa_preds[:, cid])))

        acc_pdf = np.sum((component_probs*vm_pdfs), axis=1)

        return vm_pdfs, acc_pdf, component_probs

    def _von_mises_mixture_log_likelihood_np(self, y_true, y_pred):

        component_log_likelihoods = []

        mu, kappa, comp_probs = self.parse_output_np(y_pred)

        comp_probs = np.squeeze(comp_probs)

        for cid in range(0, self.n_components):
            component_log_likelihoods.append(von_mises_log_likelihood_np(y_true, mu[:, cid], kappa[:, cid]))

        component_log_likelihoods = np.concatenate(component_log_likelihoods, axis=1)

        log_likelihoods = np.log(np.sum(comp_probs*np.exp(component_log_likelihoods), axis=1))

        return log_likelihoods

    def _von_mises_mixture_log_likelihood_tf(self, y_true, y_pred):

        component_log_likelihoods = []

        mu, kappa, comp_probs = self.parse_output_tf(y_pred)

        for cid in range(0, self.n_components):
            component_log_likelihoods.append(von_mises_log_likelihood_tf(y_true, mu[:, cid], kappa[:, cid]))

        component_log_likelihoods = tf.concat(component_log_likelihoods, axis=1, name='component_likelihoods')

        log_likelihoods = tf.log(tf.reduce_sum(comp_probs*tf.exp(component_log_likelihoods), axis=1))

        return log_likelihoods

    def _neg_mean_vmm_loglikelihood_tf(self, y_true, y_pred):

        log_likelihoods = self._von_mises_mixture_log_likelihood_tf(y_true, y_pred)

        return -tf.reduce_mean(log_likelihoods)

    def evaluate(self, x, ytrue_deg, data_part, return_per_image=False):

        ytrue_bit = deg2bit(ytrue_deg)
        ypreds = self.model.predict(x)

        results = dict()

        vmmix_mu, vmmix_kappas, vmmix_probs = self.parse_output_np(ypreds)
        vmmix_mu_rad = np.deg2rad(bit2deg_multi(vmmix_mu))
        samples = sample_von_mises_mixture_multi(vmmix_mu_rad, vmmix_kappas, vmmix_probs, n_samples=100)
        point_preds = maximum_expected_utility(np.rad2deg(samples))
        maad_errs = maad_from_deg(point_preds, ytrue_deg)
        results['maad_loss'] = float(np.mean(maad_errs))
        results['maad_sem'] = float(sem(maad_errs))

        log_likelihoods = self._von_mises_mixture_log_likelihood_np(ytrue_bit, ypreds)
        results['log_likelihood_mean'] = float(np.mean(log_likelihoods))
        results['log_likelihood_sem'] = float(sem(log_likelihoods, axis=None))

        print("MAAD error (%s) : %f pm %fSEM" % (data_part,
                                                results['maad_loss'],
                                                results['maad_sem']))
        print("log-likelihood (%s) : %f pm %fSEM" % (data_part,
                                                    results['log_likelihood_mean'],
                                                    results['log_likelihood_sem']))

        if return_per_image:
            results['point_preds'] = bit2deg(deg2bit(point_preds))
            results['maad'] = maad_errs
            results['log_likelihood'] = log_likelihoods

        return results

