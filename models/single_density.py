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

from utils.angles import deg2bit, bit2deg, rad2bit
from utils.losses import mad_loss_tf, cosine_loss_tf, von_mises_loss_tf, von_mises_log_likelihood_tf
from utils.losses import von_mises_log_likelihood_np, von_mises_neg_log_likelihood_keras
from utils.losses import maad_from_deg
from scipy.stats import sem


def vgg_model(n_outputs=1, final_layer=False, l2_normalize_final=False,
              image_height=50, image_width=50, n_channels=3,
              conv_dropout_val=0.2, fc_dropout_val=0.5, fc_layer_size=512):

    model = Sequential(name='VGG')

    model.add(Conv2D(24, kernel_size=(3, 3),
                     activation=None,
                     input_shape=[image_height, image_width, n_channels]))
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


class DegreeVGG:

    def __init__(self,
             image_height=50,
             image_width=50,
             n_channels=3,
             n_outputs=1,
             predict_kappa=False,
             fixed_kappa_value=1.0):

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels

        self.X = Input(shape=[image_height, image_width, self.n_channels])

        self.vgg_x = vgg_model(n_outputs=1,
                          final_layer=True,
                          image_height=self.image_height,
                          image_width=self.image_width,
                          n_channels=self.n_channels)(self.X)

        self.model = Model(self.X, self.vgg_x)

    def evaluate(self, x, ytrue_deg, data_part):

        ypreds_deg = np.squeeze(self.model.predict(x))

        loss = maad_from_deg(ypreds_deg, ytrue_deg)

        results = dict()

        results['maad_loss'] = float(np.mean(loss))
        results['maad_loss_sem'] = float(sem(loss, axis=None))
        print("MAAD error (%s) : %f ± %fSEM" % (data_part,
                                             results['maad_loss'],
                                             results['maad_loss_sem']))

        return results


class BiternionVGG:

    def __init__(self,
                 image_height=50,
                 image_width=50,
                 n_channels=3,
                 loss_type='cosine',
                 predict_kappa=False,
                 fixed_kappa_value=1.0,
                 **kwargs):

        self.image_height = image_height
        self.image_width = image_width
        self.n_channels = n_channels
        self.predict_kappa = predict_kappa
        self.fixed_kappa_value = fixed_kappa_value
        self.hyp_params = kwargs
        self.n_u = kwargs.get('n_hidden_units', 8)
        self.learning_rate = kwargs.get('learning_rate', 1.0e-3)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1.0e-7)
        self.conv_dropout = kwargs.get('conv_dropout', 0.2)
        self.fc_dropout = kwargs.get('fc_dropout', 0.5)
        self.vgg_fc_layer_size = kwargs.get('vgg_fc_layer_size', 512)
        self.loss_type = loss_type
        self.loss = self._pick_loss()

        self.X = Input(shape=[image_height, image_width, self.n_channels])

        vgg_x = vgg_model(final_layer=False,
                          image_height=self.image_height,
                          image_width=self.image_width,
                          n_channels=self.n_channels,
                          conv_dropout_val=self.conv_dropout,
                          fc_dropout_val=self.fc_dropout,
                          fc_layer_size=self.vgg_fc_layer_size)(self.X)

        self.y_pred = Lambda(lambda x: K.l2_normalize(x, axis=1))(Dense(2)(vgg_x))

        if self.predict_kappa:
            self.kappa_pred = Lambda(lambda x: K.abs(x))(Dense(1)(vgg_x))
            self.model = Model(self.X, concatenate([self.y_pred, self.kappa_pred]))
        else:
            self.model = Model(self.X, self.y_pred)

        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate,
                                               beta_1=self.beta1,
                                               beta_2=self.beta2,
                                               epsilon=self.epsilon)

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def _pick_loss(self):

        if self.loss_type == 'cosine':
            print("using cosine loss..")
            loss = cosine_loss_tf
        elif self.loss_type == 'von_mises':
            print("using von-mises loss..")
            loss = von_mises_loss_tf
        elif self.loss_type == 'mad':
            print("using mad loss..")
            loss = mad_loss_tf
        elif self.loss_type == 'vm_likelihood':
            print("using likelihood loss..")
            if self.predict_kappa:
                loss = von_mises_neg_log_likelihood_keras
            else:

                def _von_mises_neg_log_likelihood_keras_fixed(y_true, y_pred):
                    mu_pred = y_pred[:, 0:2]
                    kappa_pred = tf.ones([tf.shape(y_pred[:, 2:])[0], 1])*self.fixed_kappa_value
                    return -K.mean(von_mises_log_likelihood_tf(y_true, mu_pred, kappa_pred))

                loss = _von_mises_neg_log_likelihood_keras_fixed
        else:
            raise ValueError("loss should be 'mad','cosine','von_mises' or 'vm_likelihood'")

        return loss

    def fit(self, train_data, val_data, n_epochs, batch_size, callbacks=None):

        xtr, ytr_bit,  = train_data
        xval, yval_bit = val_data

        self.model.fit(xtr, ytr_bit,
                       batch_size=batch_size,
                       epochs=n_epochs,
                       validation_data=(xval, yval_bit),
                       callbacks=callbacks)

        if not self.predict_kappa:
            self.finetune_kappa(xval, yval_bit)

        return

    def save_weights(self, path):

        self.model.save_weights(path)

    def load_weights(self, path):

        self.model.load_weights(path)

    def finetune_kappa(self, x, y_bit, max_kappa=1000.0, verbose=False):
        ytr_preds_bit = self.model.predict(x)[:, 0:2]
        kappa_vals = np.arange(0, max_kappa, 1.0)
        log_likelihoods = np.zeros(kappa_vals.shape)
        for i, kappa_val in enumerate(kappa_vals):
            kappa_preds = np.ones([x.shape[0], 1]) * kappa_val
            log_likelihoods[i] = np.mean(von_mises_log_likelihood_np(y_bit, ytr_preds_bit, kappa_preds))
            if verbose:
                print("kappa: %f, log-likelihood: %f" % (kappa_val, log_likelihoods[i]))
        max_ix = np.argmax(log_likelihoods)
        self.fixed_kappa_value = kappa_vals[max_ix]
        if verbose:
            print("best kappa : %f" % self.fixed_kappa_value)
        return self.fixed_kappa_value

    def evaluate(self, x, ytrue_deg, data_part, return_per_image=False):

        ytrue_bit = deg2bit(ytrue_deg)
        ypreds = self.model.predict(x)
        ypreds_bit = ypreds[:, 0:2]
        ypreds_deg = bit2deg(ypreds_bit)

        if self.predict_kappa:
            kappa_preds = ypreds[:, 2:]
        else:
            kappa_preds = np.ones([ytrue_deg.shape[0], 1]) * self.fixed_kappa_value

        loss = maad_from_deg(ypreds_deg, ytrue_deg)

        results = dict()

        results['maad_loss'] = float(np.mean(loss))
        results['maad_loss_sem'] = float(sem(loss))
        print("MAAD error (%s) : %f pm %fSEM" % (data_part,
                                                results['maad_loss'],
                                                results['maad_loss_sem']))

        results['mean_kappa'] = float(np.mean(kappa_preds))
        results['std_kappa'] = float(np.std(kappa_preds))

        log_likelihoods = von_mises_log_likelihood_np(ytrue_bit, ypreds_bit, kappa_preds)

        results['log_likelihood_mean'] = float(np.mean(log_likelihoods))
        results['log_likelihood_sem'] = float(sem(log_likelihoods, axis=None))
        print("log-likelihood (%s) : %f pm %fSEM" % (data_part,
                                                    results['log_likelihood_mean'],
                                                    results['log_likelihood_sem']))

        if return_per_image:
            results['point_preds'] = bit2deg(deg2bit(ypreds_deg))
            results['maad'] = loss
            results['log_likelihood'] = log_likelihoods

        return results

    def pdf(self, x, x_vals):

        n_images = x.shape[0]

        x_vals_tiled = np.ones(n_images)

        preds = self.model.predict(x)
        mu_preds_bit = preds[:, 0:2]

        if self.predict_kappa:
            kappa_preds = preds[:, 2:]
        else:
            kappa_preds = np.ones([x.shape[0], 1]) * self.fixed_kappa_value

        log_likelihoods = np.zeros([n_images, len(x_vals)])

        for xid, xval in enumerate(x_vals):

            x_bit = rad2bit(x_vals_tiled*xval)

            log_likelihoods[:, xid] = np.exp(np.squeeze(von_mises_log_likelihood_np(x_bit, mu_preds_bit, kappa_preds)))

        return log_likelihoods
