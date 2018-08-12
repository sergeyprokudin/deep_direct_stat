import keras
import numpy as np
import pandas as pd
import warnings


class SideModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, model_name, model_to_save, save_path, save_weights_only=False):
        self.model_name = model_name
        self.model = model_to_save
        self.save_path = save_path
        self.save_weights_only = save_weights_only

    def on_train_begin(self, logs={}):
        self.epoch_id = 0
        self.min_val_loss = float("inf")

    def on_epoch_end(self, batch, logs={}):
        self.epoch_id += 1
        self.curr_val_loss = logs.get('val_loss')
        if self.curr_val_loss < self.min_val_loss:
            filepath = self.save_path.format(epoch=self.epoch_id, val_loss=self.curr_val_loss)
            print("val_loss improved from %f to %f, saving %s to %s" %
                  (self.min_val_loss, self.curr_val_loss, self.model_name, filepath))
            self.min_val_loss = self.curr_val_loss
            if self.save_weights_only:
                self.model.save_weights(filepath)
            else:
                self.model.save(filepath)


class EvalCVAEModel(keras.callbacks.Callback):
    """ Run CVAE evaluation on selected data

    """

    def __init__(self, x, y_deg, data_title, cvae_model, ckpt_path):
        self.x = x
        self.y_deg = y_deg
        self.data_title = data_title
        self.cvae_model = cvae_model
        self.ckpt_path = ckpt_path
        self.max_log_likelihood = float('-inf')

    def on_epoch_end(self, epoch, logs=None):
        results = self.cvae_model.evaluate(self.x, self.y_deg, self.data_title)
        if results['importance_log_likelihood'] > self.max_log_likelihood:
            print('max log likelihood improved from %f to %f' % (self.max_log_likelihood,
                                                                 results['importance_log_likelihood']))
            self.max_log_likelihood = results['importance_log_likelihood']
            self.model.save_weights(self.ckpt_path)
        print("Evaluation is done.")


class ModelCheckpointEveryNBatch(keras.callbacks.Callback):
    """Save the model after every n batches, based on validation loss

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of batches) between checkpoints.
    """

    def __init__(self, ckpt_path, log_path, xval, yval, verbose=0,
                 save_best_only=False, save_weights_only=False, period=1, patience=50):
        super(ModelCheckpointEveryNBatch, self).__init__()
        self.xval = xval
        self.yval = yval
        self.verbose = verbose
        self.ckpt_path = ckpt_path
        self.log_path = log_path
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.batches_since_last_save = 0
        self.min_val_loss = float('inf')
        self.n_steps = 0
        self.log_cols = ['train_step', 'val_loss', 'batch_loss']
        self.log_df = pd.DataFrame(columns=self.log_cols)
        self.n_epochs_no_improvement = 0
        self.patience = patience

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.n_steps += 1
        self.batches_since_last_save += 1
        if self.batches_since_last_save >= self.period:
            self.batches_since_last_save = 0
            filepath = self.ckpt_path
            if self.save_best_only:
                curr_batch_loss = logs.get('loss')
                curr_val_loss = self.model.evaluate(self.xval, self.yval, verbose=0)
                log_entry_np = np.asarray([self.n_steps, curr_val_loss, curr_batch_loss]).reshape([1, -1])
                log_entry_df = pd.DataFrame(log_entry_np, columns=self.log_cols)
                self.log_df = self.log_df.append(log_entry_df)
                self.log_df.to_csv(self.log_path, sep=';')
                if curr_val_loss < self.min_val_loss:
                    if self.verbose > 0:
                        print('Batch %05d: val_loss improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (batch, self.min_val_loss,
                                 curr_val_loss, filepath))
                    self.min_val_loss = curr_val_loss
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
                    self.n_epochs_no_improvement = 0
                else:
                    self.n_epochs_no_improvement += 1
                    if self.n_epochs_no_improvement > self.patience:
                        if self.verbose > 0:
                            print('Batch %05d: val_loss did not improve' % batch)
                        self.model.stop_training = True
                    if self.verbose > 0:
                        print('Batch %05d: val_loss did not improve' % batch)
                        print('number of steps with no improvement: %d' % self.n_epochs_no_improvement)

            else:
                if self.verbose > 0:
                    print('Batch %05d: saving model to %s' % (batch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)