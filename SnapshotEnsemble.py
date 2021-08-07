import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, concatenate
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import os
import time
import math
import random
import shutil

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
from tensorflow.keras.models import load_model

from sklearn.model_selection import StratifiedKFold

from Evaluator import analyze_results
from consts import DEBUG_ON, parent_dir_path


def log(dataset_name, msg):
    with open(parent_dir_path + "/log_" + dataset_name + '/' + dataset_name + '.log', 'a+') as f:
        f.write(msg + '\n')


def encode_feature(feature, series, is_categorical):
    # Create a lookup layer which will turn strings into integer indices
    if is_categorical:
        layer = IntegerLookup(output_mode="binary")
    else:
        layer = Normalization()
    layer.adapt(tf.constant(series))

    # Turn the string input into integer indices
    encoded_feature = layer(feature)
    return encoded_feature


def build_model(categorical_features, numericals, n_classes, X_train, metric="accuracy"):
    inputs = []
    encoded_features = []
    for feature in categorical_features:
        feature_input = keras.Input(shape=(1,), name=str(feature), dtype="int64")
        inputs.append(feature_input)
        encoded_features.append(encode_feature(feature_input, X_train[feature], is_categorical=True))

    for feature in numericals:
        feature_input = keras.Input(shape=(1,), name=str(feature))
        inputs.append(feature_input)
        encoded_features.append(encode_feature(feature_input, X_train[feature], is_categorical=False))

    model_input = concatenate(encoded_features)
    x = Dense(int(len(inputs) * 2.5), kernel_regularizer='l2', activation="relu")(model_input)
    x = Dropout(0.2)(x)
    output = Dense(n_classes, activation="sigmoid")(x)
    model = keras.Model(inputs, output)
    loss = "binary_crossentropy" if n_classes == 1 else "categorical_crossentropy"
    model.compile("sgd", loss, metrics=["accuracy"])
    return model


def load_models(categorical_features, numericals, n_classes, train_ds, M_models, num_of_models, dataset):
    members = []
    range_to_loop = range(M_models, M_models - num_of_models, -1) if (M_models - num_of_models) > 0 else range(M_models,
                                                                                                               0, -1)
    for i in range_to_loop:
        model = build_model(categorical_features, numericals, n_classes, train_ds)
        try:
            model.load_weights(filepath=parent_dir_path + "/snapshots_" + dataset + "/snapshot_" + str(i))
        except Exception as err:
            print(dataset, " ", i)
            print(err)
        if DEBUG_ON:
            print('load %d snapshot model' % (i))
        members.append(model)
    return members


def model_prediction(members, X_test, n_samples, n_classes):
    num_of_models = len(members)
    preds = np.zeros((n_samples, n_classes))
    for i in range(num_of_models):
        model = members[i]
        X_test_new = []
        for i in range(len(X_test.columns)):
            X_test_new.append(X_test[X_test.columns[i]])
        pred = model.predict(X_test_new)
        #       pred = [round(x) for x in pred]
        preds = np.sum([preds, np.array(pred)], axis=0)
    pred = preds / num_of_models
    return pred


def train_model(model, X_train, y_train, n_classes, num_of_models, dataset, batch_size=32, M_models=20, ephocs_cycle=50,
                lr=.01):
    se_callback = SnapshotEnsemble(n_models=M_models, num_of_models=num_of_models, n_epochs_per_model=ephocs_cycle,
                                   lr_max=lr, dataset=dataset)
    # fit model
    X_train_new = []
    for i in range(len(X_train.columns)):
        X_train_new.append(X_train[X_train.columns[i]])
    X_train = X_train_new
    if n_classes > 1:
        y_train = pd.get_dummies(y_train)
    else:
        y_train = y_train.values
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=se_callback.n_epochs_total,
                        callbacks=[se_callback], verbose=0)
    return history


def random_search(X, y, X_total, param_grid, categorical_features, numericals, n_classes, dataset, num_of_iter=50):
    best_hyperparameters = None
    best_score = 0
    for i in range(num_of_iter):
        i += 1
        log(dataset, "\tRandom Search Iter  " + str(i))
        # log.info("Random Search Iter  " + str(i))
        # print(dataset + ': Random Search Iter ', i)
        try:
            hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
            cv_inner = StratifiedKFold(n_splits=3)
            accuracy_fold_score = 0
            j = 0
            inner_iter = 0
            for train_index, test_index in cv_inner.split(X, y):
                try:
                    inner_iter += 1
                    j += 1
                    log(dataset, "\t\tRandom Search Inner-CV Iter  " + str(inner_iter))
                    # log.info("Random Search Inner-CV Iter  " + str(j))
                    # print(dataset + ': Random Search Inner-CV Iter ', j)

                    X_train, X_test = X.loc[X.index.isin(train_index)], X.loc[X.index.isin(test_index)]
                    y_train, y_test = y.loc[y.index.isin(train_index)], y.loc[y.index.isin(test_index)]
                    if DEBUG_ON:
                        print("Number of samples in train set: ", len(X_train))
                        print("Number of  samples in test set: ", len(X_test))
                    model = build_model(categorical_features, numericals, n_classes, X_total)
                    #             if DEBUG_ON:
                    #                 print(model.summary())
                    train_model(model, X_train, y_train, n_classes, hyperparameters['N_top_Models'], dataset,
                                hyperparameters['batch_size'], hyperparameters['M_models'],
                                hyperparameters['ephocs_cycle'], hyperparameters['lr'])
                    num_of_models = hyperparameters['N_top_Models']
                    models = load_models(categorical_features, numericals, n_classes, X_total,
                                         hyperparameters['M_models'], hyperparameters['N_top_Models'], dataset)
                    start_inference = time.time()
                    len_test = len(X_test)
                    y_pred = model_prediction(models, X_test, len_test, n_classes)
                    results_dict = analyze_results(y_test, y_pred, n_classes, return_dict=True)
                    score = results_dict["accuracy"]
                    accuracy_fold_score += score
                except Exception as err:
                    j -= 1
                    log(dataset,
                        "\t\tERROR: j = " + str(j) + " error: " + str(err) + ". Hyperparams: " + str(hyperparameters))
            if j != 0:
                accuracy_fold_score = accuracy_fold_score / j
            if accuracy_fold_score > best_score:
                best_hyperparameters = hyperparameters
                best_score = accuracy_fold_score
                log(dataset, "\tBest Params Snapshot Ensemble: " + str(hyperparameters) +
                    ", Best Score: " + str(best_score))
                # log.info(
                #     "Best Params Snapshot Ensemble: " + str(hyperparameters) +
                #     ", Best Score: " + str(best_score))
        except Exception as err:
            # log.error("ERROR: " + str(err) + ". Hyperparams: " + str(hyperparameters))
            log(dataset, "\tERROR: " + str(err) + ". Hyperparams: " + str(hyperparameters))
            if best_hyperparameters is None:
                best_hyperparameters = hyperparameters
    return best_hyperparameters


# this callback applies cosine annealing, saves snapshots and allows to load them
class SnapshotEnsemble(Callback):

    def __init__(self, n_models, num_of_models, n_epochs_per_model, lr_max, dataset):
        """
        n_models -- quantity of models (snapshots)
        n_epochs_per_model -- quantity of epoch for every model (snapshot)
        lr_max -- maximum learning rate (snapshot starter)
        """
        self.n_epochs_per_model = n_epochs_per_model
        self.n_models = n_models
        self.n_epochs_total = self.n_models * self.n_epochs_per_model
        self.lr_max = lr_max
        self.lrs = []
        self.num_of_models = num_of_models
        self.dataset = dataset
        self.__snapshot_name_fmt = parent_dir_path + "/snapshots_" + dataset

        if os.path.isdir(self.__snapshot_name_fmt):
            shutil.rmtree(self.__snapshot_name_fmt)
        # os.makedirs(self.__snapshot_name_fmt, exist_ok=True)

    # calculate learning rate for epoch
    def cosine_annealing(self, epoch):
        cos_inner = (math.pi * (epoch % self.n_epochs_per_model)) / self.n_epochs_per_model
        return self.lr_max / 2 * (math.cos(cos_inner) + 1)

    # when epoch begins update learning rate
    def on_epoch_begin(self, epoch, logs={}):
        # update learning rate
        lr = self.cosine_annealing(epoch)
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrs.append(lr)

    # when epoch ends check if there is a need to save a snapshot
    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % self.n_epochs_per_model == 0:
            # save model to file
            filename = self.__snapshot_name_fmt + "/snapshot_%d" % ((epoch + 1) // self.n_epochs_per_model)
            # save_model(self.model,filename,save_format='tf')
            # print(str((epoch + 1) // self.n_epochs_per_model) ),">",str((self.n_models - self.num_of_models))
            if ((epoch + 1) // self.n_epochs_per_model) > (self.n_models - self.num_of_models):
                # print('save')
                # print(filename)
                self.model.save_weights(filename, save_format='tf')
                if DEBUG_ON:
                    print('Epoch %d: snapshot saved to %s' % (epoch, filename))

    # load all snapshots after training
    def load_ensemble(self):
        models = []
        for i in range(self.n_models):
            models.append(load_model(self.__snapshot_name_fmt + "/snapshot_%d" % (i + 1)))
        return models
