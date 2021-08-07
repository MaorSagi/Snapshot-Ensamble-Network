import sys

import pandas as pd
import os
import time
from datetime import datetime

from scipy.stats import mannwhitneyu
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

from Evaluator import analyze_results
from SnapshotEnsemble import random_search, build_model, train_model, load_models, model_prediction
from consts import HYPER_TUNING, algorithm_tested, other_algorithm, DEBUG_ON, CV_OFF, \
    param_grid_snap, num_of_iter, param_grid_other, datasets_dicts, parent_dir_path, results_df_columns, \
    stats_results_df_columns, alpha, metrics
from DataLoader import load_data
import warnings
import os


def log(dataset_name, msg):
    with open(parent_dir_path + "/log_" + dataset_name + '/' + dataset_name + '.log', 'a+') as f:
        f.write(msg + '\n')


def run(input):
    dataset_name = input[0]
    # main

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # os.makedirs(parent_dir_path + "/snapshots_" + dataset_name, exist_ok=True)
    warnings.filterwarnings("ignore")
    try:
        results_df = pd.DataFrame([], columns=results_df_columns)
        # Load
        dataset_dict = datasets_dicts[dataset_name]
        X, y, df = load_data(dataset_name, dataset_dict)
        if DEBUG_ON:
            print("***********************************************************************")
        # print(dataset_name, " Dataset Loaded, Shape: ", df.shape)
        # log.info("Dataset Loaded, Shape: " + str(df.shape))
        log(dataset_name, "Dataset Loaded, Shape: " + str(df.shape))
        n_classes = 1 if len(y.unique()) == 2 else len(y.unique())
        # if DEBUG_ON:
        log(dataset_name, "Number of classes in dataset: " + str(n_classes))
        # log.info("Number of classes in dataset: " + str(n_classes))
        # print("Number of classes in dataset: ", n_classes)
        categorical_integers = dataset_dict["categorical_integers"]
        categorical_strings = dataset_dict["categorical_strings"]
        le_dict = {}
        X_encoded = X.copy()
        for feature in categorical_strings:
            le_dict[feature] = LabelEncoder()
            X_encoded[feature] = le_dict[feature].fit_transform(X[feature])
        X = X_encoded
        numericals = list(filter(lambda col: col not in (categorical_integers + categorical_strings), X.columns))
        i = 0
        cv_outer = StratifiedKFold(n_splits=10)
        for train_index, test_index in cv_outer.split(X, y):
            i += 1
            log(dataset_name, "Outer-CV Iter " + str(i))
            # log.info("Outer-CV Iter " + str(i))
            # print(dataset_name + ": Outer-CV Iter ", str(i))
            if CV_OFF and i > 1:
                break

            experiment_record_snap = {"Timestamp": datetime.now(),
                                      "Dataset Name": dataset_name,
                                      "Cross Validation": i,
                                      "Algorithm Name": algorithm_tested}

            # Split data
            X_train, X_test = X.loc[X.index.isin(train_index)], X.loc[X.index.isin(test_index)]
            y_train, y_test = y.loc[y.index.isin(train_index)], y.loc[y.index.isin(test_index)]

            # if DEBUG_ON:
            # log.info("Number of samples in train set: " + str(
            #     len(X_train)) + '\n' + "Number of  samples in test set: " + str(len(X_test)))
            log(dataset_name, "Number of samples in train set: " + str(
                len(X_train)) + '\n' + "Number of  samples in test set: " + str(len(X_test)))
            # print("Number of samples in train set: ", len(X_train))
            # print("Number of  samples in test set: ", len(X_test))

            # Hyperparameters Tuning
            if HYPER_TUNING:
                hyperparameters = random_search(X_train, y_train, X, param_grid_snap,
                                                categorical_integers + categorical_strings, numericals, n_classes,
                                                dataset_name,
                                                num_of_iter=num_of_iter)
                if DEBUG_ON:
                    print("Best Params: ", hyperparameters)
                log(dataset_name, "Best Params " + algorithm_tested + ": " + str(hyperparameters))
                # log.info("Best Params " + algorithm_tested + ": " + str(hyperparameters))
            # Build
            model = build_model(categorical_integers + categorical_strings, numericals, n_classes, X)
            #         if DEBUG_ON:
            #             print(model.summary())

            # Train
            # print('Train')
            start_train = time.time()
            if HYPER_TUNING:
                results = train_model(model, X_train, y_train, n_classes, hyperparameters['N_top_Models'],
                                      dataset_name,
                                      hyperparameters['batch_size'],
                                      hyperparameters['M_models'], hyperparameters['ephocs_cycle'],
                                      hyperparameters['lr'])
            else:
                results = train_model(model, X_train, y_train, n_classes, num_of_models, dataset_name)
            #         plot_results(results,dataset_name)
            end_train = time.time()
            training_time = end_train - start_train  # seconds
            if DEBUG_ON:
                print("Training time of our model: %d seconds" % (training_time))
            # print('Test')
            # Eval
            if HYPER_TUNING:
                num_of_models = hyperparameters['N_top_Models']
                M_models = hyperparameters['M_models']
            models = load_models(categorical_integers + categorical_strings, numericals, n_classes, X, M_models,
                                 num_of_models, dataset_name)
            start_inference = time.time()
            y_pred = model_prediction(models, X_test, len(X_test), n_classes)
            end_inference = time.time()
            inference_time = end_inference - start_inference  # seconds
            inference_time_for_1000 = (1000 / len(X_test)) * inference_time
            if DEBUG_ON:
                print("Inference time for 1000 instances of our model: %d seconds" % inference_time_for_1000)
                print("Results for our model:")
            accuracy, tpr, fpr, precision, roc_auc, auprc = analyze_results(y_test, y_pred, n_classes)
            experiment_record_snap.update({"Accuracy": accuracy,
                                           "TPR": tpr, "FPR": fpr,
                                           "Precision": precision,
                                           "AUC": roc_auc, "AUPRC": auprc,
                                           "Training Time": training_time,
                                           "Inference Time": inference_time_for_1000})
            results_df = results_df.append(experiment_record_snap, ignore_index=True)

            experiment_record_other = {"Timestamp": datetime.now(),
                                       "Dataset Name": dataset_name,
                                       "Cross Validation": i,
                                       "Algorithm Name": other_algorithm}

            clf = RandomForestClassifier()

            if HYPER_TUNING:
                rf_random = RandomizedSearchCV(estimator=clf, param_distributions=param_grid_other, n_iter=50, cv=3,
                                               n_jobs=-1)
                rf_random.fit(X_train, y_train)
                hyperparameters = rf_random.best_params_
                log(dataset_name, "Best Params " + other_algorithm + ": " + str(hyperparameters))
                # log.info("Best Params " + other_algorithm + ": " + str(hyperparameters))
                clf = RandomForestClassifier(n_estimators=hyperparameters["n_estimators"],
                                             max_features=hyperparameters["max_features"],
                                             max_depth=hyperparameters["max_depth"])
                if DEBUG_ON:
                    print("Best Params: ", hyperparameters)

            start_train = time.time()
            clf.fit(X_train, y_train)
            end_train = time.time()
            training_time = end_train - start_train  # seconds
            if DEBUG_ON:
                print("Training time of compared model: %d seconds" % (training_time))

            start_inference = time.time()
            y_pred = clf.predict(X_test)
            end_inference = time.time()
            inference_time = end_inference - start_inference  # seconds
            inference_time_for_1000 = (1000 / len(X_test)) * inference_time
            if DEBUG_ON:
                print("Inference time for 1000 instances of compared model: %d seconds" % inference_time_for_1000)
                print("Results for compared model:")
            accuracy, tpr, fpr, precision, roc_auc, auprc = analyze_results(y_test, y_pred, n_classes)
            experiment_record_other.update({"Accuracy": accuracy,
                                            "TPR": tpr, "FPR": fpr,
                                            "Precision": precision,
                                            "AUC": roc_auc, "AUPRC": auprc,
                                            "Training Time": training_time,
                                            "Inference Time": inference_time_for_1000})
            results_df = results_df.append(experiment_record_other, ignore_index=True)

        # write to file results
        results_df.to_csv(parent_dir_path + "/Results " + dataset_name + ".csv")
        for metric in metrics:
            metric_snap = results_df.loc[
                (results_df["Algorithm Name"] == algorithm_tested) & (results_df["Dataset Name"] == dataset_name)][
                metric]
            metric_other = results_df.loc[
                (results_df["Algorithm Name"] == other_algorithm) & (results_df["Dataset Name"] == dataset_name)][
                metric]
            try:
                stat_less, p_less = mannwhitneyu(metric_snap, metric_other, alternative="less")
                stat_greater, p_greater = mannwhitneyu(metric_snap, metric_other, alternative="greater")
                stats_record_less = {"Timestamp": datetime.now(), "Metric": metric, "Dataset Name": dataset_name,
                                     "p-value": p_less, "alpha": alpha, "Less": 1}
                stats_record_greater = {"Timestamp": datetime.now(), "Metric": metric, "Dataset Name": dataset_name,
                                        "p-value": p_greater, "alpha": alpha, "Less": 0}
                if DEBUG_ON:
                    print('Less: Statistics=%.3f, p=%.3f' % (stat_less, p_less))
                    print('Greater: Statistics=%.3f, p=%.3f' % (stat_greater, p_greater))
                if p_less > alpha:
                    stats_record_less.update({"Reject H0": 0})
                    if DEBUG_ON:
                        print('Less: Same distribution (fail to reject H0)')
                else:
                    stats_record_less.update({"Reject H0": 1})
                    if DEBUG_ON:
                        print('Less: Different distribution (reject H0)')
                if p_greater > alpha:
                    stats_record_greater.update({"Reject H0": 0})
                    if DEBUG_ON:
                        print('Greater: Same distribution (fail to reject H0)')
                else:
                    stats_record_greater.update({"Reject H0": 1})
                    if DEBUG_ON:
                        print('Greater: Different distribution (reject H0)')
                stats_results_df = stats_results_df.append(stats_record_less, ignore_index=True)
                stats_results_df = stats_results_df.append(stats_record_greater, ignore_index=True)

            except Exception as err:
                print(err)
        stats_results_df.to_csv(parent_dir_path + "/Stats Results " + dataset_name + ".csv")

    except Exception as err:
        log(dataset_name, "ERROR: " + str(err))
        print(dataset_name, "ERROR: " + str(err))
        # log.error(str(err))
        results_df.to_csv(parent_dir_path + "/ERROR:Results " + dataset_name + ".csv")
    # log.info('end')
    log(dataset_name, "end")
