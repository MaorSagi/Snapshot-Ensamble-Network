import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, accuracy_score, \
    classification_report, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_sample_weight

from consts import DEBUG_ON


def get_rates(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    return TN, FP, FN, TP


def analyze_results(y_test, y_pred, n_classes, return_dict=False):
    y_proba = y_pred
    if n_classes != 1:
        y_pred = [(np.argmax(x) + 1) for x in y_pred]
        TN, FP, FN, TP = get_rates(y_test, y_pred)
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = report["accuracy"]
        tpr = report['weighted avg']['recall']
        precision = report["weighted avg"]["precision"]
        fpr = (FP / (FP + TN)).mean()
        roc_auc = roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr', labels=y_test.unique(),
                                sample_weight=sample_weight)
        auprc = average_precision_score(y_test, y_proba, average='weighted', sample_weight=sample_weight)

    else:
        y_pred = [round(x) for x in y_pred.squeeze()]
        TN, FP, FN, TP = get_rates(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        if "1" in report:
            tpr = report["1"]["recall"]
            precision = report["1"]["precision"]
        else:
            tpr = (TP / (TP + FN))[1]
            precision = (TP / (TP + FP))[1]
        fpr = (FP / (FP + TN))[1]
        accuracy = accuracy_score(y_test, y_pred)
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_test)
        roc_auc = roc_auc_score(y_test, y_proba.squeeze(), labels=y_test.unique(), sample_weight=sample_weight)
        precision_, recall_, _ = precision_recall_curve(y_test, y_proba.squeeze(), sample_weight=sample_weight)
        auprc = auc(recall_, precision_)
    if DEBUG_ON:
        print('Accuracy ', str(round(accuracy, 4)))
        print('TPR ', str(round(tpr, 4)))
        print('FPR ', str(round(fpr, 4)))
        print('Precision ', str(round(precision, 4)))
        print('AUC ', str(round(roc_auc, 4)))
        print('AUPRC ', str(round(auprc, 4)))
    if return_dict:
        return dict(accuracy=accuracy, tpr=tpr, fpr=fpr, precision=precision, roc_auc=roc_auc, auprc=auprc)
    return accuracy, tpr, fpr, precision, roc_auc, auprc
