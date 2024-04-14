import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import logging
import random
import CommonModules as CM
from joblib import dump, load
# from pprint import pprint
import json
import os
import sys
import warnings
from sklearn.exceptions import ConvergenceWarning

if not sys.warnoptions:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

logging.basicConfig(level=logging.INFO)
SVM_MODEL_NAME = "SVMModel.pkl"
XGB_MODEL_NAME = "XGBModel.pkl"
RF_MODEL_NAME = "RFModel.pkl"
DT_MODEL_NAME = "DTModel.pkl"


def GetDatafromSplit(split, good_label, oversample=False):
    '''
    Load the training and test data from the split files.
    Construct the feature vector for each sample.

    :param String split: specify the split number
    :param String good_label: specify the label for goodware
    :param Boolean oversample: specify whether to oversample the minority class

    :rtype Tuple: x_train, x_test, y_train, y_test, all_samples
    '''
    Logger = logging.getLogger('GetDatafromSplit.stdout')
    Logger.setLevel("INFO")

    train_df = pd.read_csv('train_{}.csv'.format(split))
    test_df = pd.read_csv('test_{}.csv'.format(split))

    x_test_samplenames = test_df["sha256"].tolist()
    test_df.loc[test_df["label"] == 0, "label"] = good_label
    y_test = test_df["label"].tolist()

    all_samples = pd.concat([train_df, test_df])["sha256"].tolist()

    # Separate majority and minority class samples
    majority_samples = train_df[train_df["label"] == 0]["sha256"].tolist()
    minority_samples = train_df[train_df["label"] == 1]["sha256"].tolist()

    if oversample:
        # Oversample the minority class to match the majority class
        minority_oversampled = resample(minority_samples,
                                        replace=True,      # Sample with replacement
                                        # Match majority class
                                        n_samples=len(majority_samples),
                                        random_state=0)    # Set random seed for reproducibility

        x_train_samplenames = np.concatenate(
            [minority_oversampled, majority_samples], axis=0)

        malware_labels = np.ones(len(minority_oversampled))
        good_labels = np.empty(len(majority_samples))
        good_labels.fill(good_label)
        y_train = np.concatenate((malware_labels, good_labels), axis=0)

    else:
        x_train_samplenames = np.concatenate(
            [minority_samples, majority_samples], axis=0)
        # label malware as 1 and goodware as -1
        malware_labels = np.ones(len(minority_samples))
        good_labels = np.empty(len(majority_samples))
        good_labels.fill(good_label)
        y_train = np.concatenate((malware_labels, good_labels), axis=0)
        Logger.info("Label array - generated")

    return x_train_samplenames, x_test_samplenames, y_train, y_test, all_samples


def SVMClassification(model, numTopFeats, oversample, split, generateExplaination):
    '''
    Train a classifier for classifying malwares and goodwares using Support Vector Machine technique.
    Compute the prediction accuracy and f1 score of the classifier.
    Modified from Jiachun's code.

    :param String MalwareCorpus: absolute path of the malware corpus
    :param String GoodwareCorpus: absolute path of the goodware corpus
    :param String FeatureOption: tfidf or binary, specify how to construct the feature vector

    :rtype String Report: result report
    '''

    Logger = logging.getLogger('SVMClf.stdout')
    Logger.setLevel("INFO")

    # step 2: split all samples to training set and test set
    x_train_samplenames, x_test_samplenames, y_train, y_test, all_samples = GetDatafromSplit(
        split=split, good_label=-1, oversample=oversample)

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=True)
    x = FeatureVectorizer.fit_transform(all_samples)

    x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
    x_test = FeatureVectorizer.transform(x_test_samplenames)

    # Logger.debug("Test set split = %s", TestSize)
    Logger.info("train-test split done")

    # step 3: train the model
    Logger.info("Perform Classification with SVM model")
    Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    T0 = time.time()
    if not model:
        Clf = GridSearchCV(LinearSVC(dual=True), Parameters,
                           cv=5, scoring='f1', n_jobs=-1)
        SVMModels = Clf.fit(x_train, y_train)
        Logger.info("Processing time to train and find best model with GridSearchCV is %s sec." % (
            round(time.time() - T0, 2)))
        BestModel = SVMModels.best_estimator_
        Logger.info("Best model Selected : {}".format(BestModel))
        print("The training time for random split classification is %s sec." %
              (round(time.time() - T0, 2)))
        dump(Clf, SVM_MODEL_NAME)
    else:
        SVMModels = load(model)
        BestModel = SVMModels.best_estimator

    # step 4: Evaluate the best model on test set
    T0 = time.time()
    y_pred = SVMModels.predict(x_test)
    print("The testing time for random split classification is %s sec." %
          (round(time.time() - T0, 2)))
    Accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy = {}".format(Accuracy))
    print(classification_report(y_test,
                                y_pred, labels=[1, -1],
                                target_names=['Malware', 'Goodware']))
    Report = "Test Set Accuracy = " + str(Accuracy) + "\n" + classification_report(y_test,
                                                                                   y_pred,
                                                                                   labels=[
                                                                                       1, -1],
                                                                                   target_names=['Malware',
                                                                                                 'Goodware'])

    if generateExplaination == False:
        return Report
    else:
        # pointwise multiplication between weight and feature vect
        w = BestModel.coef_
        w = w[0].tolist()
        v = x_test.toarray()
        vocab = FeatureVectorizer.get_feature_names_out()
        explanations = {os.path.basename(s): {} for s in x_test_samplenames}
        for i in range(v.shape[0]):
            wx = v[i, :] * w
            wv_vocab = zip(wx, vocab)
            wv_vocab = list(wv_vocab)
            if y_pred[i] == 1:
                wv_vocab.sort(reverse=True)
                # print "pred: {}, org: {}".format(y_pred[i],y_test[i])
                # pprint(wv_vocab[:10])
                explanations[os.path.basename(
                    x_test_samplenames[i])]['top_features'] = wv_vocab[:numTopFeats]
            elif y_pred[i] == -1:
                wv_vocab.sort()
                # print "pred: {}, org: {}".format(y_pred[i],y_test[i])
                # pprint(wv_vocab[-10:])
                explanations[os.path.basename(
                    x_test_samplenames[i])]['top_features'] = wv_vocab[-numTopFeats:]
            explanations[os.path.basename(
                x_test_samplenames[i])]['original_label'] = y_test[i]
            explanations[os.path.basename(
                x_test_samplenames[i])]['predicted_label'] = y_pred[i]

        with open('explanations_RC.json', 'w') as FH:
            json.dump(explanations, FH, indent=4)

    # return TestLabels, PredictedLabels
    return Report


def XGBoostClassification(model, numTopFeats, oversample, split, generateExplaination):
    '''
    Train a classifier for classifying malwares and goodwares using Support Vector Machine technique.
    Compute the prediction accuracy and f1 score of the classifier.
    Modified from Jiachun's code.

    :param String MalwareCorpus: absolute path of the malware corpus
    :param String GoodwareCorpus: absolute path of the goodware corpus
    :param String FeatureOption: tfidf or binary, specify how to construct the feature vector

    :rtype String Report: result report
    '''
    Logger = logging.getLogger('XGBClf.stdout')
    Logger.setLevel("INFO")

    x_train_samplenames, x_test_samplenames, y_train, y_test, all_samples = GetDatafromSplit(
        split=split, good_label=0, oversample=oversample)

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=True)
    x = FeatureVectorizer.fit_transform(all_samples)
    x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
    x_test = FeatureVectorizer.transform(x_test_samplenames)
    Logger.info("train-test split done")

    T0 = time.time()
    if not model:
        Clf = GridSearchCV(XGBClassifier(booster="gblinear"), {},
                           cv=5, scoring='f1')
        xgb = Clf.fit(x_train, y_train)
        Logger.info("Processing time to train and find best model with GridSearchCV is %s sec." % (
            round(time.time() - T0, 2)))
        BestModel = xgb.best_estimator_
        Logger.info("Best model Selected : {}".format(BestModel))
        print("The training time for random split classification is %s sec." %
              (round(time.time() - T0, 2)))
        dump(Clf, XGB_MODEL_NAME)

    else:
        SVMModels = load(model)
        BestModel = SVMModels.best_estimator

    # step 4: Evaluate the best model on test set
    T0 = time.time()
    y_pred = xgb.predict(x_test)

    print("The testing time for random split classification is %s sec." %
          (round(time.time() - T0, 2)))
    Accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy = {}".format(Accuracy))

    print(classification_report(y_test,
                                y_pred, labels=[1, 0],
                                target_names=['Malware', 'Goodware']))

    if generateExplaination == False:
        return Report
    else:

        # pointwise multiplication between weight and feature vect
        w = BestModel.coef_
        w = w[0].tolist()
        v = x_test.toarray()
        vocab = FeatureVectorizer.get_feature_names_out()
        explanations = {os.path.basename(s): {} for s in x_test_samplenames}
        for i in range(v.shape[0]):
            wx = v[i, :] * w
            wv_vocab = zip(wx, vocab)
            wv_vocab = list(wv_vocab)
            if y_pred[i] == 1:
                wv_vocab.sort(reverse=True)
                explanations[os.path.basename(
                    x_test_samplenames[i])]['top_features'] = wv_vocab[:numTopFeats]
            elif y_pred[i] == 0:
                wv_vocab.sort()
                explanations[os.path.basename(
                    x_test_samplenames[i])]['top_features'] = wv_vocab[-numTopFeats:]
            explanations[os.path.basename(
                x_test_samplenames[i])]['original_label'] = y_test[i]
            explanations[os.path.basename(
                x_test_samplenames[i])]['predicted_label'] = int(y_pred[i])

        with open('explanations_RC.json', 'w') as FH:
            json.dump(explanations, FH, indent=4, default=str)

    return


def RFClassification(model, numTopFeats, oversample, split, generateExplaination):
    '''
    Train a classifier for classifying malwares and goodwares using Random Forest technique.
    Compute the prediction accuracy and f1 score of the classifier.
    Modified from Jiachun's code.

    :param String MalwareCorpus: absolute path of the malware corpus
    :param String GoodwareCorpus: absolute path of the goodware corpus
    :param String FeatureOption: tfidf or binary, specify how to construct the feature vector

    :rtype String Report: result report
    '''

    Logger = logging.getLogger('RFClf.stdout')
    Logger.setLevel("INFO")

    x_train_samplenames, x_test_samplenames, y_train, y_test, all_samples = GetDatafromSplit(
        split=split, good_label=-1, oversample=oversample)

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=True)
    x = FeatureVectorizer.fit_transform(all_samples)
    x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
    x_test = FeatureVectorizer.transform(x_test_samplenames)
    Logger.info("train-test split done")

    # step 3: train the model
    Logger.info("Perform Classification with RF model")
    # Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    T0 = time.time()
    if not model:
        rf = RandomForestClassifier(random_state=42)
        Clf = GridSearchCV(rf, {},
                           cv=5, scoring='f1', n_jobs=-1)
        RFModels = Clf.fit(x_train, y_train)
        Logger.info("Processing time to train and find best model with GridSearchCV is %s sec." % (
            round(time.time() - T0, 2)))
        BestModel = RFModels.best_estimator_
        Logger.info("Best model Selected : {}".format(BestModel))
        print("The training time for random split classification is %s sec." %
              (round(time.time() - T0, 2)))
        dump(Clf, RF_MODEL_NAME)
    else:
        RFModels = load(model)
        BestModel = RFModels.best_estimator

    # step 4: Evaluate the best model on test set
    T0 = time.time()
    y_pred = RFModels.predict(x_test)
    print("The testing time for random split classification is %s sec." %
          (round(time.time() - T0, 2)))
    Accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy = {}".format(Accuracy))
    print(classification_report(y_test,
                                y_pred, labels=[1, -1],
                                target_names=['Malware', 'Goodware']))
    Report = "Test Set Accuracy = " + str(Accuracy) + "\n" + classification_report(y_test,
                                                                                   y_pred,
                                                                                   labels=[
                                                                                       1, -1],
                                                                                   target_names=['Malware',
                                                                                                 'Goodware'])
    if generateExplaination == False:
        return Report
    else:

        # pointwise multiplication between weight and feature vect
        w = BestModel.feature_importances_
        w = w[0].tolist()
        v = x_test.toarray()
        vocab = FeatureVectorizer.get_feature_names_out()
        explanations = {os.path.basename(s): {} for s in x_test_samplenames}
        for i in range(v.shape[0]):
            wx = v[i, :] * w
            wv_vocab = zip(wx, vocab)
            wv_vocab = list(wv_vocab)
            if y_pred[i] == 1:
                wv_vocab.sort(reverse=True)
                # print "pred: {}, org: {}".format(y_pred[i],y_test[i])
                # pprint(wv_vocab[:10])
                explanations[os.path.basename(
                    x_test_samplenames[i])]['top_features'] = wv_vocab[:numTopFeats]
            elif y_pred[i] == -1:
                wv_vocab.sort()
                # print "pred: {}, org: {}".format(y_pred[i],y_test[i])
                # pprint(wv_vocab[-10:])
                explanations[os.path.basename(
                    x_test_samplenames[i])]['top_features'] = wv_vocab[-numTopFeats:]
            explanations[os.path.basename(
                x_test_samplenames[i])]['original_label'] = y_test[i]
            explanations[os.path.basename(
                x_test_samplenames[i])]['predicted_label'] = y_pred[i]

        with open('explanations_RC.json', 'w') as FH:
            json.dump(explanations, FH, indent=4)

    # return TestLabels, PredictedLabels
    return


def DTClassification(model, numTopFeats, oversample, split, generateExplaination):
    '''
    Train a classifier for classifying malwares and goodwares using Decision Tree technique.
    Compute the prediction accuracy and f1 score of the classifier.
    Modified from Jiachun's code.

    :param String MalwareCorpus: absolute path of the malware corpus
    :param String GoodwareCorpus: absolute path of the goodware corpus
    :param String FeatureOption: tfidf or binary, specify how to construct the feature vector

    :rtype String Report: result report
    '''

    Logger = logging.getLogger('DTClf.stdout')
    Logger.setLevel("INFO")

    x_train_samplenames, x_test_samplenames, y_train, y_test, all_samples = GetDatafromSplit(
        split=split, good_label=-1, oversample=oversample)

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=True)
    x = FeatureVectorizer.fit_transform(all_samples)
    x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
    x_test = FeatureVectorizer.transform(x_test_samplenames)
    Logger.info("train-test split done")

    # step 3: train the model
    Logger.info("Perform Classification with DT model")
    # Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    T0 = time.time()
    if not model:
        dt = DecisionTreeClassifier(random_state=20)
        Clf = GridSearchCV(dt, {},
                           cv=5, scoring='f1', n_jobs=-1)
        DTModels = Clf.fit(x_train, y_train)
        Logger.info("Processing time to train and find best model with GridSearchCV is %s sec." % (
            round(time.time() - T0, 2)))
        BestModel = DTModels.best_estimator_
        Logger.info("Best model Selected : {}".format(BestModel))
        print("The training time for random split classification is %s sec." %
              (round(time.time() - T0, 2)))
        dump(Clf, DT_MODEL_NAME)
    else:
        DTModels = load(model)
        BestModel = DTModels.best_estimator

    # step 4: Evaluate the best model on test set
    T0 = time.time()
    y_pred = DTModels.predict(x_test)
    print("The testing time for random split classification is %s sec." %
          (round(time.time() - T0, 2)))
    Accuracy = accuracy_score(y_test, y_pred)
    print("Test Set Accuracy = {}".format(Accuracy))
    print(classification_report(y_test,
                                y_pred, labels=[1, -1],
                                target_names=['Malware', 'Goodware']))
    Report = "Test Set Accuracy = " + str(Accuracy) + "\n" + classification_report(y_test,
                                                                                   y_pred,
                                                                                   labels=[
                                                                                       1, -1],
                                                                                   target_names=['Malware',
                                                                                                 'Goodware'])
    if generateExplaination == False:
        return Report
    else:

        # pointwise multiplication between weight and feature vect
        w = BestModel.feature_importances_
        w = w[0].tolist()
        v = x_test.toarray()
        vocab = FeatureVectorizer.get_feature_names_out()
        explanations = {os.path.basename(s): {} for s in x_test_samplenames}
        for i in range(v.shape[0]):
            wx = v[i, :] * w
            wv_vocab = zip(wx, vocab)
            wv_vocab = list(wv_vocab)
            if y_pred[i] == 1:
                wv_vocab.sort(reverse=True)
                # print "pred: {}, org: {}".format(y_pred[i],y_test[i])
                # pprint(wv_vocab[:10])
                explanations[os.path.basename(
                    x_test_samplenames[i])]['top_features'] = wv_vocab[:numTopFeats]
            elif y_pred[i] == -1:
                wv_vocab.sort()
                # print "pred: {}, org: {}".format(y_pred[i],y_test[i])
                # pprint(wv_vocab[-10:])
                explanations[os.path.basename(
                    x_test_samplenames[i])]['top_features'] = wv_vocab[-numTopFeats:]
            explanations[os.path.basename(
                x_test_samplenames[i])]['original_label'] = y_test[i]
            explanations[os.path.basename(
                x_test_samplenames[i])]['predicted_label'] = y_pred[i]

        with open('explanations_RC.json', 'w') as FH:
            json.dump(explanations, FH, indent=4)

    # return TestLabels, PredictedLabels
    return
