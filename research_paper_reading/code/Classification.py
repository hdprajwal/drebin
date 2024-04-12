import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor

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


def SVMClassification(MalwareCorpus, GoodwareCorpus, TestSize, FeatureOption, Model, NumTopFeats):
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

    # step 1: creating feature vector
    Logger.debug("Loading Malware and Goodware Sample Data")
    AllMalSamples = CM.ListFiles(MalwareCorpus, "")
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, "")
    AllSampleNames = AllMalSamples + AllGoodSamples
    Logger.info("Loaded samples")

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=FeatureOption)
    x = FeatureVectorizer.fit_transform(AllMalSamples + AllGoodSamples)

    # label malware as 1 and goodware as -1
    Mal_labels = np.ones(len(AllMalSamples))
    Good_labels = np.empty(len(AllGoodSamples))
    Good_labels.fill(-1)
    y = np.concatenate((Mal_labels, Good_labels), axis=0)
    Logger.info("Label array - generated")

    # step 2: split all samples to training set and test set
    x_train_samplenames, x_test_samplenames, y_train, y_test = train_test_split(AllSampleNames, y, test_size=TestSize,
                                                                                random_state=random.randint(0, 100))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TestSize,
    #                                             random_state=random.randint(0, 100))
    x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
    x_test = FeatureVectorizer.transform(x_test_samplenames)
    Logger.debug("Test set split = %s", TestSize)
    Logger.info("train-test split done")

    # step 3: train the model
    Logger.info("Perform Classification with SVM Model")
    Parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    T0 = time.time()
    if not Model:
        Clf = GridSearchCV(LinearSVC(dual=True), Parameters,
                           cv=5, scoring='f1', n_jobs=-1)
        SVMModels = Clf.fit(x_train, y_train)
        Logger.info("Processing time to train and find best model with GridSearchCV is %s sec." % (
            round(time.time() - T0, 2)))
        BestModel = SVMModels.best_estimator_
        Logger.info("Best Model Selected : {}".format(BestModel))
        print("The training time for random split classification is %s sec." %
              (round(time.time() - T0, 2)))
        dump(Clf, SVM_MODEL_NAME)
    else:
        SVMModels = load(Model)
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
                x_test_samplenames[i])]['top_features'] = wv_vocab[:NumTopFeats]
        elif y_pred[i] == -1:
            wv_vocab.sort()
            # print "pred: {}, org: {}".format(y_pred[i],y_test[i])
            # pprint(wv_vocab[-10:])
            explanations[os.path.basename(
                x_test_samplenames[i])]['top_features'] = wv_vocab[-NumTopFeats:]
        explanations[os.path.basename(
            x_test_samplenames[i])]['original_label'] = y_test[i]
        explanations[os.path.basename(
            x_test_samplenames[i])]['predicted_label'] = y_pred[i]

    with open('explanations_RC.json', 'w') as FH:
        json.dump(explanations, FH, indent=4)

    # return TestLabels, PredictedLabels
    return Report


def XGBoostClassification(MalwareCorpus, GoodwareCorpus, TestSize, FeatureOption, Model, NumTopFeats):
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

    # step 1: creating feature vector
    Logger.debug("Loading Malware and Goodware Sample Data")
    AllMalSamples = CM.ListFiles(MalwareCorpus, "")
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, "")
    AllSampleNames = AllMalSamples + AllGoodSamples
    Logger.info("Loaded samples")

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=True)
    x = FeatureVectorizer.fit_transform(AllMalSamples + AllGoodSamples)

    # label malware as 1 and goodware as -1
    Mal_labels = np.ones(len(AllMalSamples))
    Good_labels = np.empty(len(AllGoodSamples))
    Good_labels.fill(0)
    y = np.concatenate((Mal_labels, Good_labels), axis=0)
    Logger.info("Label array - generated")

    # step 2: split all samples to training set and test set
    x_train_samplenames, x_test_samplenames, y_train, y_test = train_test_split(AllSampleNames, y, test_size=TestSize,
                                                                                random_state=random.randint(0, 100))

    x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
    x_test = FeatureVectorizer.transform(x_test_samplenames)
    Logger.debug("Test set split = %s", TestSize)
    Logger.info("train-test split done")

    T0 = time.time()
    if not Model:
        # xgb = XGBClassifier(n_estimators=3, booster='gblinear', n_jobs=4)
        # xgb.fit(x_train, y_train)
        # Logger.info("Processing time to train and find best model with XGB is %s sec." % (
        #     round(time.time() - T0, 2)))
        # Logger.info("Saving Model to disk")
        # dump(xgb, XGB_MODEL_NAME)

        Clf = GridSearchCV(XGBClassifier(booster="gblinear"), {},
                           cv=5, scoring='f1')
        xgb = Clf.fit(x_train, y_train)
        Logger.info("Processing time to train and find best model with GridSearchCV is %s sec." % (
            round(time.time() - T0, 2)))
        BestModel = xgb.best_estimator_
        Logger.info("Best Model Selected : {}".format(BestModel))
        print("The training time for random split classification is %s sec." %
              (round(time.time() - T0, 2)))
        dump(Clf, XGB_MODEL_NAME)

    else:
        SVMModels = load(Model)
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
                x_test_samplenames[i])]['top_features'] = wv_vocab[:NumTopFeats]
        elif y_pred[i] == 0:
            wv_vocab.sort()
            explanations[os.path.basename(
                x_test_samplenames[i])]['top_features'] = wv_vocab[-NumTopFeats:]
        explanations[os.path.basename(
            x_test_samplenames[i])]['original_label'] = y_test[i]
        explanations[os.path.basename(
            x_test_samplenames[i])]['predicted_label'] = int(y_pred[i])

    with open('explanations_RC.json', 'w') as FH:
        json.dump(explanations, FH, indent=4, default=str)

    return