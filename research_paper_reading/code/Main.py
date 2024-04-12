import psutil
import argparse
import logging
from Classification import SVMClassification, XGBoostClassification
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('main.stdout')


def main(args):

    if args.classifier == "SVM":
        Logger.info("Using SVM Classifier")
        SVMClassification(args.maldir, args.gooddir,
                          args.testsize, True, args.model, args.numfeatforexp)
        Logger.info("SVM Classification Complete")
    elif args.classifier == "XGBoost":
        Logger.info("Using XGBoost Classifier")
        XGBoostClassification(args.maldir, args.gooddir,
                              args.testsize, "tfidf", args.model, args.numfeatforexp)
        Logger.info("XGB Classification Complete")
    else:
        Logger.error("Invalid Classifier")
        exit(1)


def ParseArgs():
    Args = argparse.ArgumentParser(
        description="Classification of Android Applications")
    Args.add_argument("--classifier", default="SVM",
                      help="Type of Classifier used to perform Classification ie. SVM, XGBoost")
    Args.add_argument("--maldir", default="../data/small_proto_apks/malware",
                      help="Absolute path to directory containing malware apks")
    Args.add_argument("--gooddir", default="../data/small_proto_apks/goodware",
                      help="Absolute path to directory containing benign apks")
    Args.add_argument("--testsize", type=float, default=0.3,
                      help="Size of the test set when split by Scikit Learn's Train Test Split module")
    Args.add_argument("--model",
                      help="Absolute path to the saved model file(.pkl extension)")
    Args.add_argument("--numfeatforexp", type=int, default=30,
                      help="Number of top features to show for each test sample")
    return Args.parse_args()


if __name__ == "__main__":
    main(ParseArgs())
