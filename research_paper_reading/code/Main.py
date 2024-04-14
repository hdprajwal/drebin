import psutil
import argparse
import logging
from Classification import SVMClassification, XGBoostClassification, RFClassification, DTClassification
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('main.stdout')


def main(args):

    if args.classifier == "SVM":
        Logger.info("Using SVM Classifier")
        print(args)
        SVMClassification(args.model, args.numfeatforexp,
                          args.oversample, args.split, args.genexplaination)
        Logger.info("SVM Classification Complete")
    elif args.classifier == "XGBoost":
        Logger.info("Using XGBoost Classifier")
        XGBoostClassification(args.model, args.numfeatforexp,
                              args.oversample, args.split, args.genexplaination)
        Logger.info("XGB Classification Complete")
    elif args.classifier == "RF":
        Logger.info("Using Random Forest Classifier")
        RFClassification(args.model, args.numfeatforexp,
                         args.oversample, args.split, args.genexplaination)
        Logger.info("RF Classification Complete")
    elif args.classifier == "DT":
        Logger.info("Using Decision Tree Classifier")
        DTClassification(args.model, args.numfeatforexp,
                         args.oversample, args.split, args.genexplaination)
        Logger.info("DT Classification Complete")
    else:
        Logger.error("Invalid Classifier")
        exit(1)


def ParseArgs():
    Args = argparse.ArgumentParser(
        description="Classification of Android Applications")
    Args.add_argument("--classifier", default="SVM",
                      help="Type of Classifier used to perform Classification ie. SVM, XGBoost")
    Args.add_argument("--split", default=1,
                      help="Split of the dataset to use for training and testing")
    Args.add_argument("--oversample", action=argparse.BooleanOptionalAction, type=bool, default=False,
                      help="Whether to oversample the minority class")
    Args.add_argument("--genexplaination", action=argparse.BooleanOptionalAction, type=bool, default=False,
                      help="Whether to generate explanations for the classification")
    Args.add_argument("--model",
                      help="Absolute path to the saved model file(.pkl extension)")
    Args.add_argument("--numfeatforexp", type=int, default=30,
                      help="Number of top features to show for each test sample")
    return Args.parse_args()


if __name__ == "__main__":
    main(ParseArgs())
