from Classification import RandomClassification, XGBoostClassification

# RandomClassification("../../../bad_dir_1", "../../../good_dir_1",
#                      0.3, "tfidf", None, 10)


XGBoostClassification("../../../bad_dir_1", "../../../good_dir_1",
                      0.3, "tfidf", None, 10)
