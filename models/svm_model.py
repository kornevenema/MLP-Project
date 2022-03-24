import abc
from sklearn import svm, metrics


class svm_baseline:

    def __init__(self, kernel="linear", **kwargs):
        self.classifier = svm.SVC(kernel=kernel, **kwargs)
        self.preds = None

    def fit(self, train_features, train_labels):
        self.classifier.fit(train_features, train_labels)

    def predict(self, test_features):
        self.preds = self.classifier.predict(test_features)

    def report_scores(self, true_labels):
        return metrics.classification_report(true_labels, self.preds)
