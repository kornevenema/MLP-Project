from sklearn import tree, metrics


class DecisionTreeBaseline:
    def __init__(self, criterion='entropy'):
        self.preds = None
        self.classifier = tree.DecisionTreeClassifier(criterion=criterion)

    def fit(self, train_features, train_labels):
        self.classifier.fit(train_features, train_labels)

    def predict(self, test_features):
        self.preds = self.classifier.predict(test_features)

    def report_scores(self, true_labels):
        return metrics.classification_report(true_labels, self.preds)
