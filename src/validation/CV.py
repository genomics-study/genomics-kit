from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import numpy as np

def validator(samples, labels, classifier, method):
    splits_no =  method.get_n_splits(samples, labels)
    splits = method.split(samples, labels)
    scores = []
    for split in splits:
        training_idx, test_idx = split
        training_samples, test_samples = samples[training_idx], samples[test_idx]
        training_labels, test_labels = labels[training_idx], labels[test_idx]
        classifier.fit(training_samples, training_labels)
        prediction = classifier.predict(test_samples)
        score = [
            np.mean(test_labels != prediction), # misclassification
            accuracy_score(test_labels, prediction), # 1 - misclassification
            precision_score(test_labels, prediction, average='micro'), 
            f1_score(test_labels, prediction, average='micro'),
            confusion_matrix(test_labels, prediction, labels=[0,1,2]) * splits_no, # confusion matrix scaled to match real samples number 
            confusion_matrix(test_labels, prediction, labels=[0,1,2]) / len(test_labels) # confusion matrix converted to probabilities 
                ]
        scores.append(score) # missclassification rate
    mean_score = np.mean(np.array(scores),axis=0)
    method_names = ["avg_missclass", "avg_acc", "avg_prec", "avg_F1", "avg_scaled_conf_mx", "avg_conf_prob_mx"]
    return dict(zip(method_names,mean_score))

# usage example
# validator(load_iris().data, 
#     load_iris().target, 
#     sklearn.neighbors.KNeighborsClassifier(7), 
#     sklearn.model_selection.KFold(n_splits=5, shuffle=True))