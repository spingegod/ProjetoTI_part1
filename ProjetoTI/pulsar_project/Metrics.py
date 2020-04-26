#Metrics:
#Precision the number of true positives divided by all positive predictions. 
#Recall is the number of true positives divided by the number of positive 
#values in the test data.
#F1 score the weighted average of precision and recall.

def check_accuracy(y_pred,y_true):
    n=0
    for i in range(len(y_true)):
        if y_pred[i]==y_true[i]:
            n=n+1
    acc = n/len(y_true)
    return acc

def precision(pred,test):
    true_positives = 0
    positive_predictions = (pred == 1).sum()
    for i in range(0,len(pred)):
        if pred[i] == test[i] and test[i] == 1:
            true_positives+=1
    return true_positives/positive_predictions
    
def recall(pred,test):
    true_positives = 0
    positives = (test == 1).sum()
    for i in range(0,len(pred)):
        if pred[i] == test[i] and test[i] == 1:
            true_positives+=1
    return true_positives/positives

def F1score(precision,recall):
    score = 2*(precision*recall)/(precision+recall)
    return score