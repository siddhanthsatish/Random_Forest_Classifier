from cmath import nan
from dataclasses import replace
from gettext import find
from pyexpat import features
from random import randrange
from tokenize import String
import numpy as np
import pandas as pd #for data franes 
import matplotlib.pyplot as plt # for data visualization 
import warnings
from sklearn.utils import shuffle
import statistics
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
import math
import statistics
from statistics import mode

class Node():
    def __init__(self, feature=None, ch0=None, ch1=None, ch2= None, target=None, isLeaf=False):
        self.feature = feature
        self.ch0 = ch0
        self.ch1 = ch1
        self.ch2 = ch2
        self.target = target
        self.isLeaf = isLeaf

#converting into dataframe
def dataframe(data):
        return pd.read_csv(data)

#shuffling the datframe
def shuff(data):
        return shuffle(data)
        
#80-20 split on the data frame
def ttsplit(data):
        X, y = train_test_split(data, test_size=0.3, random_state=42) 
        return X,y

#calculating the entropy of an attribute 
def entropy(target_col):
    val ,counts = np.unique(target_col,return_counts = True)
    entropy = 0
    for i in range(len(val)):
        entropy += (-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts))
    return entropy

#calculating the information gain of an attribute
def infogain(data, split_name, target_name, attribute_type):
    total_entropy = entropy(data[target_name])
    vals,counts= np.unique(data[split_name],return_counts=True)
    average_entropy = 0
    # print(len(vals))
    
    if(attribute_type[split_name]=='categorical'):
        for i in range(len(vals)):
            attribute =  data.where(data[split_name]==vals[i]).dropna()[target_name]
            # print(attribute)
            average_entropy  += (counts[i]/ np.sum(counts))* entropy(attribute)
        return total_entropy - average_entropy
    else:
        if(len(vals)!=0):
            a1 = data.where(data[split_name]>=(sum(vals)/len(vals)))[target_name].dropna()
            a2 = data.where(data[split_name]<(sum(vals)/len(vals)))[target_name].dropna()
            average_entropy += entropy(a1)
            average_entropy += entropy(a2)
            # print(total_entropy - average_entropy)
            return total_entropy - average_entropy
        else:
            return total_entropy
    


#function to find if all the values in the array are unique
def unique(s):
    a = s.to_numpy() 
    return (a[0] == a).all()

def preprocessing(data, attributes):
    attribute_type = {}
    for attribute in attributes:
        values = np.unique(data[attribute])
        if(len(values)<5):
            attribute_type[attribute] = "categorical"
        else:
            attribute_type[attribute] = "numerical"
    return attribute_type

#decision tree creation
def decision_tree(data, attributes, label, attribute_type, parent=None):
    
    #create node
    node = Node()

    #check if all the data in the target column have the same value
    if(data['target'].empty == False and unique(data['target'])==True):
        node.target = data['target'].value_counts().idxmax()
        node.isLeaf = True
        return node
    #check if the features set is empty 
    elif(len(attributes)==0):
        node.isLeaf = True
        node.target = data['target'].value_counts().idxmax()
        return node

    #find the best attribute with helper functions and remove it from the attribute list
    infogains = []
    for attribute in attributes:
        infogains.append(infogain(data, attribute, label, attribute_type))
    best = attributes[infogains.index(max(infogains))]
    new_attributes = []
    for attribute in attributes:
        if(attribute!=best):
            new_attributes.append(attribute)
    

    
    #handling leaf nodes
    if(attribute_type[best]=='categorical'):
        value = np.unique(data[best])
        if data[label].empty == False:
            if len(value)<3:
                #if one of the nodes is empty
                node.isLeaf = True
                node.target = data['target'].value_counts().idxmax()
                return node
            else:
                #recrusive calls to form the decison rree
                node.feature = best
                node.ch0 = decision_tree(data[data[best] == 0], new_attributes, label, attribute_type)
                node.ch1 = decision_tree(data[data[best] == 1], new_attributes, label, attribute_type)
                node.ch2 = decision_tree(data[data[best] == 2], new_attributes, label, attribute_type)
                return node
    elif(attribute_type[best]=='numerical'):
        vals,counts= np.unique(data[best],return_counts=True)
        if(len(vals)!=0):
            if(data[data[best]>=(sum(vals)/len(vals))].empty or data[data[best]<(sum(vals)/len(vals))].empty):
                node.isLeaf = True
                return node
            else:
                # if(data.shape[0]<10):
                #     node.isLeaf = True
                #     node.target = data['target'].value_counts().idxmax()
                #     return node
                # else:
                node.feature = [best, (sum(vals)/len(vals))]
                node.ch0 = decision_tree(data[data[best]>=(sum(vals)/len(vals))], new_attributes, label, attribute_type)
                node.ch1 = decision_tree(data[data[best]<(sum(vals)/len(vals))], new_attributes, label, attribute_type)
        else:
            node.isLeaf = True
            return node
        return node #??



#predicting for a particular row
def predict(test, node, attribute_type):
    
    if(node.isLeaf==True):
        return node.target
    else:
        # print(isinstance(node.feature, str))
        if(isinstance(node.feature, str)==False):
            branch = test[node.feature[0]]
            if branch >= node.feature[1]:
                return predict(test, node.ch0, attribute_type)
            elif branch < node.feature[1]:
                return predict(test, node.ch1, attribute_type)
        else:
            branch = test[node.feature]
            if branch == 0:
                return predict(test, node.ch0, attribute_type)
            elif branch == 1:
                return predict(test, node.ch1, attribute_type)
            elif branch == 2:
                return predict(test, node.ch2, attribute_type)



def create_k_folds(data, k):
    dataset_split = []
    df_copy = data
    fold_size = int(df_copy.shape[0] / k)
    
    # for loop to save each fold
    for i in range(k):
        fold = []
        # while loop to add elements to the folds
        while len(fold) < fold_size:
            # select a random element
            r = randrange(df_copy.shape[0]) # random or no ????
            # determine the index of this element 
            index = df_copy.index[r]
            # save the randomly selected line 
            fold.append(df_copy.loc[index].values.tolist())
            # delete the randomly selected line from
            # dataframe not to select again
            df_copy = df_copy.drop(index)
        # save the fold     
        test = pd.DataFrame(fold, columns = df_copy.columns.values.tolist())
        train = df_copy
        dataset_split.append([train, test])
    return dataset_split 

def get_random_samples(X):
    return X.sample(frac=1, replace=True, random_state=1)

def get_random_features(X):
    m = X.columns[:-1]
    # print(len(m))
    size = int(math.sqrt(len(m)))
    # print(type(m))
    m = m.to_numpy()
    res = np.random.choice(m, size= size, replace=False)
    return pd.Series(res) 

def most_common(List):
    return(mode(List))

def evaluation_metrics(original, pred):
    count = 0
    # print(len(original)==len(pred))
    for i in range(len(original)):
        if(original[i]== pred[i]):
            count+=1
    accuracy = count/len(original)
    # return count/len(y)
    # matrix=np.zeros((2,2)) # form an empty matric of 2x2
    # for i in range(len(pred)): #the confusion matrix is for 2 classes: 1,0
    #     #1=positive, 0=negative
    #     if int(pred[i])==1 and int(original[i])==1: 
    #         matrix[0,0]+=1 #True Positives
    #     elif int(pred[i])==1 and int(original[i])==0:
    #         matrix[0,1]+=1 #False Positives
    #     elif int(pred[i])==0 and int(original[i])==1:
    #         matrix[1,0]+=1 #False Negatives
    #     elif int(pred[i])==0 and int(original[i])==0:
    #         matrix[1,1]+=1 #True Negatives
    # accuracy = (matrix[0,0] + matrix[1,1])/(matrix[0,0] + matrix[0,1] + matrix[1,0] + matrix[1,1])
    # # print("Accuracy:",accuracy)
    # precision=matrix[0,0]/(matrix[0,0]+matrix[0,1])
    # # print("Precision:",precision)
    # recall=matrix[0,0]/(matrix[0,0]+matrix[1,0])
    # # print("Recall:",recall)
    # # print("Specificity:",specificity)
    # # negative_pred_value=matrix[1,1]/(matrix[1,0]+matrix[1,1])
    # # print("Negative Predicted Value:",negative_pred_value)
    # f1= 2*(precision*recall)/(precision+recall)
    # print("F1 score:",f1)

        # extract the different classes
    classes = np.unique(original)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):
           # count the number of instances in each combination of actual / predicted classes
           confmat[i, j] = np.sum((pred == classes[i]) & (original== classes[j]))

    # k = np.array([[2,1,0], [3,4,5], [6,7,8]])
    # # for arr in confmat:
    # #     k.append(list(arr))
    # k = np.array(k)
    # print(k)
    # true_pos = np.diag(confmat).sum()
    # false_pos = np.sum(confmat, axis=0).sum() - true_pos
    # false_neg = np.sum(confmat, axis=1).sum() - true_pos
    # true_neg = np.sum(confmat) - (true_pos+false_pos+false_neg)
    # print(confmat)
    # accuracy = (true_pos + true_neg) / (np.sum(confmat))
    # precision = true_pos / np.sum(confmat, axis=0).sum()
    # recall = true_pos / np.sum(confmat, axis=1).sum()
    
    
    precision = 0
    f1_score = 0
    recall = 0
    cm = confmat
    true_pos = np.diag(cm)
    false_pos = np.sum(cm, axis=0) - true_pos
    false_neg = np.sum(cm, axis=1) - true_pos
    for i in range(len(classes)):
        if true_pos[i] != 0:
            precision += true_pos[i] / (true_pos[i] + false_pos[i])
    precision = precision / len(classes)
   
    for i in range(len(classes)):
        if true_pos[i] != 0:
            recall += true_pos[i] / (true_pos[i] + false_neg[i])
    recall = recall / len(classes)
    if precision == 0 and recall == 0:
        f1_score = 0
    else:
        f1_score = 2*(precision*recall)/(precision + recall)

    return accuracy, precision, recall, f1_score

def majority_voting(d):
    final_predictions = []
    for i in d:
        final_predictions.append(most_common(d[i]))
    return final_predictions

#testing for all the rows in the test set
def forest_predict(data,forest, attribute_type):
    #convert it to a dictionary
    predictions = []
    final_predictions = []
    actual_values = []
    count = 0
    accuracies = [] 
    precisions = [] 
    recalls = []
    f1_scores = []
    ntree = 0
    d = {}
    actual_values = []
    # print(actual_values)
    for tree in forest:
        ntree += 1
        actual_values = []
        for i in range(len(data)):
            queries = data.iloc[:,:-1].to_dict(orient = "records")
            targets = data.iloc[:,-1:].to_dict(orient = "records")
            #create a empty DataFrame to store the predictions
            predicted = pd.DataFrame(columns=["predicted"]) 
            # #calculate the prediction accuracy by comparing prediction with the target values
            predictions.append(predict(queries[i],tree, attribute_type))
            value = most_common(predictions)
            if i not in d.keys():
                d[i] = [value]
            else:
                d[i].append(value)
            actual_values.append(targets[i]['target'])
        final_predictions = majority_voting(d)
        accuracy, precision, recall, f1_score = evaluation_metrics(actual_values, final_predictions)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    # print(count)
    # print(count/len(data))
    # return count/len(data)
    return accuracies, precisions, recalls, f1_scores



    

#plotting function for training data
def training_plot(data):    
    k_accuracies = []
    k_precisions = []
    k_recalls = []
    k_f1_scores = []
    label = 'target'
    attributes = data.columns[:-1]
    k = 10
    kfolds = create_k_folds(data, k)
    prediction_list = []
    ntrees = 2
    ntree_arr = list(np.arange(start=1, stop= ntrees+1))
    for kfold in kfolds:
        X = kfold[0]
        y = kfold[1]
        forest = []

        for i in range(1,ntrees+1):
    
            new_attributes = get_random_features(X)
            # print(type(new_attributes))
            attributes = new_attributes
            # print(attributes)
            attribute_type = preprocessing(X, attributes)
            # print(attribute_type)
            X = get_random_samples(X)
            # print(X.empty)
            # print(y.empty)
            if X.empty or y.empty:
                continue
            tree = decision_tree(X, attributes, label, attribute_type)
            forest.append(tree)
        accuracies, precisions, recalls, f1_scores = forest_predict(y,forest, attribute_type)
        k_accuracies.append(accuracies)
        k_precisions.append(precisions)
        k_recalls.append(recalls)
        k_f1_scores.append(f1_scores)
    k_accuracies = np.array(k_accuracies)
    mean1 = list(k_accuracies.mean(axis = 0))
    print(mean1)
    k_precisions = np.array(k_precisions)
    mean2 = list(k_precisions.mean(axis = 0))
    print(mean2)
    k_recalls = np.array(k_recalls)
    mean3 = list(k_recalls.mean(axis = 0))
    print(mean3)
    print()
    # plt.plot(mean)
    print(k_accuracies)
    print()
    print(k_precisions)
    print()
    print(k_recalls)
    print()
    print(k_f1_scores)
    # print(accuracies)
    # print("The average training accuracy is: ", sum(accuracies) / len(accuracies))
    # print("The standard deviation of the training accuracy is: ", statistics.pstdev(accuracies))
    # # Creating histogram
    # fig, ax = plt.subplots(figsize =(10, 7))
    # ax.hist(accuracies)

    # Show plot
    # plt.show()

#plotting function for testing data
def testing_plot(k):    
    accuracies = []
    # attributes = k.columns[:-1]
    attributes = k.columns[:-1]
    label = 'target'
    for i in range(10):
        X, y = ttsplit(shuff(k))
        tree = decision_tree(X, attributes, label)
        # accuracies.append(test(y, tree))

    print("The average of the testing accuracy is: ", sum(accuracies) / len(accuracies))
    print("The standard deviation of the testing accuracy is: ", statistics.pstdev(accuracies))
    # Creating histogram
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(accuracies)

    # Show plot
    plt.show()

#driver code for calling the functions
data = 'datasets/votes.csv'
df = dataframe(data)
training_plot(df)
# testing_plot(df)