from copy import deepcopy
import math
from math import factorial
from time import time
import random
import sys

start=time()
def load_dataset(f_name):
    f_string=""
    with open(f_name,'r') as f:
        f_string=f_string+f.read()
    f_string=f_string.lower()
    f_string=f_string.replace('\W',' ')
    file_list=list(f_string.split('\n'))
    dataset=[]
    for var in file_list:
        dataset.append(var.split('\t'))
    return dataset

def multinomial(data,k):
    folds=[]
    accuracy=0
    for r in range(k):
        folds.append(data[(len(data)*r//k):(min(len(data),len(data)*(r+1)//k))])
    for fold in range(len(folds)):
        train_d=[]
        for item in range(len(folds)):
            if item!=fold:
                train_d.extend(folds[item])
        
        test_data=[i for i in folds[fold]]

        N=len(train_d)
        #print(N)
        Class_list=list(set([str(i[0]).lower() for i in train_d]))
        #print(Class_list)
        Class={}
        for i in range(len(Class_list)):
            Class[Class_list[i]]=i
        #print(Class)
        prior={i:0 for i in Class.keys()}
        #print(prior)
        for i in train_d:
            prior[str(i[0]).lower()]=prior[str(i[0]).lower()]+1
        #print(prior)
        V={}
        mega_document=['' for i in prior.keys()]
        #print(mega_document)
        for i in train_d:
            mega_document[Class[i[0]]]+=i[1]
        #print(mega_document)
        #counting occurence of word in spam and ham 
        vocab_set=0
        for i in train_d:
            vocab=i[1].split(' ')
            vocab_set=vocab_set+len(list(set(vocab)))
            for j in list(set(vocab)):
                if j not in V.keys():
                    V[j.lower()]=[0 for i in range(len(Class))]
                if V[j.lower()][Class[i[0]]]==0:
                    V[j.lower()][Class[i[0]]]=mega_document[Class[i[0]]].count(j.lower())
        ############################################################################
        ###calculating probabilities for each word
        for i in V.keys():
            for j in prior.keys():
                V[i][Class[j]]=(V[i][Class[j]]+1)/(prior[j]+vocab_set)
        #print(V)
        for i in prior.keys():
            prior[i]=prior[i]/N
    
        #print(prior)
        pred=[]
        for value in range(len(test_data)):
            test=list(test_data[value][1].split(' '))
            test=[i.lower() for i in test]
            score={}
            maxm=-100000
            res_class=''
            for c in Class.keys():
                score[c]=math.log2(prior[c])/math.log2(math.exp(1))
                for t in V.keys():
                    if t in test:
                        score[c]=score[c]+(math.log2(V[t][Class[c]])/math.log2(math.exp(1)))
                if score[c]>maxm:
                    maxm=score[c]
                    res_class=c
            pred.append([test_data,test_data[value][0],res_class])

        correct_class=[1 for i in pred if i[1]==i[2]]
        print('accuracy for fold ',fold+1,': ',round(len(correct_class)*100/len(test_data),3))
        accuracy=accuracy+len(correct_class)/len(test_data)
        break
    return accuracy/k

if __name__ == '__main__':
    data=load_dataset("C:/Users/com/Desktop/smsspamcollection/SMSSpamCollection")	
    print('length of dataset: ', len(data))
    k=5
    #accuracy_MV=multivariateNB(deepcopy(data),k)
    #print('\nFinal Accuracy of multivariate:', round(accuracy_MV*100,3),'\n')
    accuracy=multinomial(deepcopy(data),k)
    print('\naverage Accuracy of multinomial model:', round(accuracy*100,3),'\n')
    diff_time=time()-t0
    print('time taken for execution :', diff_time)
    print('\n')