import numpy as np
from sklearn import tree
from feature_extract import *
from sklearn.decomposition import PCA
import random
import pandas as pd
import time
import math

pca=PCA(n_components=10)

nm = 495 #number of miRNAs
nd = 383 # number of diseases
nc = 5430 # number of miRNA-disease associations 
r=0.5 # Decising the size of feature subset
nn = 495*383-5430 # number of unknown samples
M = 50 # number of decison trees 

miRNAnumbercode = np.loadtxt(r'.\data\miRNA number ID.txt',dtype=bytes).astype(str)
diseasenumbercode = np.genfromtxt(r'.\data\disease number ID.txt',dtype=str,delimiter='\t')
DS1 = np.loadtxt(r'.\data\disease semantic similarity matrix 1.txt') 
DS2 = np.loadtxt(r'.\data\disease semantic similarity matrix 2.txt')
DS = (DS1 + DS2) / 2  
DSweight = np.loadtxt(r'.\data\disease semantic similarity weight matrix.txt') 

FS = np.loadtxt(r'.\data\miRNA functional similarity matrix.txt')  
FSweight = np.loadtxt(r'.\data\miRNA functional similarity weight matrix.txt') 


def Getgauss_miRNA(adjacentmatrix,nm):
       """
       MiRNA Gaussian interaction profile kernels similarity
       """
       KM = np.zeros((nm,nm))

       gamaa=1
       sumnormm=0
       for i in range(nm):
           normm = np.linalg.norm(adjacentmatrix[i])**2
           sumnormm = sumnormm + normm  
       gamam = gamaa/(sumnormm/nm)


       for i in range(nm):
              for j in range(nm):
                      KM[i,j]= math.exp (-gamam*(np.linalg.norm(adjacentmatrix[i]-adjacentmatrix[j])**2))
       return KM
       
def Getgauss_disease(adjacentmatrix,nd):
       """
       Disease Gaussian interaction profile kernels similarity
       """
       KD = np.zeros((nd,nd))
       gamaa=1
       sumnormd=0
       for i in range(nd):
              normd = np.linalg.norm(adjacentmatrix[:,i])**2
              sumnormd = sumnormd + normd
       gamad=gamaa/(sumnormd/nd)

       for i in range(nd):
           for j in range(nd):
               KD[i,j]= math.exp(-(gamad*(np.linalg.norm(adjacentmatrix[:,i]-adjacentmatrix[:,j])**2)))
       return KD

# adjacency matrix
A = np.zeros((nm,nd),dtype=float)
ConnectDate = np.loadtxt(r'.\data\known disease-miRNA association number ID.txt',dtype=int)-1 
for i in range(nc):
    A[ConnectDate[i,0], ConnectDate[i,1]] = 1 # the element is 1 if the miRNA-disease pair has association
    
dataset_n = np.argwhere(A == 0)
Trainset_p = np.argwhere(A == 1)
       
KM = Getgauss_miRNA(A,nm)  
KD = Getgauss_disease(A,nd)  
    
#integrating miRNA functional similarity and Gaussian interaction profile kernels similarity   
FS_integration=np.zeros((nm,nm))
for i in range(nm):
    for j in range(nm):
        if  FSweight[i,j] == 1:
            FS_integration[i,j] = FS[i,j]
        else:
            FS_integration[i,j] = KM[i,j]
  
#integrating disease semantic similarity and Gaussian interaction profile kernels similarity 
DS_integration=np.zeros((nd,nd))
for i in range(nd):
    for j in range(nd):
        if  DSweight[i,j] == 1:
            DS_integration[i,j] = DS[i,j]
        else:
            DS_integration[i,j] = KD[i,j]

MirnaFeature,DiseaseFeature,numberOfDiseaseNeighborAssociations,\
numberOfMiRNANeighborAssociations = threetypes_features(nm,nd,A,FS_integration,DS_integration)
            
predict_0 =np.zeros((dataset_n.shape[0]))
for i_M in range(M):
    Trainset_n = dataset_n[random.sample(list(range(nn)),nc)]
    
    # print (Trainset_n)
    Trainset= np.vstack((Trainset_n,Trainset_p))   
    
    
    TrainMirnaFeature = MirnaFeature[Trainset[:,0]]
    TrainDiseaseFeature = DiseaseFeature[Trainset[:,1]]

    MirnaNumberNeighborTrain = numberOfMiRNANeighborAssociations[Trainset[:,0],Trainset[:,1]]
    DiseaseNumberNeighborTrain = numberOfDiseaseNeighborAssociations[Trainset[:,0],Trainset[:,1]]
    
    TrainMirnaFeatureOfPair = np.hstack((TrainMirnaFeature, DiseaseNumberNeighborTrain.reshape(DiseaseNumberNeighborTrain.shape[0],1)))
    randomNum_mirnaFeature=random.sample(list(range(TrainMirnaFeatureOfPair.shape[1])),int(r*TrainMirnaFeatureOfPair.shape[1]))
    TrainMirnaFeatureOfPair_random=TrainMirnaFeatureOfPair[:,randomNum_mirnaFeature]    
    PCA_TrainMirnaFeatureOfPair = pca.fit_transform(TrainMirnaFeatureOfPair_random)
    PCA_miRNATrainVarianceRatio = pca.explained_variance_ratio_
    TrainDiseaseFeatureOfPair = np.hstack((TrainDiseaseFeature, MirnaNumberNeighborTrain.reshape(MirnaNumberNeighborTrain.shape[0],1)))
    randomNum_diseaseFeature=random.sample(list(range(TrainDiseaseFeatureOfPair.shape[1])),int(r*TrainDiseaseFeatureOfPair.shape[1]))
    TrainDiseaseFeatureOfPair_random=TrainDiseaseFeatureOfPair[:,randomNum_diseaseFeature] 
    PCA_TrainDiseaseFeatureOfPair = pca.transform(TrainDiseaseFeatureOfPair_random)
    PCA_diseaseTrainVarianceRatio = pca.explained_variance_ratio_
   
    X_train = np.hstack((PCA_TrainMirnaFeatureOfPair,PCA_TrainDiseaseFeatureOfPair))
    
    Y_value=[]
    for i in range(Trainset_n.shape[0]):
        Y_value.append(0.0)
    for i in range(Trainset_n.shape[0],Trainset.shape[0]):
        Y_value.append(1.0)
   
    clf = tree.DecisionTreeRegressor(splitter='random',min_samples_split=3,min_samples_leaf = 2)
    clf = clf.fit(X_train, Y_value)
   
    TestMirnaFeature = MirnaFeature[dataset_n[:,0]]
    TestDiseaseFeature = DiseaseFeature[dataset_n[:,1]]  
   
    MirnaNumberNeighborTest = numberOfMiRNANeighborAssociations[dataset_n[:,0],dataset_n[:,1]]
    DiseaseNumberNeighborTest = numberOfDiseaseNeighborAssociations[dataset_n[:,0],dataset_n[:,1]]
    
    TestMirnaFeatureOfPair = np.hstack((TestMirnaFeature, DiseaseNumberNeighborTest.reshape(DiseaseNumberNeighborTest.shape[0],1)))
    TestMirnaFeatureOfPair_random=TestMirnaFeatureOfPair[:,randomNum_mirnaFeature]
    PCA_TestMirnaFeatureOfPair=pca.transform(TestMirnaFeatureOfPair_random)
    PCA_miRNATestVarianceRatio = pca.explained_variance_ratio_
    TestDiseaseFeatureOfPair = np.hstack((TestDiseaseFeature, MirnaNumberNeighborTest.reshape(MirnaNumberNeighborTest.shape[0],1)))
    TestDiseaseFeatureOfPair_random=TestDiseaseFeatureOfPair[:,randomNum_diseaseFeature]
    PCA_TestDiseaseFeatureOfPair=pca.transform(TestDiseaseFeatureOfPair_random)
    PCA_diseaseTestVarianceRatio = pca.explained_variance_ratio_

    #X_test = np.hstack((FS_test,DS_test,MiRNADiseaseFeatureTest))
    X_test = np.hstack((PCA_TestMirnaFeatureOfPair,PCA_TestDiseaseFeatureOfPair))

   
    predict_0 = predict_0 + clf.predict(X_test)  # prediction results of all unknown samples

    

predict_0 = predict_0/M
   
predict_0scoreranknumber =np.argsort(-predict_0)
predict_0scorerank = predict_0[predict_0scoreranknumber]
diseaserankname_pos = dataset_n[predict_0scoreranknumber,1]
diseaserankname = diseasenumbercode[diseaserankname_pos,1]
miRNArankname_pos = dataset_n[predict_0scoreranknumber,0]
miRNArankname = miRNAnumbercode[miRNArankname_pos,1]
predict_0scorerank_pd=pd.Series(predict_0scorerank)
diseaserankname_pd=pd.Series(diseaserankname)
miRNArankname_pd=pd.Series(miRNArankname)
prediction_0_out = pd.concat([diseaserankname_pd,miRNArankname_pd,predict_0scorerank_pd],axis=1)
prediction_0_out.columns=['Disease','miRNA','Score']
prediction_0_out.to_excel(r'prediction results for all unknown samples.xlsx', sheet_name='Sheet1',index=False)
    






     
        

    
        
    
    
    
    


