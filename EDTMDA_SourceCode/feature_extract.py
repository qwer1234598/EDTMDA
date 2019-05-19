#!user/bin/python
#coding=utf-8
from sklearn.decomposition import NMF
import math
#from math import e
#from nmf import *
import numpy as np
import time
import networkx as nx



  ##############################
  ## Type 1 feature of miRNAs ##
  ##############################


def threetypes_features(nm,nd,A,FS_integration,DS_integration):
    noOfObervationsOfmiRNA=np.zeros((nm,1)) # number of observations in each row of MDA
    aveOfSimilaritiesOfmiRNA=np.zeros((nm,1))# average of all similarity scores for each miRNA
    # histogram feature: cut [0, 1] into five bins and count the proportion of similarity scores that fall into each bin
    hist1miRNA=np.zeros((nm,1))
    hist2miRNA=np.zeros((nm,1))
    hist3miRNA=np.zeros((nm,1))
    hist4miRNA=np.zeros((nm,1))
    hist5miRNA=np.zeros((nm,1))

    for i in range(nm):
        noOfObervationsOfmiRNA[i,0]=np.sum(A[i, ])
        aveOfSimilaritiesOfmiRNA[i,0]=np.mean(FS_integration[i, ])
        #print (aveOfSimilaritiesOfmiRNA[i,0])
        hist1Count = 0.0
        hist2Count = 0.0
        hist3Count = 0.0
        hist4Count = 0.0
        hist5Count = 0.0
        for j in range(nm):
            if(FS_integration[i, j] < 0.2):
                hist1Count = hist1Count + 1.0
            elif(FS_integration[i, j] < 0.4):
                hist2Count = hist2Count + 1.0
            elif(FS_integration[i, j] < 0.6):
                hist3Count = hist3Count + 1.0
            elif(FS_integration[i, j] < 0.8):
                hist4Count = hist4Count + 1.0
            elif(FS_integration[i, j] <= 1):
                hist5Count = hist5Count + 1.0
            
            
        hist1miRNA[i,0]=hist1Count /nm
        hist2miRNA[i,0]=hist2Count /nm
        hist3miRNA[i,0]=hist3Count /nm
        hist4miRNA[i,0]=hist4Count /nm
        hist5miRNA[i,0]=hist5Count /nm
                
   
    #print (hist1miRNA,hist2miRNA,hist3miRNA,hist4miRNA,hist5miRNA)
    feature1OfmiRNA=np.hstack((noOfObervationsOfmiRNA, aveOfSimilaritiesOfmiRNA, hist1miRNA,hist2miRNA, hist3miRNA, hist4miRNA, hist5miRNA))
    #print ('feature1OfmiRNA',feature1OfmiRNA[0])
      ################################
      ## Type 1 feature of diseases ##
      ################################


    noOfObervationsOfdisease=np.zeros((nd,1))# number of observations in each column of MDA
    aveOfsimilaritiesOfDisease=np.zeros((nd,1))# average of all similarity scores for each disease
    hist1disease=np.zeros((nd,1))# histogram feature: cut [0, 1] into five bins and count the proportion of similarity scores that fall into each bin
    hist2disease=np.zeros((nd,1))
    hist3disease=np.zeros((nd,1))
    hist4disease=np.zeros((nd,1))
    hist5disease=np.zeros((nd,1))
    for i in range(nd):
        noOfObervationsOfdisease[i,0]=np.sum(A[:, i])
        aveOfsimilaritiesOfDisease[i]=np.mean(DS_integration[i])
        hist1Count = 0.0
        hist2Count = 0.0
        hist3Count = 0.0
        hist4Count = 0.0
        hist5Count = 0.0
        for j in range(nd):
            if(DS_integration[i, j] < 0.2):
                hist1Count = hist1Count + 1.0
            elif(DS_integration[i, j] < 0.4):
                hist2Count = hist2Count + 1.0
            elif(DS_integration[i, j] < 0.6):
                hist3Count = hist3Count + 1.0
            elif(DS_integration[i, j] < 0.8):
                hist4Count = hist4Count + 1.0
            elif(DS_integration[i, j] <= 1):
                hist5Count = hist5Count + 1.0

        hist1disease[i,0]=hist1Count /nd
        hist2disease[i,0]=hist2Count /nd
        hist3disease[i,0]=hist3Count /nd
        hist4disease[i,0]=hist4Count /nd
        hist5disease[i,0]=hist5Count /nd

    feature1OfDisease=np.hstack((noOfObervationsOfdisease, aveOfsimilaritiesOfDisease, hist1disease,hist2disease, hist3disease, hist4disease, hist5disease))
    #print ('feature1OfDisease',feature1OfDisease[0])
      #############################
      # Type 2 feature of miRNAs ##
      #############################

    #number of neighbors of miRNAs and similarity values for 10 nearest neighbors
    numberOfNeighborsMiRNA=np.zeros((nm,1))
    similarities10KnnMiRNA=np.zeros((nm,10))
    averageOfFeature1MiRNA=np.zeros((nm,7))
    weightedAverageOfFeature1MiRNA=np.zeros((nm,7))
    similarityGraphMiRNA=np.zeros((nm,nm))
    meanSimilarityMiRNA=np.mean(FS_integration)
    for i in range(nm):
        neighborCount = 0 - 1 # similarity between an miRNA and itself is not counted
        for j in range(nm):
            if(FS_integration[i, j] >= meanSimilarityMiRNA):
                neighborCount = neighborCount + 1
                similarityGraphMiRNA[i, j] = 1
        numberOfNeighborsMiRNA[i,0]=neighborCount

        similarities10KnnMiRNA[i, ]=sorted(FS_integration[i, ], reverse= True )[1:11]
        indices=np.argsort(-FS_integration[i, ])[1:11]

        averageOfFeature1MiRNA[i, ]=np.mean(feature1OfmiRNA[indices, ],0)
        weightedAverageOfFeature1MiRNA[i, ]=np.dot(similarities10KnnMiRNA[i, ],feature1OfmiRNA[indices, ])/10
        # build miRNA similarity graph
    mSGraph = nx.from_numpy_matrix(similarityGraphMiRNA)
    betweennessCentralityMiRNA=np.array(list(nx.betweenness_centrality(mSGraph).values())).T
    #print ("numberOfNeighborsMiRNA",numberOfNeighborsMiRNA[0,0],'similarities10KnnMiRNA',similarities10KnnMiRNA[0])#betweennessCentralityMiRNA.shape
    #print (betweennessCentralityMiRNA)
    #print (np.array(betweennessCentralityMiRNA.values()))
    #closeness_centrality
    closenessCentralityMiRNA=np.array(list(nx.closeness_centrality(mSGraph).values())).T
    #print (closenessCentralityMiRNA.shape)
    #pagerank
    pageRankMiRNA=np.array(list(nx.pagerank(mSGraph).values())).T
    #print (pageRankMiRNA.shape)
    #eigenvector_centrality
    # eigenvector_centrality=nx.eigenvector_centrality(mSGraph)
    eigenVectorCentralityMiRNA=np.array(list(nx.eigenvector_centrality(mSGraph).values())).T
    #print (eigenVectorCentralityMiRNA.shape)
    combination=np.array([betweennessCentralityMiRNA,closenessCentralityMiRNA,pageRankMiRNA,eigenVectorCentralityMiRNA])
    #print (combination)
    #print (combination.shape)
      # # concatenation
    feature2OfmiRNA=np.hstack((numberOfNeighborsMiRNA, similarities10KnnMiRNA, averageOfFeature1MiRNA, weightedAverageOfFeature1MiRNA,combination.T))#betweennessCentralityMiRNA, closenessCentralityMiRNA, eigenVectorCentralityMiRNA, pageRankMiRNA))
    #print ('feature2OfmiRNA',feature2OfmiRNA[0])
      ###############################
      # Type 2 feature of diseases ##
      ###############################

      # number of neighbors of diseases and similarity values for 10 nearest neighbors
    numberOfNeighborsDisease=np.zeros((nd,1))
    similarities10KnnDisease=np.zeros((nd,10))
    averageOfFeature1Disease=np.zeros((nd,7))
    weightedAverageOfFeature1Disease=np.zeros((nd,7))
    similarityGraphDisease=np.zeros((nd,nd))
    meanSimilarityDisease=np.mean(DS_integration)
    for i in range(nd):
        neighborCount = 0 - 1 
        for j in range(nd):
            if(DS_integration[i, j] >= meanSimilarityDisease):
                neighborCount = neighborCount + 1
                similarityGraphDisease[i, j] = 1

        numberOfNeighborsDisease[i,0]=neighborCount

        similarities10KnnDisease[i, ]=sorted(DS_integration[i, ], reverse= True)[1:11]
        indices=np.argsort(-DS_integration[i, ])[1:11]


        averageOfFeature1Disease[i, ]=np.mean(feature1OfDisease[indices, ],0)
        weightedAverageOfFeature1Disease[i, ]=np.dot(similarities10KnnDisease[i, ],feature1OfDisease[indices, ])/10

    # build disease similarity graph
    dSGraph = nx.from_numpy_matrix(similarityGraphDisease)
    betweennessCentralityDisease=np.array(list(nx.betweenness_centrality(dSGraph).values())).T
    #print (betweenness_centrality)
    #closeness_centrality
    closenessCentralityDisease=np.array(list(nx.closeness_centrality(dSGraph).values())).T
    #print (closeness_centrality)
    #pagerank
    pageRankDisease=np.array(list(nx.pagerank(dSGraph).values())).T
    #print (pagerank)
    #eigenvector_centrality
    eigenVectorCentralityDisease=np.array(list(nx.eigenvector_centrality(dSGraph).values())).T
    #print (eigenvector_centrality)
    combination=np.array([betweennessCentralityDisease,closenessCentralityDisease,pageRankDisease,eigenVectorCentralityDisease])
    #print (combination)
    #print (combination.shape)

      # concatenation
    feature2OfDisease=np.hstack((numberOfNeighborsDisease, similarities10KnnDisease, averageOfFeature1Disease, weightedAverageOfFeature1Disease,combination.T))#betweennessCentralityDisease, closenessCentralityDisease, eigenVectorCentralityDisease, pageRankDisease))
    #print ('feature2OfDisease',feature2OfDisease[0])
      ###########################################
      ## Type 3 feature of miRNA-disease pairs ##
      ###########################################

      # matrix factorization
    # number of associations between an miRNA and a disease's neighbors
    nmf_model = NMF(n_components=20)
    latentVectorsMiRNA = nmf_model.fit_transform(A)
    latentVectorsDisease = nmf_model.components_
    numberOfDiseaseNeighborAssociations=np.zeros((nm,nd))
    numberOfMiRNANeighborAssociations=np.zeros((nm,nd))
    MDAGraph=nx.Graph() 
    MDAGraph.add_nodes_from(list(range(nm+nd)))
    for i in range(nm):
        for j in range(nd):
            if A[i,j]==1:
                MDAGraph.add_edge(i, j+495)# build MDA graph
            for k in range(nd):
                if DS_integration[j,k]>= meanSimilarityDisease and A[i,k]==1 :
                    numberOfDiseaseNeighborAssociations[i,j]= numberOfDiseaseNeighborAssociations[i,j] + 1
                    
            for l in range (nm):
                if FS_integration[i,l]>= meanSimilarityMiRNA and A[l,j]==1 :
                    numberOfMiRNANeighborAssociations[i,j]= numberOfMiRNANeighborAssociations[i,j] + 1

    #betweennessCentralityMDA=nx.betweenness_centrality(MDAGraph)
    betweennessCentralityMDA=np.array(list(nx.betweenness_centrality(MDAGraph).values())).T
    betweennessCentralityMiRNAInMDA=betweennessCentralityMDA[0:495]
    betweennessCentralityDiseaseInMDA=betweennessCentralityMDA[495:878]
    #print (betweenness_centrality)
    closenessCentralityMDA=np.array(list(nx.closeness_centrality(MDAGraph).values())).T
    closenessCentralityMiRNAInMDA=closenessCentralityMDA[0:495]
    closenessCentralityDiseaseInMDA=closenessCentralityMDA[495:878]
    eigenVectorCentralityMDA=np.array(list(nx.eigenvector_centrality_numpy(MDAGraph).values())).T#nx.eigenvector_centrality(MDAGraph)
    eigenVectorCentralityMiRNAInMDA=eigenVectorCentralityMDA[0:495]
    eigenVectorCentralityDiseaseInMDA=eigenVectorCentralityMDA[495:878]
    pageRankMDA=np.array(list(nx.pagerank(MDAGraph).values())).T#nx.pagerank(MDAGraph)
    pageRankMiRNAInMDA=pageRankMDA[0:495]
    pageRankDiseaseInMDA=pageRankMDA[495:878]

    Diseasecombination=np.array([betweennessCentralityDiseaseInMDA,closenessCentralityDiseaseInMDA,eigenVectorCentralityDiseaseInMDA,pageRankDiseaseInMDA])
    feature3OfDisease=np.hstack((latentVectorsDisease.T,Diseasecombination.T))
    MiRNAcombination=np.array([betweennessCentralityMiRNAInMDA,closenessCentralityMiRNAInMDA,eigenVectorCentralityMiRNAInMDA,pageRankMiRNAInMDA])
    feature3OfmiRNA=np.hstack((latentVectorsMiRNA,MiRNAcombination.T))
    #print ('feature3OfmiRNA',feature3OfmiRNA[0])
    #print ('feature3OfDisease',feature3OfDisease[0])
    Feature_miRNA = np.hstack ((feature1OfmiRNA,feature2OfmiRNA,feature3OfmiRNA))
    Feature_disease = np.hstack((feature1OfDisease,feature2OfDisease,feature3OfDisease))
    return Feature_miRNA,Feature_disease,numberOfDiseaseNeighborAssociations,numberOfMiRNANeighborAssociations





