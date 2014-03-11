#!/usr/bin/env python

import numpy as np
import sklearn 
import argparse

parser = argparse.ArgumentParser(description="Probabilistic Matrix Factorization!")
parser.add_argument('trainFileName',type=str,help="Input Train filename")
parser.add_argument('testFileName', type=str, help="Input Test filename")
args = parser.parse_args()


trainfilename = args.trainFileName
testfilename = args.testFileName

traindata = np.memmap(trainfilename, dtype='int', mode='r+')
traindata.shape = (len(traindata)/3,3)

num_m = max(traindata[0:len(traindata),0])+1  #Number of movies
num_p = max(traindata[0:len(traindata),1])+1   #Number of users

lentrain = len(traindata) * 0.9
testdata = traindata[lentrain:]
lentest = len(testdata)
traindata = traindata[0:lentrain]

#testdata = np.memmap(testfilename,dtype='int', mode='r+')
#testdata.shape = (len(testdata)/2,2)

mean_rating = np.mean(traindata[0:lentrain,2])

epsilon = 50
lambda_t = 0.01
momentum = 0.8

maxepoch=50

numbatches = 9 #Number of batches

num_feat = 10 #Rank 10 decomposition

w1_M1 = 0.1 * np.random.randn(num_m, num_feat)
w1_P1 = 0.1 * np.random.randn(num_p, num_feat)
w1_M1_inc = np.zeros((num_m, num_feat))
w1_P1_inc = np.zeros((num_p, num_feat))
err_train = np.zeros((maxepoch,1))
err_valid = np.zeros((maxepoch,1))

for epoch in range(maxepoch):
    traindata = np.random.permutation(traindata)
    
    for batch in range(numbatches):
         print( " epoch %d batch %d " % (epoch, batch))
         N = 100000 
         
         aa_m = traindata[batch*N:(batch+1)*N,0]
         aa_p = traindata[batch*N:(batch+1)*N,1]
         rating = traindata[batch*N:(batch+1)*N,2]
         rating = rating - mean_rating
         
         #Compute Predictions
         pred_out = np.sum(np.multiply(w1_M1[np.ix_(aa_m)], w1_P1[np.ix_(aa_p)]), axis = 1)
         
         f = np.sum(np.multiply(pred_out - rating, pred_out - rating) 
                     + 0.5 * lambda_t 
                     * np.sum(np.multiply(w1_M1[np.ix_(aa_m)], w1_M1[np.ix_(aa_m)]) + np.multiply(w1_P1[np.ix_(aa_p)], w1_P1[np.ix_(aa_p)]), axis = 1))
         
         #Compute Gradients
         tmpIO = np.repeat([2*(pred_out - rating)],num_feat, axis = 0)
         tmpIO = tmpIO.transpose()
         Ix_m = np.multiply(tmpIO, w1_P1[np.ix_(aa_p)]) + lambda_t * w1_M1[np.ix_(aa_m)]
         Ix_p = np.multiply(tmpIO, w1_M1[np.ix_(aa_m)]) + lambda_t * w1_P1[np.ix_(aa_p)]
         
         dw1_M1 = np.zeros((num_m, num_feat))
         dw1_P1 = np.zeros((num_p, num_feat))
         
         for ii in range(N):
            dw1_M1[aa_m[ii]] = dw1_M1[aa_m[ii]] + Ix_m[ii]
            dw1_P1[aa_p[ii]] = dw1_P1[aa_p[ii]] + Ix_p[ii]
          
          #update movie and user features
         w1_M1_inc = momentum * w1_M1_inc + epsilon * dw1_M1/N 
         w1_M1 = w1_M1 - w1_M1_inc
         w1_P1_inc = momentum * w1_P1_inc + epsilon * dw1_P1/N 
         w1_P1 = w1_P1 - w1_P1_inc
         
    #Compute predictions after parameters updates
    pred_out = np.sum(np.multiply(w1_M1[np.ix_(aa_m)], w1_P1[np.ix_(aa_p)]), axis = 1)
    f_s = np.sum(np.multiply(pred_out - rating, pred_out - rating) 
                     + 0.5 * lambda_t 
                     * np.sum(np.multiply(w1_M1[np.ix_(aa_m)], w1_M1[np.ix_(aa_m)]) + np.multiply(w1_P1[np.ix_(aa_p)], w1_P1[np.ix_(aa_p)]), axis = 1))
    err_train[epoch] = np.sqrt(f_s / N)

    NN=lentest
    aa_m = traindata[0:NN,0]
    aa_p = traindata[0:NN,1]
    rating = traindata[0:NN,2]

    pred_out = np.sum(np.multiply(w1_M1[np.ix_(aa_m)], w1_P1[np.ix_(aa_p)]), axis = 1) + mean_rating
    ff = pred_out > 5
    pred_out[ff] = 5
    ff = pred_out < 1
    pred_out[ff] = 1

    err_valid[epoch] = np.sqrt(np.sum(np.multiply(pred_out - rating, pred_out - rating))/NN)
    print ("epoch %d batch %d Training RMSE %f Test RMSE %f" % (epoch, batch, err_train[epoch], err_valid[epoch]))  
          
