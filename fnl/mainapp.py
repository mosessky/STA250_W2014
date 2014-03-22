import h5py
import numpy as np
import argparse
import mbase_utils as mu
import facemodel as fm

rawtrdata = h5py.File('/home/shikai/study/ECS271/finalprj/data/COFW_train.mat')
rawtstdata = h5py.File('/home/shikai/study/ECS271/finalprj/data/COFW_test.mat')

phisTr = np.matrix(rawtrdata['phisTr']).T
bboxesTr = np.matrix(rawtrdata['bboxesTr']).T
nTrrow, nTrcol = rawtrdata['IsTr'].shape 
IsTr = []
for i in range(nTrcol):
    IsTr.append(np.matrix(rawtrdata[rawtrdata['IsTr'][0,i]]).T)

phisT = np.matrix(rawtstdata['phisT']).T
boxesT = np.matrix(rawtstdata['bboxesT']).T
nTstrow, nTstcol = rawtstdata['IsT'].shape 
IsT = []
for i in range(nTstrow):
    IsT.append(np.matrix(rawtstdata[rawtstdata['IsT'][0,i]]).T)

amodel = {'isFace':True, 'name': 'cofw', 'nfids': 29, 'D':87}

#mu.drawImage( amodel, IsTr[13], phisTr[13,:] )

T = 100
K = 15
L = 20
RT1 = 5
ftrPrm = {'type':4, 'F':400, 'nChn':1, 'radius':2}
prm = {'thrr': [-0.2, 0.2], 'reg': 0.01}
occlPrm = {'nrows': 3, 'ncols':3, 'nzones':1, 'Stot':3, 'th':0.05}
regPrm = {'type': 1, 'K': K, 'occlPrm': occlPrm, 'loss': 'L2', 'R' :0, 'M':5, 'model':amodel, 'prm': prm}
prunePrm = {'prune':1, 'maxIter':2, 'th':0.15, 'tIni':10}

pCur, pGt, pGtN, pStar, imgIds, N, N1 = mu.initTr(IsTr, phisTr, amodel, np.array([]), bboxesTr, L, 10)

fm.rcprTrain(IsTr, pGt, amodel, pStar,  bboxesTr, pCur, pGtN, imgIds, T, L, N, N1, regPrm, ftrPrm, regModel = np.array([]), pad = 10, verbose = True)

#mu.drawImage( amodel, IsTr[1], phisTr[:,1] )
