import numpy as np
import mbase_utils as mu
from numpy import linalg as LA

def rcprTrain(Is, pGt, model, pStar, posInit, pCur, pGtN, imgIds, T, L, N, N1, regPrm, ftrPrm, regModel, pad, verbose):
    D = pGt.shape[1]
    pAll = np.zeros((N1,D,T+1), dtype = float )
    regs = np.tile({'regInfo': [], 'ftrPos': []}, (T,1))
    
    if regModel.size == 0:
        t0 = 0
        pAll[:,:,0] = pCur[0:N1,:]
    
    loss = np.mean(mu.dist(model,pCur, pGt))
    print 't = %d/%d , loss = %f' % (t0, T, loss)        
    bboxes = posInit[imgIds,:]
    
    for t in range(t0,T):
        pTar = mu.inverse(model, pCur, bboxes)
        pTar = mu.compose(model, pTar, pGt, bboxes)
        
        if ftrPrm['type'] > 2:
            ftrPos = mu.ftrsGenDup(model)
            ftrs, regPrm['occlD'] = mu.ftrsCompDup(model, pCur, Is, ftrPos, imgIds, pStar, posInit, regPrm['occlPrm'])

        regPrm['ftrPrm'] = ftrPrm
        regInfo, pDel = regTrain(ftrs, pTar, regPrm)
        pCur = mu.compose(model, pDel, pCur, bboxes)
        pCur = mu.reprojectPose(model, pCur, bboxes)
        if loss < 1e-5 :
            T = t
            break
       
         
def regTrain(data, ys, regPrm): 
     type = regPrm['type']
     ftrPrm = regPrm['ftrPrm']
     K = regPrm['K']
     loss = regPrm['loss']
     R = regPrm['R']
     M = regPrm['M']
     model = regPrm['model']
     prm = regPrm['prm']
     occlD = regPrm['occlD']
     occlPrm  = regPrm['occlPrm']
    
     stdFtrs, dfFtrs = statsFtrs(data,ftrPrm)
     
     N,D = ys.shape
     
     ysSum = np.zeros((N,D), dtype = float)
     Stot = occlPrm['Stot']
     
     regInfo = np.empty((K,Stot), dtype = object)
     
     nGroups = 9
     masks = np.zeros(Stot)
     for s in range(Stot):
         masks[s] =  np.random.randint(0, nGroups)
     mg = np.median(occlD['group'], axis = 0)
     ftrsOccl = np.zeros((N, K, Stot), dtype = float)
     
     for k in range(K):
         ysTar = ys - ysSum
         ysPred = np.zeros((N,D, Stot))
         for s in range(Stot):
             if s > 0:
                keep = 0   
             else:
                 use, ftrs = selectCorrFeat(M,ysTar, data, ftrPrm, stdFtrs, dfFtrs)
             reg1, ys1 = trainFern(ysTar, ftrs, M, prm)
             reg1['fids'] = use
             best = {reg1, ys1}
             regInfo[k,s] = best
             ysPred[:,:,s] = best
             ftrsOccl[:,k,s] = np.sum(occlD['featOccl'][:, regInfo[k,s]]['fids'], axis = 1) / K
                  
             
def trainFern(Y, X, S, thrr = [-0.2, 0.2], reg = 0.01):
    N,D = Y.shape
    fids = range(S)
    thrs = np.random.uniform(thrr[0],thrr[1],size = S)
    
                 
def statsFtrs(ftrs, ftrPrm):
    N,_ = ftrs.shape             
    if ftrPrm['type'] == 1:
        stdFtrs = np.std(ftrs)
        muFtrs = np.mean(ftrs, axix = 0)
        dfFtrs = ftrs - np.tile(muFtrs, (N,1))
    else:
        muFtrs = np.mean(ftrs,axis = 0)
        dfFtrs = ftrs - np.tile(muFtrs, (N,1))
        stdFtrs = stdDFftrs(ftrs)
    return (stdFtrs, dfFtrs)
 
   
   
def stdDFftrs(ftrs):  
    F = ftrs.shape[1]            
    stdrst = np.zeros((F,F),dtype = float)
    for i in range(F):
        for j in range(i+1, F):
            stdrst[i,j] = np.std(ftrs[:,i]-ftrs[:,j])
    
    return stdrst

   
def trainFerm(Y,X,M,prm):
    thrr = prm['thrr']
    reg = prm['reg']  
    N,D = Y.shape
    fids = np.arange(M)  
    thrs = np.random.rand(0,M)*(thrr[1] - thrr[0]) + thrr[0]
    inds, mu, ysFern, ncount, _ = fernsInds2(X, fids, thrs, Y)
    cnts = np.tile(ncount, (1,D))
    S = cnts.shape[1]
    
    for d in range(D):
        ysFern[:, d] = ysFern[:, d] / np.maximum(cnts[:,d]+reg*N, np.spacing(1)) + mu[d]

    Y_pred = ysFern[inds,:]
    regSt = {'ysFern' : ysFern, 'thrs' : thrs} 

    return (regSt, Y_pred)

def fernsInds2(X, fids, thrs, Y ):     
     N,F = X.shape
     M,S = fids.shape
     D = len(Y)
     inds = np.zeros((N,M), dtype = float)
     for m in range(M):
         for s in range(S):
             for n in range(N):
                 inds[n,m] *= 2
                 f = fids[m,s]
                 if X[n,f] < thrs[m, s]:
                     inds[n,m] += 1
     mu = np.zeros((1,D), dtype = float)
     dfYs = np.zeros((N,D), dtype = float)
     
     for d in range(D):
         mu[d] = np.mean(Y[:,d])
         for n in range(N):
             dfYs[n,d] = Y[n,d] - mu[d]
     
             
     S2 = np.power(2, S)
     sumys = np.zeros((S2,D), dtype = float)
     counts = np.zeros((S2,1), dtype = float)
     
      
     for n in range(N*M):
         s = inds.ravel()[n]
         counts[s] += 1
         for d in range(D):
             sumys[s,d] = sumys[s,d] + dfYs[n,d]
         inds[n] += 1                     
     
     return (inds, mu, sumys, counts, dfYs)
     
     

def selectCorrFeat(S, pTar, ftrs, ftrPrm, stdFtrs, dfFtrs):
             _, D = pTar.shape
             b = np.random.rand(S,D) * 2 - 1
             b = np.matrix(b)
             for s in range(S):
                 b[s, :] = b[s, :] / LA.norm(b[s, :])
             scalar = pTar * b.T
             stdSc = np.std(scalar, axis = 0)
             muSc = np.mean(scalar, axis = 0)
             type = ftrPrm['type']
             N,F = ftrs.shape
             if type > 2:
                 type = type - 2
                 use = np.zeros((type,S), dtype = float)  
                 covF = np.zeros(F, dtype = float)
                 used = -1 * np.ones((F,F), dtype = int)
                 for s in range(S):
                     for f in range(F):
                         covF[f] = (dfFtrs[:,f] * (scalar[:,s] - muSc[0,s])).getA()[0] / N
                     for f1 in range(F):
                         for f2 in range(f1+1, F):
                             if used[f2, f1] == 0:
                                 val = (covF[f1] - covF[f2])/(stdFtrs[f1,f2] * stdSc[0,s])
                                 if f1 ==0 and f2 == 1:
                                     maxCo = val
                                     use[0,s] = 0
                                     use[1, s] = 1
                                     used[f2,f1] = 1
                                 else:
                                     if val > maxCo:
                                         maxCo = val
                                         use[0, s] = f1 
                                         use[1, s] = f2 
                                         used[f2,f1] = 1
                                         
                 if np.any(use[:] == -1) :
                     use[use[:] == -1] = np.random.randint(0,F, len(use[use[:] == -1]))
             return (use, ftrs[:,use[0,:]] - ftrs[:, use[1,:]])        
                                         
'''            
def cprApply(Is, regModel, pInit = np.array([]), imgIds = np.array([]), K = 1, rad = 1, chunk = float('inf')):
    p = pInit
    if len(imgIds) == 0:
        imgIds = range(Is.shape[3])
    M = len(imgIds)
    if len(p) == 0:
        p = regModel['pStar'][0:M, 0:2]
        
    if K == 1:
        if p.shape[1] == 2:
            p = [p, regModel['pStar'][0:M, 3:]]
 '''
 
       