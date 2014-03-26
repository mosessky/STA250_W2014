import matplotlib.pyplot as plt
import numpy as np

#Display Image with markers
def drawImage( fmodel, Is, phis, drawIs = 1, lw = 10 ):
    try:
       if drawIs == 1:
          plt.imshow(Is, cmap = "Greys_r")           
          
       phis = np.squeeze(phis.getA())
       if fmodel['isFace'] :
            D = len(phis)
       if fmodel['name'] == 'cofw':
           nfids =D/3
           occl = phis[nfids * 2 : nfids * 3]
           occl = np.concatenate((occl, occl))
           plt.scatter(phis[occl==0][0:len(phis[occl==0])/2], phis[ occl==0][len(phis[ occl==0])/2:len(phis[ occl==0])],c='g',s=lw )
           plt.scatter(phis[occl==1][0:len(phis[occl==1])/2], phis[ occl==1][len(phis[ occl==1])/2:len(phis[ occl==1])],c='r',s=lw )
       
        
       plt.show()
    except:    
        print "Error in Image input data"

     
   
#initial Train model
def initTr(Is, pGt, model, pStar, posInit, L, pad):
    N,D = pGt.shape
    
    if pStar.size == 0:
       pStar, pGtN,_,_,_ = compPhiStar(model, pGt, Is, pad, np.array([]), posInit)

    nfids = D / 3
    pCur =  np.vstack([ [pGt] * L])
    nfids = model['nfids']
    
    for n in range(N):
        imgsIds = np.random.choice(np.concatenate((range(n),range(n+1,N)), axis = 0) , size = L, replace= False)
        for l in range(L):
            maxDisp = posInit[n, 2:4]/16.0
            uncert= np.multiply((2 * np.random.rand(1,2) - 1), maxDisp)
            bbox=np.matrix(np.copy(posInit[n,:]))
            bbox[0, 0:2] += uncert
            
            pCur[l,n,:] = reprojectPose(model, pGtN[imgsIds[l], :] ,bbox)
            
    pCur = pCur.reshape(L*N, D)
    imgIds = np.tile(range(N),L)
    pGt = np.tile(pGt,  (L, 1))
    pGtN = np.tile(pGtN,(L, 1))
    N1 = N 
    N = N * L
    
    return (pCur, pGt, pGtN, pStar, imgIds, N, N1)


def reprojectPose(model, phis, bboxes):
   
    nfids = model['nfids']
    szX = bboxes[0,2]/2
    szY = bboxes[0,3]/2
    ctrX = bboxes[0,0] + szX
    ctrY = bboxes[0,1] + szY
    phis1 = np.concatenate((phis[0:nfids] * szX + ctrX,  phis[nfids:nfids*2] * szY + ctrY, phis[nfids*2:nfids*3]), axis = 1)


    return phis1

#project Pose
def projectPose(model, phis, bboxes):
    N,D = phis.shape
    nfids = model['nfids']
    szX = bboxes[:,2]/2
    szY = bboxes[:,3]/2
    ctrX = bboxes[:, 0] + szX
    ctrY = bboxes[:, 1] + szY
    szX = np.tile(szX,(1,nfids))
    szY = np.tile(szY,(1,nfids))
    ctrX = np.tile(ctrX,(1,nfids))
    ctrY = np.tile(ctrY,(1,nfids))
    phis = np.concatenate(((phis[:,0:nfids]-ctrX)/szX, (phis[:,nfids:nfids*2]-ctrY)/szY, phis[:,nfids*2:nfids*3]), axis = 1)
    return phis

#compute phiStar
def compPhiStar(model, phis, Is, pad, imgIds, bboxes):
    N,D = phis.shape
    sz = np.zeros((N,2), dtype=float)
    if imgIds.size == 0:
        imgIds = np.array(range(N))
    
    nfids = model['nfids']
    phisN = np.zeros((N,D), dtype = float)
    distPup = np.sqrt(np.multiply(phis[:,16] - phis[:,17], phis[:,16] - phis[:,17]) + 
                      np.multiply(phis[:,16+nfids] - phis[:,17+nfids], phis[:,16+nfids] - phis[:,17+nfids]))
    
    sz = np.zeros((N,2), dtype = float)
    
    #normalize pixel locations,
    #central point's local become (zero, zero)
    for n in range(N):
        img = Is[imgIds[n]]
        sz[n, 0], sz[n, 1] = img.shape
        sz[n, :] = sz[n, :]/2
   
        szX = bboxes[imgIds[n], 2]/2
        szY = bboxes[imgIds[n], 3]/2
        ctr1 = bboxes[imgIds[n],0] + szX
        ctr2 = bboxes[imgIds[n],1] + szY
        
        phisN[n, :] = np.concatenate(((phis[n,0:nfids]-ctr1)/szX, (phis[n,nfids:nfids*2]-ctr2)/szY, phis[n,nfids*2:nfids*3]), axis = 1)
        
    return (np.mean(phisN, axis = 0), phisN, distPup, sz, bboxes)   

'''
Train single random fern regressor.


@param X     [N*F] data measurements
@param fids  [S] fern range 
@param thrs  [S] random thresholds
@param Y     [N*D] target output values

'''
def fernsInd(X, fids, thrs, Y):
     N,F = X.shape()
     S = len(fids)
     D = Y.shape[1]
     inds = np.zeros(N, dtype= float)
     
     for s in range(S):
         for n in range(N):
             inds[n] *= 2 
             if X[n, fids[s] ] < thrs[s]:
                 inds[n] += 1
                              
     mu = np.mean(Y, axis=0) 
     S2 = np.power(2,S)
     
     sumys = np.zeros((S2,D), dtype=float)    
     counts = np.zeros(S2, dtype=float)    
     dfYs = Y - mu
     inds += 1
     
     for i in range(N):
         s = inds[i]
         counts[s] = len(inds[np.where(inds == s)])
     
     sumys += dfYs
     
     return (inds, mu, sumys, counts, dfYs)

      
            
             
def dif(phis0, phis1):             
    
    diff = phis0 - phis1    
    return diff     

def initTest(Is, bboxes, model, pStar, pGtN, RT1):
    N = len(Is)
    D = pStar.shape[1]
    phisN = pGtN
    if bboxes.size == 0:
        p=pStar[np.ones(N),:]
    elif bboxes.shape[1] == 4:
        p = np.zeros((N,D,RT1), dtype = float)
        NTr = phisN.shape[0]
        for n in range(N):
            imgsIds = np.random.choice(NTr,RT1,replace = False)
            for l in range(RT1):
                maxDisp = bboxes[n, 2:3]/16
                uncert = (2*np.random.rand((1,2))-1) * maxDisp
                bbox = bboxes[n,:]
                bbox[0:1] += uncert
                p[n, :, l] = reprojectPose(model, phisN[imgsIds[l],:],bbox)
    elif bboxes.shape[1] == 4 and bboxes.shape[2] == RT1:
         p = np.zeros((N,D,RT1), dtype = float)
         for n in range(N):
             imgsIds = np.random.choice(NTr,RT1, replace = False)
             for l in range(RT1):
                 p[n, :, l] = reprojectPose(model, phisN[imgsIds[1],:], bboxes[n,:,1])
    elif bboxes.shape[1] == D and bboxes.shape[2] == RT1:
         p = bboxes
    
    return p


def ftrsOcclMasks(xs):
    pos = np.empty(9, dtype = object)
       
    pos[0] =  np.arange(xs.shape[0])
    #top half
    pos[1] =  np.nonzero(xs[:,1] <= 0)[0].getA()[0]  
    #bottom half
    pos[2] =  np.nonzero(xs[:,1 > 0])[0].getA()[0]
    #right
    pos[3] =  np.nonzero(xs[:,0] >= 0)[0].getA()[0]
    #left 
    pos[4] =  np.nonzero(xs[:,0] < 0)[0].getA()[0]
    #right top diagonal
    pos[5] =  np.nonzero(xs[:,0] >= xs[:,1])[0].getA()[0]
    #left bottom diagonal
    pos[6] =  np.nonzero(xs[:,0] < xs[:,1])[0].getA()[0]
    #left top diagonal
    pos[7] =  np.nonzero(xs[:,0] * -1.0 >= xs[:,1])[0].getA()[0]
    #right bottom diagonal
    pos[8] =  np.nonzero(xs[:,0] * -1.0 < xs[:,1])[0].getA()[0]
    
    return pos

#Generate random shape indexed features, relative to two landmarks (points in a line, RCRP contribution)
def ftrsGenDup(model):
    type =  4
    F = 400
    radius = 2
    nChn = 1
    
    F2 = np.max([100, np.ceil(F*1.5)])
    nfids = model['nfids']
    xs = np.empty((0,0),dtype = float)
    while  xs.shape[0] < F:
        xs = np.random.randint(0,nfids,(F2,2))
        xs = xs[xs[:,0]!=xs[:,1]]
    
    xs = xs[0:F,:]
    #add one column
    xs = np.append( xs, (2*radius*np.random.rand(F,1)-radius),1)
    pids = np.floor( np.linspace(0,F,2))
    
    ftrData = {'type':type, 'F': F, 'xs':xs, 'pids': pids}
    
    return ftrData        

     
def dist(model, phis0, phis1):
   
    tmpdel = dif(phis0, phis1)
    nfids = model['nfids']
    distPup = np.sqrt(np.multiply(phis1[:,16] - phis1[:,17], phis1[:,16] - phis1[:,17]) + 
                      np.multiply(phis1[:,16+nfids] - phis1[:,17+nfids], phis1[:,16+nfids] - phis1[:,17+nfids]))
    distPup = np.tile(distPup, (1,nfids))
    dsAll = np.sqrt(np.multiply(tmpdel[:,0:nfids], tmpdel[:,0:nfids]) + np.multiply(tmpdel[:,nfids:2*nfids], tmpdel[:,nfids:2*nfids]))
    dsAll = dsAll/distPup
    return np.mean(dsAll, axis = 0)

def inverse(model, phis0, bboxes):
    return - projectPose(model, phis0, bboxes)

def compose(model, phis0, phis1, bboxes):
    phis1 = projectPose(model, phis1, bboxes)
    return phis0+phis1

def getSzIm(Is):
    N = len(Is)
    w = np.zeros(N, dtype = float)
    h = np.zeros(N, dtype = float)
    for i in range(N):
        w[i], h[i], _ = Is[i].shape

#codify Positions        
def codifyPos(x, y, nrows, ncols):
    nr = 1.0/nrows
    nc = 1.0/ncols
    x = np.minimum(1, np.maximum(0,x))
    y = np.minimum(1, np.maximum(0,y))
    y2 = np.copy(y)
    x2 = np.copy(x)
    #locate features to columns
    for c in range(ncols):
        if c == 0:
            x2[x <= nc] = 0
        elif c ==  ncols -1 :
            x2[x >= nc * c] = ncols - 1
        else:
            x2[(x > nc * c) & (x <= nc * (c+1))] = c
    #locate features to rows
    for r in range(nrows):
        if r==0:
            y2[y <= nr] = 0
        elif r == nrows - 1:
            y2[y >= nc * r ] = nrows - 1
        else:
            y2[(y > nr * r) & (y <= nr * (r+1))] = r
    #translate column, row location into one dimension index
    return np.ravel_multi_index([y2.astype(int), x2.astype(int)], (nrows, ncols))

#Compute features from ftrsGenDup on Is
def ftrsCompDup(model, phis, Is, ftrData, imgIds, pStar, bboxes, occlPrm):
    N = len(Is)
    nfids = model['nfids']
    M = phis.shape[0]
    
    bboxes = np.matrix(np.copy(bboxes[imgIds,:]))
    FTot = ftrData['F']
    ftrs = np.zeros((M,FTot), dtype = float)
   
    poscs = phis[:, 0 : nfids]
    posrs = phis[:, nfids : nfids * 2]
     
   
    occl = phis[:, nfids*2: nfids*3]
    occlD = {'featOccl': np.zeros((M,FTot), dtype = int), 'group': np.zeros((M,FTot), dtype = int)}
    csStar, rsStar = getLinePoint(ftrData['xs'], pStar[0:nfids], pStar[nfids: nfids *2])   
    pos = ftrsOcclMasks(np.concatenate((np.matrix(csStar).T,np.matrix(rsStar).T), axis = 1))
    
    cs1, rs1 = getLinePoint(ftrData['xs'], poscs, posrs)
    nGroups = 9
    
    for n in range(M):
        img = Is[imgIds[n]]
        h,w = img.shape
        
        cs1[n,:] = np.maximum(0, np.minimum(w-1, cs1[n,:]))
        rs1[n,:] = np.maximum(0, np.minimum(h-1, rs1[n,:]))
        occlD['group'][n, :] = codifyPos((cs1[n, :] - bboxes[n,0])/bboxes[n,2], (rs1[n,:] - bboxes[n,1])/bboxes[n,3], occlPrm['nrows'], occlPrm['ncols'])
        groupF = codifyPos((poscs[n, :] - bboxes[n,0])/bboxes[n,2], (posrs[n,:] - bboxes[n,1])/bboxes[n,3], occlPrm['nrows'], occlPrm['ncols'])          
        
        occlAm = np.zeros(nGroups, dtype = int)  
        
        for g in range(nGroups):
            occlAm[g] = np.sum(occlD['group'][n, groupF == g])
        occlD['featOccl'][n,:] = occlAm[occlD['group'][n,:]]    
                
        ftrs1 = img[rs1[n,:].astype(int), cs1[n,:].astype(int)].astype(float) / 255.0
        ftrs[n, :] = ftrs1
   
    return (ftrs, occlD)    
                
def getLinePoint(FDxs, poscs, posrs):
  if len(poscs.shape) == 1: 
     l1 = FDxs[:, 0].astype(int)
     l2 = FDxs[:, 1].astype(int)
     xs = FDxs[:, 2]
     x1 = poscs[:, l1]
     y1 = posrs[:, l1]
     x2 = poscs[:, l2]
     y2 = posrs[:, l2]
     a = (y2 - y1) / (x2 - x1)
     b = y1 - np.multiply(a, x1)
     distX = (x2 - x1) /2
     ctrX = x1 + distX
     cs1 = ctrX + xs * distX
     rs1 = a * cs1 + b
  else:   
     l1 = FDxs[:, 0].astype(int)
     l2 = FDxs[:, 1].astype(int)
     xs = FDxs[:, 2]
     muX = np.matrix(np.mean(poscs, 1)).T
     muY = np.matrix(np.mean(posrs, 1)).T
     poscs = poscs - np.tile(muX, (1, poscs.shape[1]))
     posrs = posrs - np.tile(muY, (1, poscs.shape[1]))
     x1 = poscs[:, l1]
     y1 = posrs[:, l1]
     x2 = poscs[:, l2]
     y2 = posrs[:, l2]
     a = (y2 - y1) / (x2 - x1)
     
     b = y1 - np.multiply(a, x1)
    
     distX = (x2 - x1) /2
     ctrX = x1 + distX
     cs1 = ctrX + np.multiply(xs, distX)
     rs1 = np.multiply(a, cs1) + b
     np.place(rs1, cs1==x1, (y1[cs1 == x1] + ((y2 - y1)/2 + np.multiply(xs, (y2-y1)/2))[cs1==x1]).getA()[0,:])
     
    
     cs1 = np.round(cs1 + np.tile(muX, (1, FDxs.shape[0] )))
     rs1 = np.round(rs1 + np.tile(muY, (1, FDxs.shape[0] )))
       
  return (cs1, rs1)
                     