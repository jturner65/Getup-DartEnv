#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:09:33 2018

@author: john

"""


#utilities used throughout multiple files 
# use '~/env_expData/assist/assist_d_3_XXXX.json' wanting to make a new assistance object, and then replace with made assist object's actual name
makeNewAssistObjFName='~/env_expData/assist/assist_d_3_XXXX.json'
#location of current assist object - if not found will make from scratch
#assistFileName ='/home/john/env_expData/assist/assist_d_3_cmpNames_frcMultx_frcMulty_frcMultz_initVals_0.05_0.05_0.0_useFrcObs_N/assist_d_3_cmpNames_frcMultx_frcMulty_frcMultz_initVals_0.05_0.05_0.0_useFrcObs_N_2018-09-14-13-18-51-113337.json' 

#TODO name could get ridiculous.  probably need to come up with better config
assistFrcFileName ='~/env_expData/assist/assist_d_3_cmpNames_frcMultx_frcMulty_frcMultz_bndVals_min0.000_0.300_-0.001_max0.200_0.800_0.001_useFrcObs_N/assist_d_3_cmpNames_frcMultx_frcMulty_frcMultz_bndVals_min0.000_0.300_-0.001_max0.200_0.800_0.001_useFrcObs_N_2018-09-30-12-07-50-578172.json'
#assist file name for cosntraint
assistCnstrntFileName ='~/env_expData/assist/assist_d_3_XXXX.json'

#assistFileToUse = assistFrcFileName

#file holding standing end effector location w/respect to com stats : momments, min/max, correlation mat between dims, convex hull surface verts
#load this to build multi-variate fleishman distribution to provide end effector traj target locations for constraint target
goalEefLocRelComMmntsFile='mmntsDelPts_eefLocs_jConfigs_relCOM_160000_vals.csv'

#import and construct a fleischman distribution simualtion which will build a distribution that exhibits the same behavior as 
#the 1st 4 moments of the passed data or the 4 moments if they are passed instead (isData = False)
def buildFlDist(_data, showSimRes=False, isData=True):
    from fleishmanDist import flDistCalc
    distObj = flDistCalc(_data=_data, showSimRes=showSimRes, isData=isData)
    return distObj

#mutivariate version - always requires data, or moments + correlation mat.
def buildMVFlDist(_data, showSimRes=False, isData=True):
    from fleishmanDist import MV_flDistCalc
    distObj = MV_flDistCalc(_data=_data, showSimRes=showSimRes, isData=isData)
    return distObj

#load eefMmnts, corrMat and hull surface points in dictionary, to use to build Multi Var fleishman dist without requiring source point cloud data
def loadEefMmntsForMVFlDist(baseFileDir='~/dart-env/gym/envs/dart',fileNameMMnts=goalEefLocRelComMmntsFile):
    import numpy as np
    import os
    directory = os.path.join(os.path.expanduser(baseFileDir))
    fileName = os.path.join(directory, fileNameMMnts) 
    
    f=open(fileName, 'r')
    src_lines = f.readlines()   
    f.close()  
    #idxs : 0,4,5,10 can all be ignored
    #idxs 1-3 (cols 1->) have mmnts of data
    #idxs 6-8 (cols 1,2,3) have corr mat (3x3)
    #idx 9 col 1 has # of hull points for delaunay triangularization
    #idx 11-> have all hull surface points
    #need to build the following - hull should be  : 
    mmntsRes = {}
    # self.buildDelHull(_mmntsCorDel['delHullPts'])
    # self.corrMat = _mmntsCorDel['corrMat']
    # mmnts = _mmntsCorDel['mmnts']
    #get moments from cols 1->end of rows 1-4
    mmntsRes['mmnts']=np.array([m for m in [[float(x) for x in src_lines[i].split(',')[1:]] for i in range(1,4)]])
    #get corr mat fromm cols 1->3 of rows 6->8
    mmntsRes['corrMat']=np.array([m for m in [[float(x) for x in src_lines[i].split(',')[1:]] for i in range(6,9)]])
    #get # of hull verts
    numHullPts = int(src_lines[9].split(',')[1])  
    mmntsRes['numHullPts'] = numHullPts
    startIdx = 11
    endIdx = startIdx + numHullPts
    mmntsRes['delHullPts']=np.array([m for m in [[float(x) for x in src_lines[i].split(',')] for i in range(startIdx,endIdx)]])
    return mmntsRes

#manually modify reach hand shoulder and elbow joints, build lists of joint configs and eef locs
#ANA should be standing for this to find target standing eef positions
#anaHldr : ana's skel holder
def sweepStandingEefLocs(anaHldlr, numSmplsPerDof = 20):
    import numpy as np
    jConfigs = []    
    eefLocs = np.zeros(shape=(numSmplsPerDof**4, 3))    
    shldrXVals= np.linspace(-1.1,0, num=numSmplsPerDof, endpoint=True)
    shldrZVals =  np.linspace(0.0,0.5, num=numSmplsPerDof, endpoint=True)
    shldrTwistVals = np.linspace(-0.3,0.3, num=numSmplsPerDof, endpoint=True)
    firstVals = numSmplsPerDof//2
    secondVals = numSmplsPerDof - firstVals
    elbowVals=np.concatenate([np.linspace(2.6,1.5, num=firstVals, endpoint=False),np.linspace(1.5,2.6, num=secondVals, endpoint=True)])
    i=0
    
    for x in shldrXVals:
        qAra = anaHldlr.skel.q        
        qAra[-3] = x
        for sy in shldrTwistVals:
            qAra[-2] = sy            
            for z in shldrZVals:
                qAra[-4] = z    
                for y in elbowVals :
                    qAra[-1] = y
                    jConfigs.append((z,x,sy, y))
                    anaHldlr.skel.set_positions(qAra)
                    #find eef pos relative to body COM
                    eefPos = anaHldlr.reachBody.to_world(anaHldlr.reachBodyOffset)# - relLoc#anaHldlr.skel.com()
                    eefLocs[i,:] = eefPos
                    i+=1
    return jConfigs, eefLocs

#build momments of passed ptData matrix,where each row is a data point and each column is an independent dof
def buildStatsData(ptData):
    import numpy as np
    def buildMmnts(data):
        mu = np.mean(data)
        std = np.std(data)
        from scipy.stats import kurtosis, skew
        skew = skew(data)
        kurt = kurtosis(data, fisher=False) #all kurtosis not just excess
        return [mu, std, skew, kurt, min(data), max(data)]
    
    numDims = len(ptData[0])
    mmntsList = []
    for i in range(numDims):
        mmntsList.append(buildMmnts(np.array([x[i] for x in ptData])))
    corrMat = np.corrcoef(np.transpose(ptData))
    return mmntsList, corrMat


def getEefMmntsFileNames(lenEefLocs, homeBaseDir='~/rllab_project1/goalStandEefData/', fileBaseName='eefLocs_jConfigs_relCOM'):
    import os
    directory = os.path.join(os.path.expanduser(homeBaseDir))
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    fileName = os.path.join(directory, '{}_{}_vals.csv'.format(fileBaseName,lenEefLocs)) 
    fileNameMMnts = os.path.join(directory, 'mmntsDelPts_{}_{}_vals.csv'.format(fileBaseName,lenEefLocs)) 
    return fileName, fileNameMMnts
    

#save arm joint config and end effector data along with momments and inter-dof correlations, to both eef data file and (stats data only) to separate mmnts data file   
#homeBaseDir is directory to save within, referenced to home (begins with ~/)
#also save file with only mmnts, corr mat, and delauney triangularization verts of hull - much smaller file.
def saveEefLocs(jConfigs, eefLocs, homeBaseDir='~/rllab_project1/goalStandEefData/', fileBaseName='eefLocs_jConfigs_relCOM'):
    from scipy.spatial import ConvexHull

    mmnts, corrMat = buildStatsData(eefLocs)
    
    fileName, fileNameMMnts = getEefMmntsFileNames(len(eefLocs), homeBaseDir,fileBaseName)
    #write res
    axStrAra=['x','y','z']
    hdrMmntsStr = 'Per axis Eef Loc,mu,std,skew,kurt,min,max'
    mmntStr = '{}\n{}'.format(hdrMmntsStr,''.join(['{}Mmnts :,{}\n'.format(axStrAra[i], ','.join(['{:.18f}'.format(x) for x in mmnts[i]])) for i in range(len(axStrAra))]))
    corrStr=''
    if corrMat is not None:
        hdrCorrStr = 'Inter Column,Correlation,Matrix of, Eef Loc: \n,col 0, col 1, col 2'
        corrStr = '{}\n{}'.format(hdrCorrStr,''.join(['row {}, {}\n'.format(r, ','.join(['{:.18f}'.format(corrMat[r,c]) for c in range(3)])) for r in range(3)]))
    mmntsCorrHdrStr = '{}{}'.format(mmntStr,corrStr)
    
    #save mmnts file
    f=open(fileNameMMnts, 'w')
    f.write(mmntsCorrHdrStr)
    #save delauney traingularization points of hull of eeflocs so hull membership can be computed by MVfleishman class
    hull= ConvexHull(eefLocs)    
    #get hull points to build delauney triangularization of hull points (much smaller set than size of _data)
    hullPts = hull.points[hull.vertices]
    f.write('HullPts :,{}\nx,y,z\n'.format(len(hullPts)))
    for pt in hullPts:
        f.write('{:.18f},{:.18f},{:.18f}\n'.format(pt[0],pt[1],pt[2]))        
    f.close()    
    #save jconfig and eefs
    f=open(fileName, 'w')
    f.write(mmntsCorrHdrStr)
    f.write('Captured, Upright, Eef Locs, relative to COM, with Various, right arm, joint configs : \n')
    f.write('qIdx -4 (shldrZ), qIdx -3 (shldrX), qIdx -2 (shldrTwst), qIdx -1 (elbow), eef X, eef Y, eef Z\n')
    for x in range(len(eefLocs)):
        resStr1 = ','.join(['{:.8f}'.format(i) for i in jConfigs[x]])
        resStr2 = ','.join(['{:.8f}'.format(i) for i in eefLocs[x]])
        resStr = '{},{}'.format(resStr1, resStr2)
        #print(resStr)
        f.write('{}\n'.format(resStr))    
    f.close()    
    return fileName,fileNameMMnts

#srcData should be 3-d point data in numpy matrix - cols are dims (3) and rows are samples
# or set isData to false, and send array of per dim coeffs
#function to test multivariate fl polynomial derived non-normal distribution
def testMVFlschDist(srcData, showSimRes, isData, N, plotSrcData=True, plotSimData=True):
    #plot res
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot3dData(ax, data, numPltPts, cmap):
        idxVec = np.random.choice(len(data), size=numPltPts, replace=False)
        xd = data[idxVec,0]
        yd = data[idxVec,1]
        zd = data[idxVec,2]
        ax.scatter3D(xd, zd, yd, c=zd, cmap=cmap)

    #build sim dist object
    flSim = buildMVFlDist(srcData, showSimRes=showSimRes, isData=isData)
    simData = flSim.genMVData(N=N, doReport=True, debug=True)
    
    ax = plt.axes(projection='3d')
    
    ax.set_xlabel('X dim')
    ax.set_zlabel('Y dim')
    ax.set_ylabel('Z dim')  
    numPltPts = N//100
    plotSrcData = plotSrcData and isData    
    if (plotSrcData):        
        plot3dData(ax, srcData, numPltPts, 'Greens')
    if (plotSimData):
        plot3dData(ax, simData, numPltPts, 'Reds')

    return flSim, simData