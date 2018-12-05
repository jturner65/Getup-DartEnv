#author : john
#version for 2-bot env testing

import gym
import numpy as np
import sys
import os

def renderAndSave(env, recording, imgFN_prefix, i):
    import imageio
    img = env.render(mode='rgb_array')
    if (recording):
        fileNum = "%04d" % (i,)
        fileName = imgFN_prefix + '_'+ fileNum + '.png'
        imageio.imwrite(fileName, img)        
        print('Saving img {}'.format(fileName))
        
def plotRewards(rwdsAra):
    import matplotlib.pyplot as plt
    plt.hist(rwdsAra, bins='auto')    
    
    
if __name__ == '__main__':    
    recording = False    
    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        env = gym.make('DartStandUp3d_2Bot-v2')#getup environment using constraint delta instead of assist force
        #env = gym.make('DartStandUp3d_2Bot-v1') #main getup w/assist environment
        #env = gym.make('DartStandUp3d_GAE-v1')  #test GAE paper results
        #env = gym.make('DartStandUp3d_2BotAKCnstrnt-v1') #standard environment compatible with akanksha's constraint policy
        #env = gym.make('DartKimaStandUp-v1')#akanksha's environment
        #env = gym.make('DartWalker2d-v1')
        #env = gym.make('DartHumanWalker-v1')
    env.env.disableViewer = False
    anaHldr = env.env.skelHldrs[env.env.humanIdx]
    print('Configure View')
    for i in range(500):
        #env.reset()#test range on reset
        env.render()  
    print('Configure View Done')
    #NOTE :gae env has no botIdx component
    #env.env.skelHldrs[env.env.botIdx].debug_IK = True
    #env.env.skelHldrs[env.env.botIdx].debug=False
    #this will have every step display reward below

    #setting reward components
    #rwdList=['eefDist','action','height','lFootMovDist','rFootMovDist','comcop','UP_COMVEL','X_COMVEL','Z_COMVEL','kneeAction','matchGoalPose','assistFrcPen']
    #to be able to modify global value for using simple reward 933.48082/622.32055
    #env.env.setDesiredRwdComps(rwdList)    
        
    env.env.skelHldrs[env.env.humanIdx].numStepsDBGDisp = 1
    #curAvgFootLocAra = env.env.skelHldrs[env.env.humanIdx].calcAvgFootBodyLoc()#
    #save states via internal state recording mechanisms in skelholder
    #env.env.skelHldrs[env.env.humanIdx].dbgGetStateDofNames()
    #env.env.skelHldrs[env.env.humanIdx].skel.com()
#    rbCom = env.env.skelHldrs[env.env.humanIdx].reachBody.com()
#    rbLclCom = env.env.skelHldrs[env.env.humanIdx].reachBody.local_com()
#    rbLinVel = env.env.skelHldrs[env.env.humanIdx].reachBody.com_linear_velocity()
#    rbSpVel = env.env.skelHldrs[env.env.humanIdx].reachBody.com_spatial_velocity()
#    linJacEef = env.env.skelHldrs[env.env.humanIdx].reachBody.linear_jacobian(offset=rbLclCom)
#    wldJacEef = env.env.skelHldrs[env.env.humanIdx].reachBody.world_jacobian(offset=rbLclCom)
#    dq = env.env.skelHldrs[env.env.humanIdx].skel.dq
#    linJacEef.dot(dq)  
#    wldJacEef.dot(dq)
    
    
    #env.env.skelHldrs[env.env.humanIdx].getWorldPosCnstrntToFinger()
    #env.env.skelHldrs[env.env.humanIdx].setStateSaving( True, 'tmpSaveStates.csv')
    #env.env.skelHldrs[env.env.humanIdx].cnstrntBody.com()
    #env.env.skelHldrs[env.env.humanIdx].initEffPosInWorld
    #env.env.skelHldrs[env.env.botIdx].dbgShowTauAndA()
    #set this if recording
    imgFN_prefix=''
    if (recording):
        dirName = env.env.getRunDirName()
        directory = os.path.join(os.path.expanduser( '~/dartEnv_recdata/') + dirName)
        if not os.path.exists(directory):
            os.makedirs(directory)
        imgFN_prefix = os.path.join(directory, env.env.getImgName())      
     
    #reset environment to be training
    #env.env.setTrainAndInitBotState(False)
    rwds =[]
    #env reset required before calling step
    imgIncr = 0
    maxDisp = 0
    for j in range(3):
        env.reset()
        renderAndSave(env, recording, imgFN_prefix,imgIncr)
        imgIncr+=1
#        actRwds1 = list()
#        actRwds2 = list()
        for i in range(100):
            done = False    
        #while not (done):
            #mult = -((i%21)-10)/10.0
            #action space must be human-sized, since human is only one using external actions
            rand_action = env.env.skelHldrs[env.env.humanIdx].action_space.sample()
                #rand_action = mult*np.ones(rand_action.shape)
    #            if i % 10 == 0:
            rand_action = np.zeros(rand_action.shape)
      #      kneeDofs = anaHldr.kneeDOFActIdxs           
     #       rand_action[kneeDofs[1]] = -5
    #            elif i % 10 == 4:
    #                rand_action = np.ones(rand_action.shape)                
            ob, reward, done, _ = env.step(rand_action)  
            
            #f2CVec = env.env.skelHldrs[env.env.humanIdx].getWorldPosCnstrntToFinger()
            #lenF2CVec = np.linalg.norm(f2CVec)
            #maxDisp = lenF2CVec if lenF2CVec > maxDisp else maxDisp
            #print('Obs assist component : {} | dist between finger and ball : {}, maxDisp  : {} '.format(ob[-3:],lenF2CVec,maxDisp))
            rwds.append(reward)            
            renderAndSave(env, recording, imgFN_prefix,imgIncr)
            imgIncr+=1
            #q=anaHldr.skel.q
            #print('knee dof : {}'.format(q[kneeDofs[0]]))
            #input()
            
        env.env.skelHldrs[env.env.humanIdx].checkRecSaveState()
            #compare performance of weighted and unweighted dofs action rewards
#            actionRew, _ = env.env.skelHldrs[env.env.humanIdx].getRwd_expActionMin(optVal=env.env.skelHldrs[env.env.humanIdx].a,var=env.env.skelHldrs[env.env.humanIdx].sqrtNumActDofs, wt=1.0, mult=1.0)
#            actRwds1.append(actionRew)
#            actionRew, _ = env.env.skelHldrs[env.env.humanIdx].getRwd_expDofWtActionMin(optVal=env.env.skelHldrs[env.env.humanIdx].a,var=env.env.skelHldrs[env.env.humanIdx].sqrtNumActDofs, wt=1.0, mult=1.0)
#            actRwds2.append(actionRew)
            #print('{}'.format(env.env.skelHldrs[env.env.humanIdx].skel.dq))
            #env.env.dart_world.step()
            #print('Curr X : {}'.format(env.env.skelHldrs[env.env.botIdx].nextGuess))        
            #env.env.skelHldrs[env.env.botIdx].testMAconstVec(env.env.skelHldrs[env.env.botIdx].nextGuess)
            #env.env.skelHldrs[env.env.botIdx].nextGuess
            #env.env.skelHldrs[env.env.botIdx].skel.bodynodes[0].com()
            #env.env.skelHldrs[env.env.botIdx].skel.q
            #env.env.trackTraj.trackObj.com()
            
    #        qAra = env.env.skelHldrs[env.env.humanIdx].skel.q        
    #        #qAra[24] = .5
    #        env.env.skelHldrs[env.env.humanIdx].skel.set_positions(qAra)
    #        env.env.skelHldrs[env.env.humanIdx].standHeadCOMHeight
    #        print(env.env.skelHldrs[env.env.humanIdx].skel.q )
    #        env.env.skelHldrs[env.env.humanIdx].standCOMHeightOvFt
    #        env.render()   
#        plotRewards(actRwds1)
#        plotRewards(actRwds2)

            
            #input()
        

    #        img = env.render(mode='rgb_array')
    #        if (recording):
    #            fileNum = "%04d" % (i,)
    #            fileName = imgFN_prefix + '_'+ fileNum + '.png'
    #            scipy.misc.imsave(fileName, img)        

#    env.env.skelHldrs[env.env.botIdx].nextGuess
#    env.env.skelHldrs[env.env.botIdx].dbg_dispMinMaxGuesses()
#    env.env.skelHldrs[env.env.botIdx].dbg_dispMinMaxForce() 
#    env.env.skelHldrs[env.env.humanIdx].dbgShowDofLims()
#    skelJtLims = env.env.skelHldrs[env.env.humanIdx].getJointLimits()
       
    #env.env.displayState()    
    #env.close()
    

#def testFuncGraph():
#    import numpy as np
#    import matplotlib.pyplot as plt
#    from mpl_toolkits import mplot3d
#    x = np.linspace(-1,1,101)
#    y = np.linspace(-1,1,101)
#    X,Y = np.meshgrid(x,y)
#    
#    Z = X + Y
#    
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    
#    surf = ax.plot_surface(X, Y, Z)

def testVecDists():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    dirVec = np.array([.5, .5, .5])
    scaleVec = np.array([.1,.1,.1])*3.0
    rndDir = np.random.normal(loc=dirVec, scale=scaleVec, size=(5000,3))
    #normRndDir = np.reshape(np.linalg.norm(rndDir,axis=1), (1,1000))
    normRndDir = np.linalg.norm(rndDir,axis=1)
    rndDirNorm = rndDir/(normRndDir[:,None])
    
    #plt.scatter(rndDir[:,0], rndDir[:,1])
    ax = plt.axes(projection='3d')
    
    ax.set_xlabel('X dim')
    ax.set_zlabel('Y dim')
    ax.set_ylabel('Z dim')  
    ax.scatter3D(rndDirNorm[:,0], rndDirNorm[:,1],rndDirNorm[:,2])    
    
    
def testSaveAssist(env):
    aDict=env.env.buildAssistObj()    
    fname=aDict['objFileName']

    #/home/john/env_expData/assist/assist_d_3_cmpNames_frcMultx_frcMulty_frcMultz_initVals_0.1_0.2_0.3_useFrcObs_N/assist_d_3_cmpNames_frcMultx_frcMulty_frcMultz_initVals_0.1_0.2_0.3_useFrcObs_N_2018-09-14-13-24-39-876887.json

    newAssist=env.env.getAssistObj(fname,env.env)
    assist=aDict['assist']
    assist.compThisObjToThat(newAssist)
    
def plot3dData(ax, data, numPltPts, cmap, label=""):
    idxVec = np.random.choice(len(data), size=numPltPts, replace=False)
    xd = data[idxVec,0]
    yd = data[idxVec,1]
    zd = data[idxVec,2]
    ax.scatter3D(xd, zd, yd, c=zd, cmap=cmap, label=label)
    

#srcData should be 3-d point data in numpy matrix - cols are dims (3) and rows are samples
def testMVFlschDist(srcData, showSimRes, isData, N, flSim=None, plotSrcData=True, plotSimData=True, useHull=True):
    #build sim dist object
    if flSim is None :
        flSim = env.env.buildMVFlDist(srcData, showSimRes=showSimRes, isData=isData)
    simData = flSim.genMVData(N=N, doReport=True, debug=True, useHull=useHull)
    
    #plot res
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    
    ax = plt.axes(projection='3d')
    
    ax.set_xlabel('X dim')
    ax.set_zlabel('Y dim')
    ax.set_ylabel('Z dim')  
    numPltPts = N//100
    plotSrcData = plotSrcData and isData    
    if (plotSrcData):        
        plot3dData(ax, srcData, numPltPts, 'Greens', label='Source Data')
    if (plotSimData):
        plot3dData(ax, simData, numPltPts, 'Reds', label='Sim Data')

    return flSim, simData

def plotMultDataSets(datAra, N, listOfNames):
    #plot res
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    seqClrMap=['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X dim')
    ax.set_zlabel('Y dim')
    ax.set_ylabel('Z dim')  
    numPltPts = N//100
    i=0
    for dat in datAra : 
        plot3dData(ax, dat, numPltPts, cmap=seqClrMap[(i+1) % len(seqClrMap)], label=listOfNames[i])
        i+=1
    plt.legend(loc=2)
    
#load eefMmnts, corrMat and hull surface points, use to build Multi Var fleishman dist
def loadEefLocMmnts(fileNameMMnts):
    f=open(fileNameMMnts, 'r')
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

#manually modify skeleton
def trackStandingEefLoc(env):
    #build jnt configs and eef locs for current pose (make sure is standing)
    jConfigs, eefLocs,fileName,fileNameMMnts = env.env.buildJntCnfgEefLocs(saveToFile=True, numSmplsPerDof=20)   
    fileNameEefLocs, fileNameMMnts = env.env.getFileNameEefLocsMMnts()
    mmntsData = loadEefLocMmnts(fileNameMMnts)

    env.env.sampleGoalEefRelLoc(fileNameMMnts) #, showSimRes=True, doReport=True, dbgMVSampling=True)
    
    #test fleishman dist object with data
    eefLocsSimDataFlObj,eefSimData = testMVFlschDist(eefLocs,isData=True, showSimRes=True, N=len(eefLocs),useHull=True)
    #test fleishman multivariate sim with mmnts/corr/hull pts dict
    eefLocsSimMmntsFlObj,eefSimMmntsData = testMVFlschDist(mmntsData, isData=False, showSimRes=True, N=len(eefLocs),useHull=True)
#    uniData = np.random.uniform(low=1.0, high=11.0, size=(100000,4))
#    uniSimObj, uniSimData = testMVFlschDist(uniData, N=len(uniData))
    #compare both sim sets
    plotMultDataSets([eefLocs,eefSimData,eefSimMmntsData],N=len(eefLocs),listOfNames=['Actual Locations','Simulated Distribution from Locations','Simulated Distribution From Coefficients'])
    
   
    
    
        #frwrd step sim until robot hits ground
#    cntctInfo = env.env.skelHldrs[env.env.humanIdx].getMyContactInfo()
    #env.env.skelHldrs[env.env.botArmIdx].skel.body('base_link').com()
#    while len(cntctInfo) < 1 :#looping to enable camera control
#        env.env.dart_world.step()
#        cntctInfo = env.env.skelHldrs[env.env.botIdx].getMyContactInfo()
#        env.render()
    #get a dictionary holding variables used in optimization process
    #mass matrix, coriolis + grav, various jacobians
    #print('ball com : {} | \tball world cnst loc : {} |\thuman eff loc :{} '.format(env.env.grabLink.com(),env.env.skelHldrs[env.env.humanIdx].getWorldPosCnstrntOnCBody(),env.env.skelHldrs[env.env.humanIdx].dbg_getEffLocWorld()))


    #show skeleton quantities
    #env.env.skelHldrs[env.env.botIdx].lbndVec
    
    #resDict = env.env.getOptVars()
    
    #use the below to reference into bot holder
    #next_guess = env.env.skelHldrs[env.env.botIdx].nextGuess
    #next_tau = env.env.skelHldrs[env.env.botIdx].nextGuess[env.env.skelHldrs[env.env.botIdx].tauIDXs]
    #cntctList = env.env.skelHldrs[env.env.botIdx].env.dart_world.collision_result.contacts 
    #cntctInfo = env.env.skelHldrs[env.env.botIdx].getMyContactInfo()
    #env.env.dart_world.step()
    #ttlCntctFrces = env.env.skelHldrs[env.env.botIdx]._cntctFrcTtl
    #env.env.grabLink.mass  
    #frcRes = env.env.skelHldrs[env.env.botIdx].frcD
#    propQdot = env.env.skelHldrs[env.env.botIdx].nextGuess[env.env.skelHldrs[env.env.botIdx].qdotIDXs] 
#    propCntFrc = env.env.skelHldrs[env.env.botIdx].nextGuess[env.env.skelHldrs[env.env.botIdx].fcntctIDXs] 
#    propTau = env.env.skelHldrs[env.env.botIdx].nextGuess[env.env.skelHldrs[env.env.botIdx].tauIDXs] 
    #env.env.skelHldrs[env.env.botIdx].numOptIters = 1000
    #frcRes['jtDotTau'] + frcRes['jtDotMA'] + frcRes['jtDotCGrav'] + frcRes['jtDotCntct']
    #frcRes['jtDotTau'] - frcRes['jtDotMA'] - frcRes['jtDotCGrav'] - frcRes['jtDotCntct']
    #calls dart env - env.env.<func name>
    #env.env.setInitPos()
    #env.env.displayState()
    #env.env.frcDebugMode =True
    #unbroken iters

def MC():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    from skimage import measure
    from skimage.draw import ellipsoid
    
   
    # Generate a level set about zero of two identical ellipsoids in 3D
    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    ellip_double = np.concatenate((ellip_base[:-1, ...],
                                   ellip_base[2:, ...]), axis=0)
    
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes_lewiner(ellip_double, 0)
    
    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    
    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")
    
    ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 20)  # b = 10
    ax.set_zlim(0, 32)  # c = 16
    
    plt.tight_layout()
    plt.show()



def calcHull(eefLocs):
  
#Simplex representation
#======================
#The simplices (triangles, tetrahedra, ...) appearing in the Delaunay
#tessellation (N-dim simplices), convex hull facets, and Voronoi ridges
#(N-1 dim simplices) are represented in the following scheme::
#
#    tess = Delaunay(points)
#    hull = ConvexHull(points)
#    voro = Voronoi(points)
#
#    # coordinates of the j-th vertex of the i-th simplex
#    tess.points[tess.simplices[i, j], :]        # tessellation element
#    hull.points[hull.simplices[i, j], :]        # convex hull facet
#    voro.vertices[voro.ridge_vertices[i, j], :] # ridge between Voronoi cells
#
#For Delaunay triangulations and convex hulls, the neighborhood
#structure of the simplices satisfies the condition:
#
#    ``tess.neighbors[i,j]`` is the neighboring simplex of the i-th
#    simplex, opposite to the j-vertex. It is -1 in case of no
#    neighbor.
#
#Convex hull facets also define a hyperplane equation::
#
#    (hull.equations[i,:-1] * coord).sum() + hull.equations[i,-1] == 0
#
#Similar hyperplane equations for the Delaunay triangulation correspond
#to the convex hull facets on the corresponding N+1 dimensional
#paraboloid.
#
#The Delaunay triangulation objects offer a method for locating the
#simplex containing a given point, and barycentric coordinate
#computations.
    
    from scipy.spatial import ConvexHull, Delaunay
    #eefLocs must be numpy array
    hull= ConvexHull(eefLocs)    
    delaHull= Delaunay(hull.points[hull.vertices])
    p = np.array([1,1,1])
    delaHull.find_simplex(p)>=0
   
    return hull, 
        
