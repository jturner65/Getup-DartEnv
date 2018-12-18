#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# # Modified version of dart_env.py to support 2 skeletons

#from gym import error, spaces
from gym import error
import gym.spaces
from gym.utils import seeding
import numpy as np
#from os import path
import gym
#import six
#for font ref
import OpenGL.GLUT as GLUT


#these need to be added (at least temporarily) to path
import os, sys
glblDartEnvLoc = os.path.expanduser( '~/dart-env/gym/envs/dart')
sys.path.insert(0, glblDartEnvLoc)
from contactInfo import contactInfo
from skelHolders import * #skelHolder,ANASkelHolder,humanSkelHolder,robotArmSkelHolder,robotSkelHolder, kimaHumanSkelHolder,AK_KimaHumanSkelHolder,prone_kimaHumanSkelHolder

import utils_Getup as util

from collections import defaultdict

from gym.envs.dart.static_window import *

try:
    import pydart2 as pydart
    from pydart2.gui.trackball import Trackball
    pydart.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))


"""Superclass for dart environment with 2 bots contained in one skel file.    From gym.Env : 
    'The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed

    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards '   
"""

class DartEnv2Bot(gym.Env):
    #static variable holding what reward components we are using
    #must be some combination of implemented reward components : rwdNames = ['eefDist','action', 'height','footMovDist','lFootMovDist','rFootMovDist','comcop','contacts','UP_COMVEL','X_COMVEL','Z_COMVEL','GAE_getUp','kneeAction','matchGoalPose','assistFrcPen']#,'eefDistNeg']

    
    #below used for successful training of getup
    
    #rwdCompsUsed = ['eefDist','action','height','lFootMovDist','rFootMovDist','comcop','UP_COMVEL','X_COMVEL','Z_COMVEL']
   
    #rwdCompsUsed = ['eefDist','action','kneeAction','height','lFootMovDist','rFootMovDist','comcop','UP_COMVEL']
    #rwdCompsUsed = ['eefDist','action','kneeAction','height','lFootMovDist','rFootMovDist','comcop','UP_COMVEL','X_COMVEL','Z_COMVEL']
    #rwdCompsUsed = ['eefDist','action','height','lFootMovDist','rFootMovDist','UP_COMVEL','comcop']
    #rwdCompsUsed = ['action','height','lFootMovDist','rFootMovDist','UP_COMVEL','comcop']
    #rwdCompsUsed = ['eefDist','action','height','lFootMovDist','rFootMovDist']
    #rwdCompsUsed = ['eefDist','action','contacts','height','lFootMovDist','rFootMovDist']
    #rwdCompsUsed = ['eefDist','action','height']
    #rwdCompsUsed = ['eefDist','action','comcop']
    #rwdCompsUsed = ['action','height']
    #rwdCompsUsed = ['GAE_getUp']
    rwdCompsUsed = []
    #static variable to manage whether or not we are training a policy.  this will enable/disable the viewer and debug displays, among other things. 
    trainPolicy=False#keep this false to default to this value for testing policies
    #whether or not to clamp action values before deriving tau.  normalized environments do this already so not necessary, but not using a normalized environment might require this to be done
    clampA=False
    #!!!! These values are now governed in baseConfig.txt  !!!!!
 
    #file name holding reward configuration
    rwdConfigFileName = '{}/baseConfig.txt'.format(glblDartEnvLoc)

    def __init__(self, model_paths, frame_skip,  dt=0.002, obs_type="parameter", action_type="continuous", visualize=True, disableViewer=False, screen_width=80, screen_height=45):
        assert obs_type in ('parameter', 'image')
        assert action_type in ("continuous", "discrete")
        print('DartEnv2Bot::__init__ : pydart initialization OK')

        #load static configurations from local text file baseConfig.txt - file needs to exist! otherwise will crash here
        DartEnv2Bot.loadStaticVals()       
        #load skels, make world
        self.loadAllSkels(model_paths, dt)  
        #set world reference to this env, so custom world can call env to display on-screen info
        self.dart_world.set2BotEnv(self)
        #init RL-related vars from ctor
        self._obs_type = obs_type
        self.frame_skip = int(frame_skip)
        self.visualize = visualize  #Show the window or not
        #set false if wanting to consume policy, true to train - here this disables viewer
        #if True then manually consuming policy will yield error : AttributeError: 'NoneType' object has no attribute 'runSingleStep' since no viewer exists
        #if False then attempting to train via trpoTests_ExpLite.py will open multiple viewer windows 
        self.trainPolicy = DartEnv2Bot.trainPolicy
        #disable if either explicitly directed to by constructor or if static variable is true
        self.disableViewer = disableViewer or self.trainPolicy
        print('\n!!!!!!!!!!!!!!!!! DartEnv2Bot::Viewer is disabled : {}\n'.format(self.disableViewer))
        
        #set random seed 
        self.seed()
        #set class variables
        self.initVars()
        #set string for date time to use for names of any saved files like renders
        self.initDateTime()
        #dict of all skelHolders - each is convenience class holding all relevant info about a skeleton and interaction with sim
        self.skelHldrs = {}
        #list of skels to render as default color
        self.skelsToRndr = []
        #list of tuples of skels to render (idx0) with color list (idx1)
        self.skelsToRndrWClr= []


        #build list of slkeleton handlers from loaded skeletons list in world
        self._buildskelhldrs()

        #set which skeleton is active for RL, which also sets action and observation space
        self.setActiveRLskelhldr(self.humanIdx)

        #set perturbations if used
        self.setPerturbation(False)

        # initialize the viewer, get the window size
        # initial here instead of in _render in image learning
        self.screen_width = screen_width #these are used in obs_type "image" viewers (for sceen shots?)
        self.screen_height = screen_height
        self.viewer = None
        self._get_viewer()

        #overriding env stuff
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self.dt))
        }
        #this is to initialize all relevant variables for screen overlay text/graphics info
        self.initAllScrDispData()
        #whether or not to draw overlay text on screen - maybe disable for videos
        self.drawUIOverlayData = True

    #ctor initializations
    def initVars(self):
        #save iteration - increment so successive runs sharing today's date are not put in same directories
        self.saveIter = 0
        #number of broken sim runs for this instance
        self.numBrokeSims = 0
        # # of sim steps taken before reset
        self.sim_steps = 0
        #amount of sim time elapsed (add dt every world step)
        self.timeElapsed = 0
        #save and restore world state
        self.tot_SimState = []
        self.stateIsSaved = False
        #whether pausing for user input should be allowed.  disable this during training
        self.allowPauseForInput = False
        #set far-away center point for contact torque calculation to derive COP point from contacts with ground
        self.ctrTauPt = np.array([500,0,500])


    ###########################
    ##  static methods are used primarily by trpoTests_ExpLite to control experiment functionality before envs and skelholders are fully instanced.
    #   TODO replace with experiment class functionality?
    ##########################

    @staticmethod
    #set rewards list to use- must be set before skel holders are called
    def setStaticVals(listOfRwds=[], trainState=None, clampA=None):
        if len(listOfRwds) > 0:
            DartEnv2Bot.rwdCompsUsed = listOfRwds[:]
        if (trainState is not None):
            DartEnv2Bot.trainPolicy = trainState
        if (clampA is not None):
            DartEnv2Bot.clampA = clampA            
        with open(DartEnv2Bot.rwdConfigFileName, 'w+') as f :
            hdrCmmnt = '!-- Base Experimental Configurations\n!-- use !-- for comments\n'
            f.write(hdrCmmnt)
            trainStr = 'train : {}'.format(DartEnv2Bot.trainPolicy)
            clampAStr = 'clampA : {}'.format(DartEnv2Bot.clampA)
            rwdsStr = 'rwds : {}'.format(','.join(DartEnv2Bot.rwdCompsUsed))
            f.write('\n'.join([trainStr, clampAStr, rwdsStr]))

    @staticmethod
    def loadStaticVals():
        with open(DartEnv2Bot.rwdConfigFileName, 'r') as f :
            srcLines = f.read().splitlines()
        splitLines = [x.split(':') for x in srcLines if '!--' not in x]
        srcDict = {pair[0].strip() : pair[1].strip() for pair in splitLines}
        DartEnv2Bot.trainPolicy = ('true' in srcDict['train'].lower())
        DartEnv2Bot.rwdCompsUsed = srcDict['rwds'].split(',')
        DartEnv2Bot.clampA = ('true' in srcDict['clampA'].lower())
        print('Training set to : {}\nRewards used set to :{}'.format(DartEnv2Bot.trainPolicy, DartEnv2Bot.rwdCompsUsed))
        
    @staticmethod
    def getRwdsList():
        DartEnv2Bot.loadStaticVals()
        return DartEnv2Bot.rwdCompsUsed[:]
    
    #returns binary bitstring representation of rewards specified by name in listOfRwds, also sets rewards
    @classmethod
    def getRwdsToUseBin(cls, listOfRwds=None):
        binDig = 0
        if (listOfRwds is not None) and (sorted(listOfRwds) != sorted (cls.rwdCompsUsed)):
            cls.setStaticVals(listOfRwds)
            print('DartEnv2Bot::getRwdsToUseBin : DartEnv2Bot.rwdCompsUsed changed to match passed list of rewards : {}'.format(listOfRwds))
        elif (listOfRwds is None):
            listOfRwds = cls.getRwdsList()
        #build binary digit used to determine which reward components are used for this simulation
        for rwd in listOfRwds :
            try :
                idx = ANASkelHolder.rwdNames.index(rwd)
                binDig = binDig | (2**idx)
            except :
                print('!!!!!!!! dart_env_2bot::getRwdsToUseBin : ERROR : rwd type {} not found in implemented lists of reward components : \n{}'.format(rwd,ANASkelHolder.rwdNames)) 
                pass
        strVal = '{0:0{1}b}'.format(binDig, len(ANASkelHolder.rwdNames))
        return strVal
    
    #build a list of reward names given an int (binDig) or string, which represents bit flags for each reward component
    @staticmethod
    def bldRwdListFromBinDig(binDig):
        if isinstance(binDig, str):
            binDig = int(binDig, base=2)
        rwdList = []
        binMask = 1
        for x in range(len(ANASkelHolder.rwdNames)):
            if (binDig & binMask) > 0  : 
                rwdList.append(ANASkelHolder.rwdNames[x])
            binMask*=2
        return rwdList

    @staticmethod
    def setTrainState(trainState):
        DartEnv2Bot.setStaticVals(trainState=trainState)
        return DartEnv2Bot.trainPolicy

    @staticmethod
    def setClampAction(clampA):
        DartEnv2Bot.setStaticVals(clampA=clampA)
        return DartEnv2Bot.clampA

    #pass in string of rewards, set reward list to use based on this string
    @classmethod
    def setGlblRwrdsFromBinStr(cls, binDigStr):
        listOfRwds = DartEnv2Bot.bldRwdListFromBinDig(binDigStr)
        cls.setStaticVals(listOfRwds)
        return DartEnv2Bot.rwdCompsUsed

    #####################################################################

    #reset ana's reward components after loaded
    def setDesiredRwdComps(self, listOfRwds):
        _ = DartEnv2Bot.getRwdsToUseBin(listOfRwds)
        self.skelHldrs[self.humanIdx].setRwdsToUse(DartEnv2Bot.rwdCompsUsed)

    #return array of pose values for 
    #pose : which pose : standing, seated, prone, goal, etc
    #skelType : 'Human', 'KimaNew', 'KimaOld'
    def getPoseVals(self, pose, skel, skelType):
        #only 1 pose for bot arm or full-body robot assistant
        if ('Bot_Arm' in skelType):#bot arm init pose - gets bot in vicinity for IK 
            return np.array([-0.38625,  1.06433,  0.10963,  0.73543, -0.42456,  0.     ]) 

        if ('Robot' in skelType):#humanoid robot always same pose
            basePose = np.zeros(skel.ndofs)
            #move to initial body position
            basePose[1] = 3.14
            basePose[3] = 0.98
            basePose[4] = .85
            #bend at waist
            basePose[21] = -.4
            #stretch out left arm at shoulder to align with hand
            basePose[26] = .25
            #stretch out left hand
            basePose[27] = -1.2        
            return basePose

        if ('goal' in pose):#set to goal pose - use for reverse curriculum learning
            basePose = skel.q  #root location 0,1,2; #root orientation 3,4,5      
            #move to be standing halfway between starting position and starting foot position of seated on ground
            #avg starting seated ana foot pos : [5.646086938202805605e-01,2.951752898446417459e-02,5.915744054878484259e-03]
            #avg starting standing ana foot pos : [5.249999999999719474e-02,3.049999999999997158e-02,-1.715673206281742580e-08]
            #q layout in kima root is z,y,x location (at least in old Kima skeleton)
            basePose[1]= 0.893
            basePose[2]= .5    #idx 2 is x location (at least in old Kima skeleton)
            #bend reach hand shoulder : bicep right (3) 25, 26, 27 : 25: outward from sides, 26 : front/back (negative forward);forearm right. 28
            basePose[-4:] = [ 0.1,-0.6,0.0,1.9]

            return basePose
            #no prone for human yet TODO         

        if ('seated' in pose):#sitting with knees bent - attempting to minimize initial collision with ground to minimize NAN actions
            if('Human' in skelType):
                #only supports 1 pose for humans currently
                return [
                    #root orientation 0,1,2, location 3,4,5
                    0,0,1.57, 0,0.06,0,            
                    #left thigh 6,7,8 : 6 is bend toward chest, 7 is twist along thigh axis, 8 is spread; left shin, left heel(2), left toe 9,10,11,12
                    0.87,0,0, -1.85,-0.6, 0, 0,
                    #right thigh ,13,14,15: 13 is bend toward chest, 14 is twist along thigh axis, 15 is spread ; right shin, right heel(2), right toe 16,17,18,19
                    0.87,0,0, -1.85,-0.6,0, 0,
                    #abdoment(2), spine 20,21,22; #head 23,24
                    0,-1.5,0, 0,0,
                    #scap left, bicep left(3) 25,26,27,28; forearm left, hand left 29,30
                    0,0.3,-0.6,0, 0.6,0,
                    #scap right, bicep right (3) 31,32,33,34 ; forearm right.,hand right 35,36
                    0,.6, 1.8, 0.0, 0.5, 0.0]
            if ('KimaNew' in skelType):
                return  [
                    #root loc 0,1,2; root orient 3, 4 ,5 : 
                    0,  0.09133,  0, 0, 0, 0,
                    #left thigh 6,7,8 : 6 is spread, 7 is bend toward chest(positive is back), 8 is twist along thigh axis, 8 is spread;left shin, left heel(2) 9,10,11 : 9 : bend at knee
                    0.01324, -2.3, 0.03374, 1.66,-.7,0,  
                    #right thigh ,12,13,14: 12 is spread, 13 is bend toward chest(positive is back), 14 is twist along thigh axis, 8 is spread;right shin, right heel(2)  15,16,17 : 15 : bend at knee
                    0.01324, -2.3, -0.03374, 1.66,-.7,0,
                    #abdoment(2), spine 18,19, 20 : 18:side to side, 19:front to back : 20 : twist
                    0.0, -0.2, 0.0, 
                    #bicep left(3) 21,22,23;forearm left,24         
                    -.1,-.5, 0.0, .5,  
                    # bicep right (3) 25, 26, 27 : 25: outward from sides, 26 : front/back (negative forward);forearm right. 28
                    0.1,-1.4,0.0,1.5]
            if ('KimaOld' in skelType):
                return  [
                    #root location 0,1,2 Z Y X ;root orientation 3,4,5 Z Y X :
                    0,  0.09133,  0,  1.53,  0,  0, 
                    #left thigh 6,7,8 : 6 is bend toward chest, 7 is twist along thigh axis, 8 is spread; left shin, left heel(2) 9,10,11
                    0.82,  0.007,  0.009, 1.91208, -0.44,  0.003,  
                    #right thigh ,12,13,14: 12 is bend toward chest, 13 is twist along thigh axis, 14 is spread;#right shin, right heel(2)  15,16,17
                    0.82,  0.007,  0.009,  1.9, -0.44,  0.003,
                    #abdoment(2), spine 18,19, 20
                    0.05288, -1.5,  0.05056,  
                    #bicep left(3) 21,22,23; forearm left,24            
                    -0.25, -0.5, -0.02, 0.59213,  
                    # bicep right (3) 25, 26, 27; forearm right. 28
                    0.16, -1.56151, 0.43283, 0.72403]
        if ('prone' in pose):
            basePose = skel.q  #root location 0,1,2; #root orientation 3,4,5             
            basePose[1]= 0.09133
            #no prone for human
            if ('KimaNew' in skelType):
                return basePose
            if ('KimaOld' in skelType):
                basePose[3] = 1.53#root location 0,1,2 ;root orientation 3,4,5 Z Y X : 
                return basePose
        
        print('dart_env_2bot::getPoseVals : Unknown skeleton type, unable to set initial pose')
        return skel.q
            
    #set assist objects' collide state
    def setAssistObjsCollidable(self, doCollide):
        self._setSkelCollidable(self.boxSkel,doCollide)
        self._setSkelCollidable(self.skelHldrs[self.botIdx].skel, doCollide)

    #set which skeleton we are currently using for training/RL
    #also sets action and observation variables necessary for env to work
    def setActiveRLskelhldr(self, activeSkelIDX):
        self.actRL_SKHndlrIDX = activeSkelIDX           
        self.activeRL_skelhldr = self.skelHldrs[self.actRL_SKHndlrIDX]
        #execute this to update the observation and action space dimensions if they change
        #either because active skeleton handler has changed, or because dimension of assist
        #force has changed
        self.updateObsActSpaces()
        
    def updateObsActSpaces(self):
        #these are used externally
        self.obs_dim = self.activeRL_skelhldr.obs_dim
        self.act_dim = self.activeRL_skelhldr.action_dim
        self.action_space = self.activeRL_skelhldr.action_space
        self.observation_space = self.activeRL_skelhldr.observation_space

    #FOR DEBUGGING PURPOSES - generate a random observation of the activeRL_skelhldr
    def getRandomObservation(self):
        return self.activeRL_skelhldr.getRandomObservation()

    #specify this in instanced environment - handle initialization of ANA's skelhandler - environment-specific settings
    def _initAnaKimaSkel(self, skel, skelType, actSclBase, basePose): 
        raise NotImplementedError()

    #added 3/16/18 to address problem with Kima skeleton def
    #need to set skeleton joint stiffness and damping, and body friction
    #maybe even coulomb friction for joints.
    #set for every joint except world joint - joint limits, stiffness, damping
    #print('Set Joints for Skel {}'.format(skel.name))
    def _fixJointVals(self, skel, kd=10.0, ks=0.0):
        for jidx in range(skel.njoints):
            j = skel.joint(jidx)
            #don't mess with free joints
            if ('Free' in str(j)):
                continue
            nDof = j.num_dofs()
            for didx in range(nDof):
                if j.has_position_limit(didx):
                    j.set_position_limit_enforced(True)             
                
                j.set_damping_coefficient(didx, kd)
                j.set_spring_stiffness(didx, ks)
        print('{}-type Skel is using ks val : {} and kd val : {}'.format(skel.name.lower(),ks, kd))
    
    #init either human or robot skeleton
    def _initLoadedSkel(self, skel, widx, skelType):    
        #base action scaling value
        actSclBase = 200   
        #eventually check for # dofs in skel to determine init
        stIDX = 0       
        if(skel.ndofs==9):#2D full body
            stIDX = 3
        elif(skel.ndofs==37):#3D full body
            stIDX = 6
        elif(skel.ndofs==29):#3d kima skel
            stIDX = 6

        #fixed robot arm has no fixed world dofs
        numActDofs = skel.ndofs - stIDX
        print('Skel : {} has {} dofs and {} act dofs'.format(skel.name,skel.ndofs, numActDofs))

        #base pose to use - only prone skeleton will deviate from this; AK's trained policy must use old skeleton
        basePose = self.getPoseVals('seated',skel, skelType)  
        #ANA is using human model
        if('Human' in skelType):
            self._fixJointVals(skel, kd=10.0, ks=0.0)
            eefOffset=np.array([0,-.1319,0])
            #skelH = humanSkelHolder(self, skel, widx, stIDX, fTipOffset=eefOffset, rtLocDofs=np.array([3,4,5]))  
            skelH = ANASkelHolder(self, skel, widx, stIDX, fTipOffset=eefOffset, rtLocDofs=np.array([3,4,5]))              
            #set reach hand elbow and both knees' action minimizing wt to be close to 0 -> minimially penalize elbow and knee action
            #idx should be actual dof == dof idx - self.stTauIdx
            skelH.buildActPenDofWtMat(np.array([3,10,27]))

            rchHandStrName = 'h_hand_right'
            bodyNamesAra = [['h_heel_left','h_toe_left'],['h_heel_right','h_toe_right'], ['h_hand_left','h_hand_right'], 'h_head']
            #list of tuples of dof idx's and multipliers for action scaling   
            actSclIdxMult = [([1,2,8,9], .75),
                            ([17,18], .5),#head
                            ([20,22,26,28], .5), #2 of each bicep
                            ([19,25], .75), # shoulders
                            ([4,5,11,12], .5), #ankles 
                            ([6,13, 23,24, 29,30], .25)]#feet and forearms, hands 
            kneeDofIDXs = [9,16]
        
        #ANA is using kima model
        elif('Kima' in skelType):
            eefOffset=np.array([0,-0.35375,0])
            skelH = ANASkelHolder(self, skel, widx, stIDX, fTipOffset=eefOffset, rtLocDofs=np.array([0,1,2])) 
            #set reach hand elbow and both knees' action minimizing wt to be close to 0 -> minimially penalize elbow and knee action
            #idx should be actual dof == dof idx - self.stTauIdx
            skelH.buildActPenDofWtMat(np.array([3,9,22]))
            #set goal pose
            skelH.setGoalPose(self.getPoseVals('goal', skel, skelType))

            #specify/override these values for kima in each instanced environment
            actSclBase, actSclIdxMult, basePose = self._initAnaKimaSkel(skel, skelType, actSclBase, basePose)

            rchHandStrName = 'r-lowerarm'
            bodyNamesAra = [['l-foot'],['r-foot'], ['l-lowerarm','r-lowerarm'], 'head']
            kneeDofIDXs = [9,15]
        elif('Bot_Arm' in skelType):
            self._fixJointVals(skel, kd=0.0, ks=0.0)
            eefOffset = np.array([.05,0,0])
            skelH = robotArmSkelHolder(self, skel, widx, stIDX, fTipOffset=eefOffset)  

            rchHandStrName = 'palm'
            #turn off collisions - NOTE This is for debugging only
            self._setSkelNoCollide(skel, skelType)
            bodyNamesAra = []
            actSclIdxMult = []
            kneeDofIDXs = []

        elif('Robot' in skelType):
            self._fixJointVals(skel, kd=10.0, ks=0.0)
            eefOffset=np.array([0,-.1319,0])
            skelH = robotSkelHolder(self, skel, widx,stIDX, fTipOffset=eefOffset)

            rchHandStrName = 'h_hand_right'
            bodyNamesAra = [['h_heel_left','h_toe_left'],['h_heel_right','h_toe_right'], ['h_hand_left','h_hand_right'], 'h_head']
            actSclIdxMult = [([1,2,8,9], .75),
                            ([17,18], .5),#head
                            ([20,22,26,28], .5), #2 of each bicep
                            ([19,25], .75), # shoulders
                            ([4,5,11,12], .5), #ankles 
                            ([6,13, 23,24, 29,30], .25)]#feet and forearms, hands 
            kneeDofIDXs = [9,16]
        else :
            print('DartEnv2Bot::_initLoadedSkel : Unknown skel type based on given skeltype {} for skel {}'.format(skelType,skel.name))
            return None  
        #give skeleton reference to its key in skels dictionary
        skelH.setInitialSkelParams(bodyNamesAra, kneeDofIDXs, rchHandStrName, skelType, basePose, actSclBase, actSclIdxMult)

        return skelH
    
    #implement this in environments to return appropriate, environment-dependent, assistance component of ANA's observation space
    #all environments supporting skelHldrs need to implement this method and the next one
    def getSkelAssistObs(self, skelHldr):
        raise NotImplementedError()

    def getSkelAssistObsNames(self, skelHldr):
        raise NotImplementedError()

    #set all body nodes in passed skel to allow or not allow collision
    def _setSkelCollidable(self, skel, canCollide):
        for body in skel.bodynodes:
            body.set_collidable(canCollide)

    def _setSkelNoCollide(self, skel, skelName):
        print('!!!!! NOTE : {} has no collisions enabled'.format(skelName))
        self._setSkelCollidable(skel, False)
        
    #get skeleton objects from list in dart_world  
    def _buildskelhldrs(self):
        #list of skels to render as default color
        self.skelsToRndr = []
        #list of tuples of skels to render (idx0) with color list (idx1)
        self.skelsToRndrWClr= []
        numSkels = self.dart_world.num_skeletons()
        self.hasHelperBot=False
        self.grabLink = None
        #apparently according to post on the subject in github from mxgrey, dart defaults to mu==100
        self.groundFric = 100.0
        
        for idx in range(numSkels):
            saveSkel = False
            skelType = ''
            skel = self.dart_world.skeletons[idx]
            self.skelsToRndr.append(skel)
            skelName = skel.name
            #this is humanoid to be helped up
            if ('getUpHumanoid' in skelName) or ('biped' in skelName) :
                skelType = 'Human'
                self.humanIdx = skelType
                #track getup humanoid in rendering
                #self.track_skeleton_id = idx
                saveSkel = True
            elif ('getUpHuman_Kima' in skelName) : #should also have "new" or "old" in name to denote which kima it uses
                #skel file name specifies which kima model is used : old which used old joint limits, otherwise newer kima (in 6.4 dart) which uses different joint config and limits
                isOldSkel = ('old' in skelName.lower() )
                skelType = 'KimaOld' if isOldSkel else 'KimaNew'
                print('Using Kima Skel File version : {}'.format(skelType))
                self.humanIdx = skelType
                #track getup humanoid in rendering
                #self.track_skeleton_id = idx
                saveSkel = True                
            elif 'sphere_skel' in skelName :
                self.grabLink = skel
                skelType = 'Sphere'
                #print('sphere_skel : {} | {}'.format(self.grabLink.com(),self.grabLink.q))
                self._setSkelNoCollide(skel, skelName)
                #explicitly set this to not simulate - override in child class if frwd simulating
                skel.set_mobile(False)
                #force to counteract gravity
                self.sphereForce = np.array([0,9.8,0]) * self.grabLink.mass()
            elif 'box_skel' in skelName : 
                skelType = 'Bot Arm Support Box'
                #set to non-collidable to speed up training
                self._setSkelNoCollide(skel, skelName)
                #turn off mobile so no frwd sim executed
                skel.set_mobile(False)
                #box skeleton
                self.boxSkel = skel
                self.track_skeleton_id = idx
            elif 'ground' in skelName : #seems to always be first, so this will be available for other skels to use
                skelType = 'Ground'
                self.groundSkel = skel
                #ground skeleton - set friction TODO verify
                #skel.friction = self.groundFric
                
            #Only have 1 type of helper bot
            elif 'helperBot' in skelName :
                skelType = 'Robot'
                self.botIdx = skelType
                saveSkel = True
                self.hasHelperBot=True
                #track helper bot
                self.track_skeleton_id = idx
                
            #if helper arm being built
            elif 'helperArm' in skelName :
                skelType = 'Bot_Arm'
                self.botIdx = skelType
                saveSkel = True
                self.hasHelperBot=True
                #track helper bot
                self.track_skeleton_id = idx
                
            #using pre-built helper arm
            elif 'KR5sixxR650WP_description' in skelName :
                skelType = 'KR5_Bot_Arm'
                self.botIdx = skelType
                saveSkel = True
                #track helper bot
                self.track_skeleton_id = idx
                self.hasHelperBot=True
                
            else :
                print('Skel Unknown : {}'.format(skelName))
                
            skel.friction = self.groundFric

            if saveSkel :
                self.skelHldrs[skelType]= self._initLoadedSkel(skel, idx, skelType)
                saveSkel = False
        
            print('Name : {} Type : {} Index : {} #dofs {} #nodes {} initial root loc : {}'.format(skelName, skelType,idx, skel.ndofs, skel.nbodies, skel.positions()[:6]))
        #give robot a ref to human skel handler if robot is there
        if (self.hasHelperBot):
            self.skelHldrs[self.botIdx].setHelpedSkelH(self.skelHldrs[self.humanIdx])
        print('dart_env_2bot::_buildskelhldrs finished\n\n')

    #return jacobians of either robot or human contact point
    #for debugging purposes to examine jacboian
    def getOptVars(self, useRobot=True):
        if(useRobot):
            skelhldr = self.skelHldrs[self.botIdx]
        else:
            skelhldr = self.skelHldrs[self.humanIdx]
        
        return skelhldr.getOptVars()
    
    #return a 2d array of lower/higher bounds of assist component
    def getAssistBnds(self):
        raise NotImplementedError()  

    # methods to override:
    # ----------------------------
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError()

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------
    def set_state(self, qpos, qvel):
        self.activeRL_skelhldr.set_state(qpos, qvel)

    def set_state_vector(self, state):
        self.activeRL_skelhldr.set_state_vector(state)        
        
    def state_vector(self):
        return self.activeRL_skelhldr.state_vector()

    #whether or not action proposals are boxed before they are used to derive tau in ana
    def setActionBox(self, clampA):
        #true or false
        self.skelHldrs[self.humanIdx].doClamp = clampA
        DartEnv2Bot.setClampAction(clampA)
        

    #bots is string of 'ANA', 'BOT', or 'ALL'
    def setStateSaving(self, bots, saveStates=True, fileName='RO_SASprimeData.csv'):
        if bots.lower() not in ['ana','bot','all'] : return
        #if not bot then save ana
        if 'bot' not in bots.lower():   self.skelHldrs[self.humanIdx].setStateSaving(saveStates, fileName)
        #if not ana then save bot
        if 'ana' not in bots.lower():   self.skelHldrs[self.botIdx].setStateSaving(saveStates, fileName)
            
    #write previous rollout data to file - necessary if reset has not been called at end of rollout
    #bots is string of 'ANA', 'BOT', or 'ALL'
    def saveROStates(self, bots):
        if bots.lower() not in ['ana','bot','all'] : return
        #if not bot then save ana
        if 'bot' not in bots.lower():    self.skelHldrs[self.humanIdx].checkRecSaveState()
        #if not ana then save bot
        if 'ana' not in bots.lower():    self.skelHldrs[self.botIdx].checkRecSaveState()

    #get ref to ana's or bot's skel holder
    def getANAHldr(self):
        return self.skelHldrs[self.humanIdx]
    def getBotHldrr(self):
        return self.skelHldrs[self.botIdx]

    @property
    def dt(self):
        return self.dart_world.dt * self.frame_skip
    
    # ##NOT CURRENTLY USED - stepping world and checking sim state implemented in instanced class
    # def checkWorldStep(self, fr):
    #     #check to see if sim is broken each frame, if so, return with broken flag, frame # when broken, and skel causing break
    #     for k,v in self.skelHldrs.items():
    #         brk, chkSt, reason = v.checkSimIsBroken()
    #         if(brk):  #means sim is broken, end fwd sim, return which skel holder was broken                
    #             return True, {'broken':True, 'frame':fr, 'skelhldr':v.name, 'reason':reason, 'skelState':chkSt}
    #     return False, {'broken':False, 'frame':fr, 'skelhldr':'None', 'reason':'FINE', 'skelState':chkSt}
    # # need to perform on all skeletons that are being simulated

    # def do_simulation(self, n_frames):
    #     if self.add_perturbation:
    #         self.checkPerturb()
    #         #apply their respective torques to all skeletons
    #         for fr in range(n_frames):
    #             for k,v in self.skelHldrs.items():
    #                 v.add_perturbation( self.perturbation_parameters[2], self.perturb_force)  
    #                 #tau is set in calling step routine
    #                 v.applyTau()                      
    #             self.dart_world.step()
    #             #check to see if sim is broken each frame, if so, return with broken flag, frame # when broken, and skel causing break
    #             chk,resDict = self.checkWorldStep(fr)
    #             #if chk break sim loop early and return dictionary of info
    #             if(chk):
    #                 return resDict        
    #     else :
    #         #apply their respective torques to all skeletons
    #         for fr in range(n_frames):
    #             for k,v in self.skelHldrs.items():
    #                 #tau is set in calling step routine
    #                 v.applyTau()                      
    #             self.dart_world.step()
    # #            #check to see if sim is broken each frame, if so, return with broken flag, frame # when broken, and skel causing break
    #             chk,resDict = self.checkWorldStep(fr)
    #             #if chk break sim loop early and return dictionary of info
    #             if(chk):
    #                 return resDict
    #     #return default resdict if nothing broke     
    #     return resDict                   
            
            
    def checkPerturb(self):
        if self.perturbation_duration == 0:
            self.perturb_force *= 0
            if np.random.random() < self.perturbation_parameters[0]:
                axis_rand = np.random.randint(0, 2, 1)[0]
                direction_rand = np.random.randint(0, 2, 1)[0] * 2 - 1
                self.perturb_force[axis_rand] = direction_rand * self.perturbation_parameters[1]

        else:
            self.perturbation_duration -= 1
          
            
    #return a dictionary arranged by skeleton, of 1 dictionary per body of contact info
    #the same contact might be referenced multiple times
    def getContactInfo(self):
        contacts = self.dart_world.collision_result.contacts
        #dictionary of skeleton name-keyed body node collisions
        cntInfoDict = {}
        for i in range(len(self.dart_world.skeletons)):
            cntInfoDict[self.dart_world.skeletons[i].name] = defaultdict(contactInfo)
        
        for contact in contacts:
            cntInfoDict[contact.bodynode1.skeleton.name][contact.bodynode1.name].addContact(contact,contact.bodynode1,contact.bodynode2, self.ctrTauPt)
            cntInfoDict[contact.bodynode2.skeleton.name][contact.bodynode2.name].addContact(contact,contact.bodynode2,contact.bodynode1, self.ctrTauPt)
            
        return cntInfoDict
        
    #get current state of sim world and all skeletons so it can be reset
    def saveSimState(self):
        self.tot_SimState = self.dart_world.states()
        self.stateIsSaved = True
        
    #restore sim state in world
    def restoreSimState(self):
        if(self.stateIsSaved):
            self.dart_world.set_states(self.tot_SimState)
            self.stateIsSaved = False
            self.tot_SimState = []

    ######################       
    #rendering stuff
    
    def close(self):
        if self.viewer is not None:
            self._get_viewer().close()
            self.viewer = None
        return
           
    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #_render deprecated in new gym code
    def render(self, mode='human', close=False):
        return self._render(mode)

    #called to reset world to start new rollout
    def reset(self):
        #reset the world
        self.dart_world.reset()
        #if perturbing anything, reset duration of perturbation here
        self.perturbation_duration = 0
        #total sim time for rollout - reset this value
        self.timeElapsed = 0
        ## of steps
        self.sim_steps = 0
        #reset indiv environment-specific quantities
        ob = self.reset_model()
        return ob
     
    def _render(self, mode='human', close=False):
        if not self.disableViewer:
            vwrWin = self._get_viewer()
            #keep focus on human skeleton com's x position
            vwrWin.scene.tb.trans[0] = -self.dart_world.skeletons[self.track_skeleton_id].com()[0]*1
            if close:
                if self.viewer is not None:
                    vwrWin.close()
                    self.viewer = None
                return

            if mode == 'rgb_array':
                data = vwrWin.getFrame()
                return data
            elif mode == 'human':
                vwrWin.runSingleStep()
    
    #build screen res based on size of monitor
    def getScreenRes(self):
        import subprocess
        output = subprocess.Popen('xrandr | grep "\*" | cut -d" " -f4',shell=True, stdout=subprocess.PIPE).communicate()[0]
        resolution = output.split()[0].split(b'x')
        return int(resolution[0]), int(resolution[1])
    
    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        #get screen res, set window to be 85% of screen res, or 1280/720, whichever is larger
        scr_width, scr_height = self.getScreenRes()
        w=scr_width*.85
        h=scr_height *.85
        self.width = w if w > 1280 else 1280
        self.height = h if h > 720 else 720
        win = StaticGLUTWindow(sim, title, int(self.width), int(self.height))
        win.scene.add_camera(Trackball(theta=-45.0, phi = 0.0, zoom=0.1), 'gym_camera')
        win.resizeGL(200,100)
        win.scene.set_camera(win.scene.num_cameras()-1)

        # to add speed,
        if self._obs_type == 'image':
            win.run(self.screen_width, self.screen_height, _show_window=self.visualize)
        else:
            win.run(_show_window=self.visualize)
            
        return win

    def _get_viewer(self):
        if self.viewer is None and not self.disableViewer:
            self.viewer = self.getViewer(self.dart_world, 'Get Up!')
            self.viewer_setup()
        return self.viewer
    
        #disable/enable viewer
    def setViewerDisabled(self, vwrDisabled):
         self.disableViewer = vwrDisabled

    #call to print to both screen and, possibly, console
    #dispStr : string to display
    #txtList : key to list to display
    #toConsole : if desired to print text to console as well
    def printToScr(self, dispStr, txtList, toConsole):
        if (not txtList in self.scrDispKeys):
            print('dart_env_2bot::printToScr : {} not found as key in ')

        self.scrDispText[txtList].append(dispStr)
        if(toConsole):
            print(dispStr)

    #set reward text to display on screen
    def setScrDispRwd(self, rwdTxtList):
        pass
   
    #called after display of data - pop results if timer is appropriate
    #calledFrom : 0 : sim step, 1 : disp step TODO
    def popScrDispTxt(self, calledFrom):
        idx = -1
        for key in self.scrDispKeys : 
            idx += 1
            #skip if doesn't decay upward or if no data to display
            if (not self.scrDispDecay[idx]) or (len(self.scrDispText[key]) == 0) :                
                continue
            self.txtDispTime[idx]+=1
            if(self.txtDispTime[idx] > self.maxDispTime):
                self.txtDispTime[idx] = 0
                self.scrDispText[key].pop(0)
        
    #display text on screen
    def dispScrText(self, ri):
        if (self.drawUIOverlayData):
            keyList = self.scrDispKeys
            idx = 0
            for key in keyList : 
                self._dispListOfText(ri, self.scrDispXYVals[idx], self.dispYOff[idx], self.scrDispClrList[idx], self.scrDispColTtl[idx], self.scrDispText[key], self.scrDispFont[idx])
                idx +=1

    #draw skeletons, with special colors being set for skeletons in list of tuples of skels and the render colors
    def renderSkels(self, ri):
        #only renders skels in this list
        for skel in self.skelsToRndr : 
            skel.render()
        #skels with specific colors - doesn't work with urdf-loaded skels apparently
        for skelTup in self.skelsToRndrWClr : 
            ri.pushState()
            clr=skelTup[1]
            ri.set_color(clr[0],clr[1],clr[2])
            skelTup[0].render_with_color(clr)
            ri.popState()

    def _dispListOfText(self, ri, stCoords, yOff, clr, hdrTxt, strList, font):
        if( len(strList) == 0) : return
        xs = stCoords[0]
        ys = stCoords[1]
        ri.pushState()
        ri.set_color(clr[0],clr[1],clr[2])
        ri.draw_text([xs, ys], hdrTxt, font=font) 
        ys+=  yOff
        for dispStr in strList : 
            ri.draw_text([xs, ys], dispStr, font=font)
            ys+= yOff
        ri.popState()
        
    #debug function to test functionality
    def dbgMakeListOfText(self):
        keyList = self.scrDispKeys
        for i in range(10):
            for key in keyList : 
                self.scrDispText[key].append("addding {} string to {} list".format(i, key)) 

        
    #dict of queues of text to display every time step
    def _initScrDispTxtDict(self):
        tmpDict = {}
        keyList = self.scrDispKeys
        for key in keyList : 
            tmpDict[key]=[]         
        self.scrDispText = tmpDict

    #set debug messages from traj component
    def setTrajDbgDisplay(self, dbgMsgList):
        self.scrDispText["trajDbgTxtList"] = []
        for msg in dbgMsgList : 
            self.scrDispText["trajDbgTxtList"].append(msg)

    #vals set in reward calculation
        # dbgStructs['rwdTyps'].append(rwdType)
        # dbgStructs['rwdComps'].append(rwdComp)
        # dbgStructs['successDone'].append(succCond)
        # dbgStructs['failDone'].append(failCond)
        # dbgStructs['dbgStrList'][0] += '\t{}\n'.format(strVal)
        # dbgStructs['dbgStrList_csv'][0] += '{},'.format(csvStrVal)
        # dbgStructs['dbgStrList'].append(strVal)  
    #set the reward portion of the on-screen display to reflect current reward values (rwdComps, done, dbgStructs, debug)
    def setRewardDisplay(self, rwdComps, done, dbgStructs, doDebug):
        self.scrDispText["rwdTextList"] = []
        if (doDebug):
            dbgDataList = dbgStructs['dbgStrList']
            self.scrDispText["rwdTextList"].append('Step : {} : AssistComponent {} : Reward : {:.7f} Done : {}'.format(self.sim_steps,self.getSkelAssistObs(self.getANAHldr()), rwdComps['reward'], done))
            for i in range(1, len(dbgDataList)):
                datStr = dbgDataList[i].replace("|", "     ")
                self.scrDispText["rwdTextList"].append(datStr)

        else :#clear out disp list
            pass


    #set constraint debug display text
    def setConstraintDisplay(self, dbgMsgList, doDebug):
        self.scrDispText["cnstrntTextList"] = []
        if(doDebug):
            for msg in dbgMsgList : 
                self.scrDispText["cnstrntTextList"].append(msg)

    def initAllScrDispData(self):
        #dict of lists of text to display every time step
        self.scrDispKeys = ['rwdTextList', 'cnstrntTextList', 'simTxtList' , 'trajDbgTxtList', 'errorTxtList']
        self.scrDispColTtl = ['Reward Info', 'Constraint Data', 'Simulation Values', 'Trajectory Debugging Text', 'Error/Issues Text']
        #whether txt decays (reward shouldn't)
        self.scrDispDecay = [False, False, True, False, True]
        #x,y start values
        dispOffX = (2 *self.width)//3
        disp1ThrdY = self.height//3
        dispHalfScrY = self.height//2#(2 *self.height)//3
        disp2ThrdY = 2 * disp1ThrdY 
        #locations and fonts to use for various display text
        self.scrDispXYVals = [(10, 20), (10, disp1ThrdY), (dispOffX, 20), (10, dispHalfScrY), (dispOffX,disp2ThrdY)]
        self.scrDispFont = [GLUT.GLUT_BITMAP_HELVETICA_12, GLUT.GLUT_BITMAP_HELVETICA_12, GLUT.GLUT_BITMAP_HELVETICA_18, GLUT.GLUT_BITMAP_HELVETICA_12, GLUT.GLUT_BITMAP_HELVETICA_18]
        self.dispYOff = [15,15,20,15,20]
        #frame counter for display of messages - used to decay display
        self.txtDispTime = [0,0,0,0,0]
        # number of frames to display messages before decaying display
        self.maxDispTime = 30
        #list of colors
        self.scrDispClrList = [(0,0,0),(0,1,1),(1,0,1),(0,1,1),(1,.5,.5)]
        self._initScrDispTxtDict()
        

    #build the world - put here to enable easy custom world class
    def buildWorld(self, dt, skelFullPath):
        from myDartEnvWorld import myWorld
        self.dart_world = myWorld(dt,skelFullPath)#self.dart_world = pydart.World(dt, skelFullPath)

    #convert to a full path to passed model file location and verify the file exists
    def setFullPath(self, model_path):
        if model_path.startswith("/"):
            fullPath = model_path
        else:
            fullPath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullPath):
            raise IOError("File {} does not exist".format(fullPath))
        return fullPath

    #load all seletons and sim world
    def loadAllSkels(self, model_paths, dt):
        if isinstance(model_paths, str):
            model_paths = [model_paths]
        #keep around so we can inspect trained policy environments, to see what skel files were used for training
        self.mode_paths = model_paths    
        
        # convert everything to fullpath
        #full_paths holds all additional skeletons (without .skel extension)
        full_paths = []
        #skelFullPath holds location of .skel files
        skelFullPath = None
        for model_path in model_paths:
            fullPath = self.setFullPath(model_path)
            if('.skel' in fullPath):
                if(skelFullPath is None):
                    skelFullPath = fullPath
                else:
                    print('Issue : multiple .skel files : {}\n\tand {}\n\tloadAllSkels only currently handles a single skel.  Subsequent skels are ignored'.format(fullPath,skelFullPath))
            else:
                full_paths.append(fullPath) 

        # for fp in full_paths : 
        #     print('path : {}'.format(fp))
               
        self.buildWorld(dt, skelFullPath)
        # if(skelFullPath is not None):
        #     #a .skel file exists
        #     self.dart_world = pydart.World(dt, skelFullPath)
        # else :
        #     self.dart_world = pydart.World(dt)
        for fullpath in full_paths:
            self.dart_world.add_skeleton(fullpath)

        # try:
        #     self.dart_world.set_collision_detector(3)
        # except Exception as e:
        #     print('Does not have ODE collision detector, reverted to bullet collision detector')
        #     self.dart_world.set_collision_detector(2)

    #set perturbation if used
    #params : probability, magnitude, bodyid
    #dur : duration of perturbation in frames
    #frc : perturbation force
    def setPerturbation(self, isPerturb, params=[0.05, 5, 2], dur=40, frc=np.array([0,0,0])):
        # random perturbation
        self.add_perturbation = isPerturb
        self.perturbation_parameters = params 
        self.perturbation_duration = dur
        self.perturb_force = frc

    #setup initial datetime string for file saving
    def initDateTime(self):        
        import datetime
        #save datetime for timestamping images
        datetimeStr = str(datetime.datetime.today())
        self.datetimeStr =  datetimeStr.replace('.','-').replace(' ','-').replace(':','-')

    ###########################
    #externally called functions
    #turn on/off ability to pause for input
    def setAllowPauseForInput(self, val):
        self.allowPauseForInput = val

    #pause for a return, displaying the passed dispStr, if allowPause enabled
    #valsToDips is list of values to display when paused
    def pauseForInput(self, dispStr, waitOnInput=True, valsToDisp=[]):
        if (not self.allowPauseForInput):return
        print("Called from {} ".format(dispStr))
        numVals = len(valsToDisp)
        for i in range(numVals):
            print("\t{} : {}".format(i, valsToDisp[i]))
        print("Paused; <hit enter to continue>", end=" ")
        if(waitOnInput):
            input()
    
    #return a string name to use as a directory name for a simulation run
    def getRunDirName(self):
        #build name of image from ANA, helper bot, and rwd func
        imgDirName = self.getImgName()
        nameStr = '_'.join([imgDirName , self.datetimeStr, '{0:02d}'.format(self.saveIter)])
        self.saveIter +=1           #increment so future sequences have different directory
        nameStr = nameStr.replace(' ','_')
        return nameStr
        
    #return a name for the image
    def getImgName(self):
        rwdFuncStr = DartEnv2Bot.getRwdsToUseBin()
        res = 'ANA_{}_BOT_{}_RWD_{}'.format(self.skelHldrs[self.humanIdx].name,self.skelHldrs[self.botIdx].name, rwdFuncStr)   
        return res
