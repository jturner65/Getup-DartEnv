#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:09:33 2018

@author: john
"""
#all skeleton holders used by dart_env_2bot

import os
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np

import nlopt
from math import sqrt
from gym import error, spaces
#from gym.spaces import *

from abc import ABC, abstractmethod

from collections import defaultdict#, OrderedDict
from sortedcontainers import SortedDict

try:
    import pydart2 as pydart
    from pydart2.gui.trackball import Trackball
    pydart.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))

#base convenience class holding relevant info and functions for a skeleton
class skelHolder(ABC):   
    #env is owning environment
    #skel is ref to skeleton
    #widx is index in world skel array
    #stIdx is starting index in force/torque array for tau calc - past root dofs that get no tau
    #fTipOffset is distance in body local coords from com of reach body for constraint/grab location
    def __init__(self, env, skel, widx, stIdx, fTipOffset):
        print("making skel : {}".format(skel.name))
        self.name=''
        self.shortName=''
        #ref to owning environment
        self.env = env
        #ref to skel object
        self.skel = skel
        #index in world skeleton array
        self.worldIdx = widx
        #start index for action application in tau array -> bypasses
        #idx of first dof that gets tau (after root/world related dofs that do not get tau applied)
        self.stTauIdx = stIdx
        #preloaded initial state to return to at reset, otherwise random variation of initstate is used
        self.initQPos = None
        self.initQVel = None
        #idxs of dofs corresponding to world location of root - replace with com in observation - make sure not orientation of root
        self.rootLocDofs = np.array([3,4,5])
        #number of dofs - get rid of length calc every time needed
        self.ndofs = self.skel.ndofs
        #number of actuated dofs : number of dofs - world dofs
        self.numActDofs = self.ndofs - self.stTauIdx
        #init action vector
        self.a = np.zeros(self.numActDofs)
        #whether or not setting tau from proposed action will clamp action or not to control box
        #normalized environment does this before action proposal is even seen by env.  not using normed env might require this
        #so that taus don't explode - non-normed env will determine costs based on non-clipped action proposal.
        self.clampA = False

        #bound on allowable qDot - for ana training and for general sim stability
        self.qDotBnd = 20
        #per frame timestep - robot needs to conduct every sim step
        self.perSimStep_dt=self.env.dart_world.dt
        #per step timestep - including frameskip == frameskip * world.dt
        self.perFrameSkip_dt=self.env.dt
        
        #ground friction from environment
        self.groundFric = self.env.groundFric
        #gravity of this world
        self.grav = self.env.dart_world.gravity()
        #print('gravity is  : {} '.format(grav))
        #precalc mg for force calculations - magnitude of grav * mass - precalc scalar value to use to derive frc val from frc mult
        self.mg = np.linalg.norm(self.grav) * self.skel.mass()
        
        #to monitor torques seen : minimum and maximum seen torques saved in arrays
        self.monitorTorques = False
        self.monTrqDict = {}
        self._resetMinMaxMonitors(self.monTrqDict, self.ndofs)

        #state flags
        #use the mean state only if set to false, otherwise randomize initial state
        self.randomizeInitState = True
        #use this to show that we are holding preset initial states - only used if randomizeInitState is false
        self.loadedInitStateSet = False
        #process actions and forward simulate (should match skel.is_mobile) - only use in prestep and poststep functions, if at all
        self.isFrwrdSim = True
        #whether to use linear or world jacobian for endeffector
        self.useLinJacob = False
        #Monitor generated force at pull point/hand contact
        self.monitorGenForce = False
        #whether states have been saved for restore after frwrd step
        self.saveStateForFwdBkwdStep = False



        #display debug information for this skel holder
        self.debug = True

        #set baseline desired external forces and force mults
        self.desExtFrcVal = np.array([0,0,0])
        self.desExtFrcDir = np.array([0,0,0])
   
        #initial torques - set to tau to be 0
        self.dbgResetTau()
                
        #+/- perturb amount of intial pose and vel for random start state
        self.poseDel = .005
        #hand reaching to other agent - name and ref to body
        self.reach_hand = ''
        self.reachBody = None
        #initial constraint location - pose is built/modified off this for assistant robot, and this is derived from human's eff location
        self.initEffPosInWorld = None
        #this is location on reach_hand in local coords of contact with constraint body
        #i.e., this is constraint location on reach_hand in local coords 
        self.reachBodyOffset = fTipOffset
        #list of body node names for feet and hand in biped
        self.feetBodyNames = list()        
        self.handBodyNames = list()
        # # of sim steps we've run (# of times tau is applied)
        self.numTauApplied = 0

        #initialize structures used to check states and build state distributions
        self.initBestStateChecks()
        #max iterations of IK - doesn't use that many, make sure there's enough to get to target
        self.IKMaxIters = 20
        #min alpha for adaptive TS
        self.minIKAlpha = .01
        #whether or not to debug IK calc
        self.debug_IK = False  
        #for kneeless agents
        self.kneeDOFIdxs= []
        self.kneeDOFActIdxs=[]



    #if using SPD to provide control to this holder's skel, call this function on init
    def buildSPDMats(self, SPDGain):
        ndofs = self.ndofs
        dt = self.perSimStep_dt
        self.Kp_const = SPDGain
        self.Kd_const = 1.00001*dt * self.Kp_const  #making slightly higher than 1*dt helps with oscillations
        self.Kp_SPD = self.Kp_const * np.identity(ndofs)
        self.Kd_SPD = self.Kd_const * np.identity(ndofs)
        self.zeroKD = np.zeros(ndofs)
        self.Kd_ts_SPD = dt * self.Kd_SPD
#        print("self.Kp_const:{}".format(self.Kp_const))
#        print("self.Kd_const:{}".format(self.Kd_const))
#        print("self.Kp_SPD:{}\n".format(self.Kp_SPD))
#        print("self.Kd_SPD:{}\n".format(self.Kd_SPD))
#        print("self.Kd_ts_SPD:{}\n".format(self.Kd_ts_SPD))
#        print("self.zeroKD:{}".format(self.zeroKD))
        
    #init all variables used to save state-action-state' values to file
    def initSaveStateRec(self):
        #record all obs/action/res_state info as list of vectors, to save to file - initially this data is not recorded.  recStateActionRes needs to be true for this to be recorded
        self.recStateActionRes = False
        self.SASTrajNum = 0
        self.listOfSASVec = []
        #build dictionary holding idxs of specific SAS' vector components 
        self.SASVecFmt = {}
        #format of SAS vec - idxs of components
        endQdot = 2*self.ndofs
        assistSize = self.obs_dim - endQdot
        
        self.SASVecFmt = {}
        self.SASVecFmt['q0']=(0,self.ndofs)
        self.SASVecFmt['qdot0']=(self.ndofs, endQdot)
        actStart = endQdot + assistSize
        self.SASVecFmt['assist0']=(endQdot, actStart)
        
        actEnd = actStart + self.numActDofs
        self.SASVecFmt['action']= (actStart, actEnd)
        #includes root 6 dofs
        #self.SASVecFmt['kneeDofIDXs']=self.kneeDOFIdxs
        #action idxs
        self.SASVecFmt['kneeDOFActIdxs']=self.kneeDOFActIdxs
        
        q1end = actEnd + self.ndofs
        q1dotend = q1end + endQdot
        self.SASVecFmt['q1']=(actEnd,q1end)
        self.SASVecFmt['qdot1']=(q1end, q1dotend)
        actStart2 = q1dotend + assistSize
        self.SASVecFmt['assist1']=(q1dotend, actStart2)
        #add any ANA or helperB
        self._initSaveStateRecPriv(actStart2)
    
    #this will initialize sturctures to hold best states seen in rollout
    def initBestStateChecks(self):
        #bestStateMon : 
        #0 : donot monitor best states
        #1 : whether to check for best state after each reward - this is to keep around the best state from the previous rollout to use as next rollout's mean state, or 
        #2 : whether to remember all states and build a CDF of them - set true in ANA if wanting to build rwd-weighted distribution of states seen
        self.bestStateMon = 0 #set to 1 or 2 in ana's ctor

        #initialize to bogus vals
        self.bestStRwd = -100000000000
        self.bestState = None
        
        #self.buildStateCDF = False#set this true in ana if wanted   
        self.numStatesToPoll = 25 # max number of best states to poll for starting state proposals        
        self.stateRwdSorted = SortedDict()#holds all states with their rewards as keys - hopefully no collisions with floats as keys     
        

    #moved from dart_env_2bot - this sets up the action and (initial) observation spaces for the skeleton.  
    # NOTE : the observation space does not initially include any external forces, and can be modified through external calls to setObsDim directly
    # this function should be called once, only upon initial construction of skel handler
    #bodyNamesAra : array of body names, list of lists for foot, hand, head
    #kneeDOFIdxs : dof idxs of knee joints.  for possible reward function - idxs in skel, not in action vec
    #rchHandStrName : name of reach hand - can be changed if necessary, but should be necessary
    #skelType : key  in env dictionary holding all skelHolders
    #basePose : core pose to start - uses this as initPose 
    #actSclBase : base action scale value 
    #actSclIdxMlt : list of tuples of list of dof indices (tup[0]) and multipliers (tup[1]) for action scaling, if used.
    def setInitialSkelParams(self, bodyNamesAra, kneeDOFIdxs, rchHandStrName, skelType, basePose, actSclBase, actSclIdxMlt):
        self.skelType = skelType
        self.kneeDOFIdxs = kneeDOFIdxs
        self.kneeDOFActIdxs = [x-6 for x in kneeDOFIdxs]
        #clips control to be within -1 and 1
        #min is idx 0, max is idx 1
        self.control_bounds = (np.array([[-1.0]*self.numActDofs,[1.0]*self.numActDofs])) 
        self.action_dim = self.numActDofs
        self.action_space = spaces.Box(self.control_bounds[0], self.control_bounds[1])
        #set base pose values - replacing pose settings in _makeInitPoseIndiv
        self.initPose = basePose
        
        #set base action scale value
        self.actionScaleBaseVal = 1.0*actSclBase
        #scaling value of action multiplier (action is from policy and in the range of approx +/- 2)
        action_scaleBase = np.array([self.actionScaleBaseVal]*self.numActDofs)
        #individual action scaling for different bot configurations - list of tuples holding idxs in list and multiplier
        for tup in actSclIdxMlt:
            action_scaleBase[tup[0]] *= tup[1]

        self.action_scale = action_scaleBase
        print('action scale : {}'.format(self.action_scale))

        #set reach hand body
        self.reach_hand = rchHandStrName
        self.reachBody = self.skel.body(self.reach_hand)

        #set initial observation dimension - NOTE : this does not include any external forces or observations
        self.setObsDim(2*self.ndofs)
        #if different body type names are specified in array...
        if (len(bodyNamesAra) > 0):
            self._setFootHandBodyNames(bodyNamesAra[0], bodyNamesAra[1], bodyNamesAra[2], bodyNamesAra[3]) 
        else :#not relevant for KR5 arm robot
            print("---- No foot/hand/head body names specified, no self.StandCOMHeight derived ----")

    #called initially before any pose modification is done - pose of skel by here is pose specified in skel/urdf file
    def _setFootHandBodyNames(self,lf_bdyNames, rf_bdyNames, h_bodyNames, headBodyName):
        self.feetBodyNames = lf_bdyNames + rf_bdyNames
        self.leftFootBodyNames = lf_bdyNames[:]
        self.rightFootBodyNames = rf_bdyNames[:]
        self.handBodyNames = h_bodyNames[:]
        self.headBodyName = headBodyName
        for ft in self.feetBodyNames:
            self.skel.body(ft).set_friction_coeff(self.groundFric)
        print('skelHolder::setFootHandBodyNames : set values for initial height above avg foot location  - ASSUMES CHARACTER IS UPRIGHT IN SKEL FILE. Must be performed before desired pose is set')   
        #specific to instancing class - only RL-involved skel holders should have code in this     
        self._setInitRWDValsPriv()     
    
    #whether or not this skeleton is dynamically simulated or not - setting to false still allows for IK
    def setSkelMobile(self, val):
        self.skel.set_mobile(val)
        #frwrdSim is boolean whether mobile/simulated or not
        self.isFrwrdSim = val
           
    #initialize constraint/target locations and references to constraint bodies
    def initConstraintLocs(self,  cPosInWorld, cBody):
        #this is initial end effector position -desired- in world coordinates, built relative to human pose fingertip - if helper bot then bot must IK to this pos
        self.initEffPosInWorld = np.copy(cPosInWorld)
        #initializing current location for finite diff velocity calc
        self.trackBodyCurPos = np.copy(self.initEffPosInWorld)#self.cnstrntBody.com()
        #body to be constrained to or track
        self.cnstrntBody = cBody
        self.cnstrntBodyMG = cBody.mass() * self.grav
        #constraint location in ball local coords - convert to world for target location for inv dyn
        self.cnstrntOnBallLoc = self.cnstrntBody.to_local(x=self.initEffPosInWorld) 
    
    #set initial constraint location in world - called every reset and before addBallConstraint
    #only called 1 time, when constraints being first built
    def addBallConstraint(self):
#        if(setConnect ):#set the constraint to connect these bodies
        #build constraint
        constraint = pydart.constraints.BallJointConstraint(self.cnstrntBody, self.reachBody, self.initEffPosInWorld)
        #add to world
        constraint.add_to_world(self.env.dart_world)
            #print('{} is currently constrainted to ball at world location : {} corresponding to local reach body loc : {}'.format(self.skel.name, self.initEffPosInWorld,self.reachBody.to_local( self.initEffPosInWorld)))
#        else : #treat as tracking body 
            #print('{} not currently holding constraint ball!!!'.format(self.skel.name))
            #initializing current location for finite diff velocity calc
#            pass
        
    #return the world position of the constraint on the constraint body
    def getWorldPosCnstrntOnCBody(self):
        return self.cnstrntBody.to_world(x=self.cnstrntOnBallLoc)
        
    #return vector from constraint body to reach body, in world coords
    def getWorldPosCnstrntToFinger(self):
        return self.cnstrntBody.to_world(x=self.cnstrntOnBallLoc) - self.reachBody.to_world(x=self.reachBodyOffset)

    #set the desired external force for this skel
    #(either the force applied to the human, or the force being applied by the robot)
    #set reciprocal is only used for debugging/displaying efficacy of force generation method for robot
    def setDesiredExtAssist(self, desFrcTrqVal, desExtFrcVal_dir, setReciprocal):
        #print('{} got force set to {}'.format(self.skel.name, desFrcTrqVal))
        #self.lenFrcVec = len(desFrcTrqVal)
        self.desExtFrcVal = np.copy(desFrcTrqVal)
        self.desExtFrcDir =  np.copy(desExtFrcVal_dir)
        #if true then apply reciprocal force to reach body
        self.setReciprocal = setReciprocal
#        if(setReciprocal):
#            #counteracting force for debugging - remove when connected
#            self.reachBody.set_ext_force(-1 * self.desExtFrcVal, _offset=self.reachBodyOffset)
    
    #set observation dimension - allow for large observation dimension to handle if assist force instead of assist frc mult is used
    def setObsDim(self, obsDim):
        self.obs_dim = obsDim
        #high = np.inf*np.ones(self.obs_dim)
        high = 1000*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        #initialize components used to describe saving state-action-state' data to file - predicated on observation dimension
        self.initSaveStateRec()
        
    #set what this skeleton's init pose should be
    #only call this whenever initial base pose _actually chagnes_ (like upon loading)
    def setStartingPose(self):
        print('Skelhandler : {} calling setToInitPose'.format(self.name))
        #initPose is base initial pose that gets varied upon reset if randomized - base pose is treated as constant 
        #This has been replaced by initial pose being set in dart_env_2bot, to eventually be replaced by config file
        #self.initPose = self._makeInitPoseIndiv()
        self.setToInitPose()
        
    #reset this skeleton to its initial base pose - uses preset self.initPose
    def setToInitPose(self):
        #priv method is so that helper bots can IK to appropriate location based on where ANA hand is
        self._setToInitPosePriv()
        self.skel.set_positions(self.initPose)
        #set skel velocities
        self.skel.set_velocities(np.zeros(self.skel.dq.shape))

        #if monitoring the force/torques at pull application point, reset minmax arrays 
        if(self.monitorGenForce):
            self.totGenFrc = list()
            self.minMaxFrcDict = {}
            self._resetMinMaxMonitors(self.minMaxFrcDict, self.useForce.shape)
       
        self._postPoseInit()

    #will reset passed dictionary of min and max values to the given shape of initial sentinel values
    def _resetMinMaxMonitors(self, minMaxDict, sizeShape):
        minMaxDict['min'] = np.ones(sizeShape)*1000000000
        minMaxDict['max'] = np.ones(sizeShape)*-1000000000
               
    #set initial state externally - call before reset_model is called by training process
    #Does not override initPose
    def setNewInitState(self, _qpos, _qvel):       
        #set to false to use specified states
        self.randomizeInitState = False
        self.initQPos = np.asarray(_qpos, dtype=np.float64)
        self.initQVel = np.asarray(_qvel, dtype=np.float64)
        self.loadedInitStateSet = True
                
    #set this skeleton's state
    def state_vector(self):        
        return np.concatenate([
            self.skel.q,
            self.skel.dq
        ])
    #sets skeleton state to be passed position and velocity
    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.ndofs,) and qvel.shape == (self.ndofs,)
        self.skel.set_positions(qpos)
        self.skel.set_velocities(qvel)

    #sets skeleton state to be passed state vector, split in half (for external use)
    def set_state_vector(self, state):
        numVals = int(len(state)/2.0)
        self.set_state(state[0:numVals], state[numVals:])

    #called in do_simulate if perturbation is true - adds external force to passed body nodes
    def add_perturbation(self, nodes, frc):
        self.skel.bodynodes[nodes].add_ext_force(frc)        
                 
    #build tau from a, using control clamping, for skel, starting at stIdx and action scale for skeleton 
    #a is specified from RL policy, and is clamped to be within control_bounds
    def setClampedTau(self, a, doClamp =False):
        self.a = np.copy(a)
        #print('pre clamped cntl : {}'.format(a))
        #in self.control_bounds : idx 0 holds mins, idx 1 holds maxes.
        #has been clipped to control bounds already by normalized environment
        if doClamp :
            clamped_control = np.clip(a, self.control_bounds[0],self.control_bounds[1])
            #print('a is clamped')
        else :
            clamped_control = a
            #print('a is not clamped')
        #print('clamped cntl : {}'.format(clamped_control))
        tau = np.zeros(self.ndofs)
        tau[self.stTauIdx:] = clamped_control * self.action_scale 
        #print('{} \n\ttau : {}'.format(self.name,tau))
        return tau  
       
    #send torques to skeleton
    def applyTau(self):
        if(self.monitorTorques):
            self._checkiMinMaxVals(self.tau, self.monTrqDict)               
        #apply torques every step since they are cleared after world steps
        #print('{} : Applying Tau : {}'.format(self.name,self.tau))
        self.skel.set_forces(self.tau)      
        #record # of tau applications
        self.numTauApplied += 1     
        #any private per-step torque application functionality - this is where the external force is applied to human skeleton during RL training to simulate robot assistant
        self.applyTauPriv()        
        
    #return limits of Obs values (q, qdot, force)
    def getObsLimits(self):
        jtlims = {}
        jtlims['lowQLims'] = np.zeros(self.ndofs)
        jtlims['highQLims'] = np.zeros(self.ndofs)
        jtlims['lowDQLims'] = np.zeros(self.ndofs)
        jtlims['highDQLims'] = np.zeros(self.ndofs)
        dofs = self.skel.dofs
        for i in range(len(dofs)):
            dof = dofs[i]
            if dof.has_position_limit():
                jtlims['lowQLims'][i]=dof.position_lower_limit()
                jtlims['highQLims'][i]=dof.position_upper_limit()
            else:
                jtlims['lowQLims'][i]=-3.14
                jtlims['highQLims'][i]=3.14
            vLL = dof.velocity_lower_limit()
            if np.isfinite(vLL):
                jtlims['lowDQLims'][i] = vLL
            else :
                jtlims['lowDQLims'][i] = -self.qDotBnd
                
            vUL = dof.velocity_upper_limit()
            if np.isfinite(vUL):
                jtlims['highDQLims'][i] = vUL
            else :               
                jtlims['highDQLims'][i] = self.qDotBnd
        #limits of location are arbitrary except for y, must be positive and less than ~2 - observation of rootLocDOFS is actually COM values, so won't be below ground
        jtlims['lowQLims'][self.rootLocDofs[1]] = 0
        jtlims['highQLims'][self.rootLocDofs[1]] = 2  
        lowLims = [jtlims['lowQLims'],jtlims['lowDQLims']]
        hiLims = [jtlims['highQLims'],jtlims['highDQLims']]
        
        #might not have frc bounds if not assist force
        if (hasattr(self.env, 'frcBnds') and (self.env.frcBnds is not None)):
            frcBnds = self.env.frcBnds        
            lowLims.append(frcBnds[0])
            hiLims.append(frcBnds[1])
            
        jtlims['obs_lowBnds']=np.concatenate(lowLims)
        jtlims['obs_highBnds']=np.concatenate(hiLims)
        return jtlims
    
    def _concat3Aras(self, a, b,c):
        res1 = np.concatenate([a,b])
        res  = np.concatenate([res1,c])
        return res    
    
    #return best state from list of saved states, based on probability of state being set as state's reward
    def pollBestStates(self, rand):
        #only poll specified # of best states
        numStatesToPoll=(self.numStatesToPoll+1)
        klist = list(self.stateRwdSorted.keys())
        if(len(self.stateRwdSorted) > numStatesToPoll):
            #in increasing order
            minRwdAdded = klist[-numStatesToPoll]
            newStateDict = {k:self.stateRwdSorted[k] for k in klist[-self.numStatesToPoll:]}
            self.stateRwdSorted = SortedDict(newStateDict)
        else :
            minRwdAdded = klist[0]
        
        if rand :        
            #build CDF and poll it
            stateRwdCDFSorted = SortedDict()
            cdfKey = 0
            for k,v in self.stateRwdSorted.items():
                stateRwdCDFSorted[cdfKey]=v
                newk = k - minRwdAdded #want to normalize to rewards starting at 0
                cdfKey += newk
            
            #print('stateRwdCDFSorted data : len : {}'.format(len(stateRwdCDFSorted)))
            #for k,v in stateRwdCDFSorted.items():
            #    print('K:V :: {} : {}'.format(k,v))
            #draw random value of sum of all rewards == self.lastKeyAdded
            rndVal = self.env.np_random.uniform(low=0, high=cdfKey)
            keysList = list(stateRwdCDFSorted.keys())
            #use maxVal key less than or equal to rndVal 
            #print('rand val : {} cdfKey : {}'.format(rndVal,cdfKey))
            key = max([k for k in keysList if k <= rndVal])
            state = stateRwdCDFSorted[key]
        else :
            #max key is max reward state
            key = klist[-1]
            state = self.stateRwdSorted[key]
        return state
    
    def addStateToStateList(self, rwd, state):
        self.stateRwdSorted[rwd]=np.copy(state)
    
    #return a random initial state for this skeleton
    def getRandomInitState(self, poseDel=None):
        skel=self.skel
        rnd = self.env.np_random
        if(poseDel is None):
            poseDel=self.poseDel
        ndofs = self.ndofs
        lPoseDel = -1*poseDel
        #checking if the len of stateRwdSorted dict is > 0 to see if any best states are being retained - bestStateMon == 0 means we are not doing this
        #this is for using different state for init state
        if (self.bestStateMon > 0) and (len(self.stateRwdSorted) > 0):
            #if 1 then only best state, if 2 then poll cdf of all states with prob based on rwd
            self.set_state_vector(self.pollBestStates(self.bestStateMon > 1))

        else :
            #set walker to be laying on ground
            self.setToInitPose()
            #clear velocity!!!!
            skel.set_velocities(np.zeros(ndofs))
            
        #perturb init state and statedot
        qpos = skel.q + rnd.uniform(low= lPoseDel, high=poseDel, size=ndofs)
        qvel = skel.dq + rnd.uniform(low= lPoseDel, high=poseDel, size=ndofs)
        return qpos, qvel

    #returns a random observation, based on the full range of possible q and qdot, governed by joint limits and joint vel limits, if any exist
    def getRandomObservation(self):
        rnd = self.env.np_random
        #get all known joint limits
        jtlims = self.getObsLimits()
        #print('{}'.format(jtlims))
        rndQ = rnd.uniform(low=jtlims['lowQLims'],high=jtlims['highQLims'])
#        if not (np.isfinite(jtlims['lowDQLims']).all()) or not (np.isfinite(jtlims['highDQLims']).all()) :
#            rndQDot = self.env.np_random.uniform(low=-self.qDotBnd, high= self.qDotBnd, size=self.ndofs)
#        else :
        rndQDot = rnd.uniform(low=jtlims['lowDQLims'],high=jtlims['highDQLims'])
        #build observation out of these state values and return this observation
        return self.getObsFromState(rndQ, rndQDot)

    #calculate orientation along certain orientation axes
    def procOrient(self, orientAxes):
        oVec = np.array(orientAxes)
        oVec_W = self.skel.bodynodes[0].to_world(oVec) - self.skel.bodynodes[0].to_world(np.array([0, 0, 0]))
        norm = np.linalg.norm(oVec_W)
        if(norm == 0):#should never happen, since this is used as a marker of failing, a large value will signal done
            return 10
        oVec_W /= norm
        ang_cos = np.arccos(np.dot(oVec, oVec_W))
        return ang_cos

    #debug init state used if not random
    def dispResetDebug(self, notice=''):
        print('{} Notice : Setting specified init q/qdot/frc'.format(notice))
        print('initQPos : {}'.format(self.initQPos))
        print('initQVel : {}'.format(self.initQVel))
        print('initFrc : {}'.format(self.desExtFrcVal))

    #return observation given passed state and state dots
    #obs is slightly different than pure q/qdot (includes height in world frame), requiring skel to be modified
    #restores skel pose when finished - make sure q is correctly configured
    def getObsFromState(self, q, qdot):
        #save current state so can be restored
        oldState = self.state_vector()
        #set passed state
        self.set_state(np.asarray(q, dtype=np.float64), np.asarray(qdot, dtype=np.float64))
        #get obs (observation can be bounded or modified, that's why we use this method) - INCLUDES FORCE VALUE - if using to build new force value, need to replace last 3 elements 
        obs = self.getObs()        
        #return to original state
        self.set_state_vector(oldState)
        return obs  
    
    #check passed array of values to find and set min/maxs
    def _checkiMinMaxVals(self, vals, minMaxDict):
        #print('minmax check performed')
        minVals = minMaxDict['min']
        maxVals = minMaxDict['max']
        #check min and max torques seen
        for i in range(len(vals)):
            if minVals[i] > vals[i] : 
                minVals[i] = vals[i]
            elif maxVals[i] < vals[i] :
                maxVals[i] = vals[i] 

    #called at beginning of each rollout - resets this model, resetting its state
    #this is for RL-controlled models
    def reset_model(self, dispDebug=False):                   
        if(self.randomizeInitState):#if random, set random perturbation from initial pose
            #sets init pose then perturbs by some random amount
            q, dq = self.getRandomInitState()
            self.set_state(q, dq)
        else:#if not randomizing initial state then setting init state to preloaded state.  if neither this is ignored (which is very bad), but that should never happen, since randomizeInitState is only changed to false if loadedInitState is set to true
            #reset to be in initial pose
            self.setToInitPose()
            #resetting to pre-set initial pose
            if (self.loadedInitStateSet):
                if(dispDebug):
                    self.dispResetDebug('skelHolder::reset_model')
                self.set_state(self.initQPos, self.initQVel)
                self.loadedInitStateSet = False
            #commented out because helper bot triggers warning message 
            #TODO address this for helper bot
            # else:
            #     print('skelHolder {}::reset_model Warning : init skel state not randomized normonFrcTrqPostStep set to precalced random state'.format(self.skel.name))
                
        if(self.monitorTorques):
            #reset all torque monitors
            self.monTrqDict = {}
            self._resetMinMaxMonitors(self.monTrqDict, self.ndofs)
        #clear this to monitor # of tau applies per rollout
        self.numTauApplied = 0
        #if saving state-action-state' info, save here and then reinitialize array holding states
        if self.recStateActionRes : 
            self._saveStateData()
            
        #individual reset functionality
        self._resetIndiv(dispDebug)
        return self.getObs()
        
    #this will yield the reward parabola given min and 
    #max vals that yield rewards (roots of parabola) and reward for peak of parabola (1/2 between min and max)
    def calcRwdRootMethod(self, val, minRootVal, maxRootVal):
        #maxRwd is y val for avg of min and max root vals - set to 1, manage magnitude by weight instead
        xMax = (minRootVal + maxRootVal)/2.0
        mult = 1.0/((xMax-minRootVal) * (xMax - maxRootVal))    
        return mult * (val - minRootVal) * (val - maxRootVal)
    
    #get dynamic quantities from skeleton and sim, put in dictionary
    #dispFrcEefRes : display resultant forces to console
    #dispEefDet : display details of individual components
    def monFrcTrqPostStep(self, dispFrcEefRes=True, dispEefDet=False):
        self.frcD = {}
        if self.isFrwrdSim:
            #build fundamental force dictionary, in self.frcD
            self.frcD = self._buildPostStepFrcDict()
            #display force results, either summary or full results
            if(dispFrcEefRes):
                self.dispFrcEefRes(dispEefDet)
            #return debug dictionary
        return self.frcD

    #dictionary of reward components and data for debugging purposes.  define here so that broken sims don't result in seg faults if missing keys
    def _buildRwdDebugDict(self):
        dbgDict = defaultdict(list)
        #debug components
        dbgDict['dbgStrList']=[]
        dbgDict['dbgStrList_csv']=[]
        dbgDict['dbgStrList'].append('Rwd Comps :\n')
        dbgDict['dbgStrList_csv'].append('')
        dbgDict['rwdComps']=[]
        dbgDict['rwdTyps']=[]
        dbgDict['successDone']=[]
        dbgDict['failDone']=[]
        return dbgDict
       
    #functionality necessary before simulation step is executed for the human needing assistance
    def preStep(self, a):
        if self.recStateActionRes : 
            self.SASVec = np.concatenate([self.getObs(), a])
        #handle individual holder's pre-step code
        self._preStepIndiv(a)
        
    #save all data necessary to restore current state
    def _saveSimState(self):
        skel=self.skel
        self.saveStateForFwdBkwdStep = True
        self.savedQ = skel.q
        self.savedDq = skel.dq
        self.savedDdq = skel.ddq
        self.savedTau = np.copy(self.tau)

    #restore all states to pre-step state
    def _restoreSimState(self):
        if self.saveStateForFwdBkwdStep :
            skel=self.skel
            skel.set_accelerations(self.savedDdq)          
            skel.set_positions(self.savedQ)
            skel.set_velocities(self.savedDq) 
            self.tau = self.savedTau
        else :
            print("skelHldr_{}:_restoreSimState : Error - attempt to restore states that were not saved.".format(self.name))

        self.saveStateForFwdBkwdStep = False
        
    #forward simulate only this skeleton, saving and restoring all moblity settings of all other skels
    def _stepNoneButMe(self):
        #turn off all other skels so we can just step this skel with its tau
        skels = self.env.dart_world.skeletons
        numSkels = len(skels)
        #save all skels' current states
        prevSkelStates = [s.is_mobile() for s in skels]
        #turn off all skels (set to imobile)
        for s in skels:
            s.set_mobile(False)
        #turn this skel back on   
        self.skel.set_mobile(True)
        #step world
        self.env.dart_world.step()
        #return all skels to previous state of mobility
        for i in range(numSkels):
            s = skels[i]
            s.set_mobile(prevSkelStates[i])
        
    #forward simulate all skeletons but me
    def _stepAllButMe(self):
        #save current state, then set false so helper bot doesn't get re-solved/re-integrated
        oldBotMobileState = self.skel.is_mobile()
        #set bot imobile - already sim'ed
        self.setSkelMobile(False)                 
        #forward sim world
        self.env.dart_world.step()            
        #restore bot skel's mobility state
        self.setSkelMobile(oldBotMobileState)

    #functionality after sim step is taken - 
    #calculate reward, determine if done(temrination conditions) and return observation
    #and return informational dictionary
    #debugRwd : whether to save debug reward data in dictionary dbgDict
    #dbgEefFrc : whether to query dynamic data to determine force at end effector and display results to console
    #dispEefDet : display detail of every component of frcD dictionary
    def postStep(self, resDict, debugRwd=False):
        #resDict holds whether the sim was broken or not - if sim breaks, we need 
        #holds these values : {'broken':False, 'frame':n_frames, 'stableStates':stblState}   
        #check first if sim is broken - illegal actions or otherwise exploding  
        
        #observation of current state
        obs = self.getObs()

        #dictionary of reward components and data for debugging purposes.  define here so that broken sims don't result in seg faults if missing keys
        if(debugRwd):
            dbgDict = self._buildRwdDebugDict()
        else :
            dbgDict = defaultdict(list)
            
        if(resDict['broken'] and (self.name in resDict['skelhldr'])):
            print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('{} has broken sim : reason : {}'.format(self.name, resDict['reason']))
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            done = True
            rwd = 0#-100 * self.env.sim_steps
            self.env.numBrokeSims+=1
            #don't save state info for broken sim - since entire trajectory is booted, get rid of the traj's data too
            self.listOfSASVec = []

        else :        
            #not broken                    
            rwd, done, dbgDict = self.calcRewardAndCheckDone(obs, debugRwd, dbgDict)            

            #save best state seen so far
            if (self.bestStateMon > 0):#(self.checkBestState) or (self.buildStateCDF):
                self.addStateToStateList( rwd, self.state_vector())
                
            #if saving state info, add current state
            if self.recStateActionRes : 
                #get extra holder-specific components for SAS, if any
                privCmp = self._addStateVecToSASlistPriv()
                stateVec = np.concatenate([self.SASVec, obs, privCmp])
                stateVecStr = np.array2string(stateVec, separator=',', formatter={'float_kind':lambda x: "%.8f" % x})
                #print('State vec str : {}, len : {}'.format(stateVecStr, len(stateVec)))
                self.listOfSASVec.append(stateVecStr)
               
        return obs, rwd, done, dbgDict
    
    #return skeleton qdot - maybe clipped, maybe not
    def getClippedSkelqDot(self):
        return np.clip(self.skel.dq, -self.qDotBnd , self.qDotBnd )
    
    #base check goal functionality - this should be same for all agents,
    #access by super()
    def checkSimIsBroken(self):
        q = self.skel.q
        dq = self.skel.dq     
        s = np.concatenate([q, dq])
        if not(self.isFrwrdSim):#if not frwrd simed then assume sim won't be broken
            return False, s, 'OK-NON_SIM'
            
        if not (np.isfinite(q).all()):  #make sure not inf or nan
            return True, s, 'INFINITE/NAN : q'        
        if ((np.abs(q[self.stTauIdx:]) > 1000).any()):#ignore world location/orientation component of state - if abs(any)> 1000, then broke
            return True, s, 'EXPLODE : q'

        if not (np.isfinite(dq).all()):  #make sure not inf or nan
            return True, s, 'INFINITE/NAN : dq'        
        if ((np.abs(dq) > 1000).any()):#ignore world location/orientation component of state - if abs(any)> 1000, then broke
            return True, s, 'EXPLODE : dq'

        return False, s, 'OK'
    
    def checkBNinNameList(self, name, nameList):
        return any(name in bodyNodeName for bodyNodeName in nameList)

    #check if passed body nodes are on two different, non-ground, skeletons - return true
    #don't want this - skeletons should not contact each other except through the ball joint constraint
    def checkBNKickOtr(self, bn1, bn2):
        if ('ground' in bn1.name) or ('ground' in bn2.name):
            return False
        #if not touching ground, then this is bad contact between robot and ANA - need to check if eef contact, which is good
        #if != then they are different skeletons contacting each other
        return (bn1.skel.name != bn2.skel.name)    
    
    #returns dictionary of per-body node contact info objects
    def getMyContactInfo(self):        
        allCntctInfo = self.env.getContactInfo()
        return allCntctInfo[self.skel.name]        
    
    #returns true only if one body node is a foot on this skeleton and the other is the ground in a contact
    #also returns body node that is responsible for contact (not ground) whether foot or not
    def checkMyFootWithGround(self, bn1, bn2):
        retVal = False
        bn = None
        if ('ground' in bn1.name) :
            bn = bn2
            if (self.skel.name == bn2.skel.name) :
                retVal = (self.checkBNinNameList(bn2.name, self.feetBodyNames))

        elif ('ground' in bn2.name) :
            bn = bn1
            if (self.skel.name == bn1.skel.name) :
                retVal = (self.checkBNinNameList(bn1.name, self.feetBodyNames))
        return retVal, bn    
        
    #calculate average foot body locations based on COM
    def calcAvgFootBodyLoc(self):        
        avgFootLoc = np.zeros(3)
        avgLFootLoc = np.zeros(3)
        avgRtFootLoc = np.zeros(3)
        if(len(self.feetBodyNames) >0):
            for ftBdyName in self.leftFootBodyNames:
                footLoc = self.skel.body(ftBdyName).com()
                avgFootLoc += footLoc
                avgLFootLoc += footLoc
            avgLFootLoc /= len(self.leftFootBodyNames)
            
            for ftBdyName in self.rightFootBodyNames:
                footLoc = self.skel.body(ftBdyName).com()
                avgFootLoc += footLoc
                avgRtFootLoc += footLoc
            avgRtFootLoc /= len(self.rightFootBodyNames)           

            avgFootLoc /= len(self.feetBodyNames)
        else :
            print('skelHolder::calcAvgFootBodyLoc : No feet for skelholder {} defined in self.feetBodyNames so avg loc is origin'.format(self.name))
        return [avgFootLoc, avgLFootLoc, avgRtFootLoc]

    #calculate contact's contribution to total contact torques around distant point for COP derivation
    #cntctBN : the body node in contact with the ground
    #contact : the contact object
    def calcSumTrqsForCOP(self, cntctBN, contact, ctrTauPt):
        trqPt = contact.point - ctrTauPt
#        if(contact.point[1] != 0):
#              print('!!!!!!!!!!!!!!!!!!!!!! non zero ground contact y : {}'.format(contact.point))
        new_COP_tau = np.cross(trqPt, contact.force)
        new_COP_ttlFrc = contact.force
        return new_COP_tau,new_COP_ttlFrc
    
    #calculate sum of contact contributions to tau - final all contact forces and contact jacobians, use contact jacobian transposes * cntct force to get tau_cntct
    def calcCntctTau(self, useLinJacob):
        contacts = self.env.dart_world.collision_result.contacts        
        tau_cntct = np.zeros(self.ndofs)
        cntctDict = defaultdict(list)
        
        if (useLinJacob) :             
            for contact in contacts : 
                if (self.skel.name != contact.bodynode1.skeleton.name ) and (self.skel.name != contact.bodynode2.skeleton.name ) :
                    #contact not from this skeleton
                    continue
                bn = contact.bodynode1 if (self.skel.name == contact.bodynode1.skeleton.name) else contact.bodynode2
                pt = bn.to_local(contact.point)
                frc = contact.force
                JTrans = np.transpose(bn.linear_jacobian(offset=pt))           
                tau_cntct += (JTrans.dot(frc))
                cntctDict['bn'].append(bn)
                cntctDict['pt'].append(pt)
                cntctDict['frc'].append(frc)
        else :        
            for contact in contacts : 
                if (self.skel.name != contact.bodynode1.skeleton.name ) and (self.skel.name != contact.bodynode2.skeleton.name ) :
                    #contact not from this skeleton
                    continue
                bn = contact.bodynode1 if (self.skel.name == contact.bodynode1.skeleton.name) else contact.bodynode2
                pt = bn.to_local(contact.point)
                frc = contact.force
                useForce = np.zeros(6)
                useForce[3:]=contact.force
                frc=useForce
                JTrans = np.transpose(bn.world_jacobian(offset=pt))
                tau_cntct += (JTrans.dot(frc))
                cntctDict['bn'].append(bn)
                cntctDict['pt'].append(pt)
                cntctDict['frc'].append(frc)
            
        return tau_cntct,cntctDict
            
    #calculate foot contact count and other terms if we want to use them for reward calc
    #TODO need to calculate ZMP instead
    def calcAllContactDataNoCOP(self):
        contacts = self.env.dart_world.collision_result.contacts        
        contactDict = defaultdict(float)       
       
        for contact in contacts:
            if (self.skel.name != contact.bodynode1.skeleton.name ) and (self.skel.name != contact.bodynode2.skeleton.name ) :
                #contact not from this skeleton
                continue
            contactDict['ttlCntcts'] +=1
            #penalize contact between the two skeletons - getup-human should not touch assistant bot except with assisting hand
            #only true if one skel contacts other and the other is not the ground - i.e. non-ground non-self contact
            if self.checkBNKickOtr(contact.bodynode1, contact.bodynode2):
                #this is a bad contact
                contactDict['BadContacts']+=1
                contactDict['kickBotContacts'] +=1            
            #only true if feet are contacting skeleton - kicking self - i.e. non-ground self contact
            elif (contact.bodynode1.skel.name == contact.bodynode2.skel.name):
                #this is a bad contact
                contactDict['BadContacts']+=1
                contactDict['selfContact'] +=1
            else:
                #this is a contact between this skeleton with the ground
                #cntctBN is skeleton body node responsible for ground contact, or none if no ground contact
                isGndCntctWFoot, cntctBN = self.checkMyFootWithGround(contact.bodynode1,contact.bodynode2)                 
                #save refs to each contact force and which body had contact with ground
                contactDict[cntctBN.name] +=1 
                if isGndCntctWFoot: #save for COP calc
                    #foot contact with ground     
                    #contactDict['COPcontacts'] += 1
                    contactDict['GoodContacts'] += 1
                    contactDict['footGroundContacts'] +=1
                    
                elif(self.checkBNinNameList(cntctBN.name, self.handBodyNames)):
                    #hand contact with ground
                    #contactDict['COPcontacts'] += 1
                    contactDict['handGroundContacts']+=1  
                    contactDict['GoodContacts'] += 1
                    #hand should not be considered for cop calc, will stay stradled with hand on ground - use COP calc to get COM over COP with standup
                    
                else :
                    #nonHand nonFoot Contact with ground 
                    contactDict['BadContacts']+=1
                    contactDict['badGroundContact']+=1

        return contactDict

        
    # #calculate foot contact count and other terms if we want to use them for reward calc
    # #TODO need to calculate ZMP instead ?
    # def calcAllContactData(self):
    #     contacts = self.env.dart_world.collision_result.contacts        
    #     contactDict = defaultdict(float)       
               
    #     COPval = np.zeros(3)
    #     COP_tau = np.zeros(3)
    #     COP_ttlFrc = np.zeros(3)

    #     #tau = loc x frc
    #     #COP calculation is the location that, when crossed with total force, will give total moment
    #     #we can calculate total moment by crossing all contact forces with all contact locations
    #     #we can get total force, and from this we can find the location that would produce the total 
    #     #torque given the total force (by first constraining one of the dimensions of the point in question)
    #     #we choose to constrain the y coordinate to be 0 since we want the cop on the ground
       
    #     for contact in contacts:
    #         if (self.skel.name != contact.bodynode1.skeleton.name ) and (self.skel.name != contact.bodynode2.skeleton.name ) :
    #             #contact not from this skeleton
    #             continue
    #         contactDict['ttlCntcts'] +=1
    #         #penalize contact between the two skeletons - getup-human should not touch assistant bot except with assisting hand
    #         #only true if one skel contacts other and the other is not the ground - i.e. non-ground non-self contact
    #         if self.checkBNKickOtr(contact.bodynode1, contact.bodynode2):
    #             #this is a bad contact
    #             contactDict['BadContacts']+=1
    #             contactDict['kickBotContacts'] +=1            
    #         #only true if feet are contacting skeleton - kicking self - i.e. non-ground self contact
    #         elif (contact.bodynode1.skel.name == contact.bodynode2.skel.name):
    #             #this is a bad contact
    #             contactDict['BadContacts']+=1
    #             contactDict['selfContact'] +=1
    #         else:
    #             #this is a contact between this skeleton with the ground
    #             #cntctBN is skeleton body node responsible for ground contact, or none if no ground contact
    #             isGndCntctWFoot, cntctBN = self.checkMyFootWithGround(contact.bodynode1,contact.bodynode2)                 
    #             #save refs to each contact force and which body had contact with ground
    #             contactDict[cntctBN.name] +=1 
    #             if isGndCntctWFoot: #save for COP calc
    #                 #foot contact with ground     
    #                 contactDict['COPcontacts'] += 1
    #                 contactDict['GoodContacts'] += 1
    #                 contactDict['footGroundContacts'] +=1
                    
    #                 #print('With Ground : contact body 1 : {} skel : {} | contact body 2 : {} skel : {} : frc {}'.format(contact.bodynode1,contact.bodynode1.skeleton.name, contact.bodynode2,contact.bodynode2.skeleton.name, contact.force))
    #                 #COP shouldnt use butt contact in calculation - want a target to get up
    #                 #find total moment of all contacts
    #                 cTau, cTtlFrc = self.calcSumTrqsForCOP(cntctBN, contact,self.env.ctrTauPt)
    #                 COP_tau += cTau
    #                 COP_ttlFrc += cTtlFrc                    
    #             elif(self.checkBNinNameList(cntctBN.name, self.handBodyNames)):
    #                 #hand contact with ground
    #                 #contactDict['COPcontacts'] += 1
    #                 contactDict['handGroundContacts']+=1  
    #                 contactDict['GoodContacts'] += 1
    #                 #hand should not be considered for cop calc, will stay stradled with hand on ground - use COP calc to get COM over COP with standup
                    
    #                 #non-foot contact with ground - check if contact with non-reaching hand
    #                 #print('Not Foot With Ground : contact body 1 : {} skel : {} | contact body 2 : {} skel : {}'.format(contact.bodynode1,contact.bodynode1.skeleton.name, contact.bodynode2,contact.bodynode2.skeleton.name))
    #                 #find total moment of all contacts of hand to ground
    #                 #cTau, cTtlFrc = self.calcSumTrqsForCOP(cntctBN, contact,self.env.ctrTauPt)
    #                 #COP_tau += cTau
    #                 #COP_ttlFrc += cTtlFrc                    
    #             else :
    #                 #nonHand nonFoot Contact with ground 
    #                 contactDict['BadContacts']+=1
    #                 contactDict['badGroundContact']+=1
    #     #determines COP based on foot contacts with ground - ignore non-foot contacts for COP calc
    #     #COP loc might be redundant - using foot location might be sufficient
    #     if((0 < contactDict['COPcontacts']) and (np.abs(COP_ttlFrc[1]) > 0)):#(COP_ttlFrc[1] > 0) and (COP_ttlFrc[0] > 0)):
    #         #COP_tau = COPval cross COP_ttlFrc ==> need to constrain possible COPvals -> set COPval.y == 0 since we want the COP at the ground, and then solve eqs : 
    #         COPval = self.calcCOPFromTauFrc(COP_tau, COP_ttlFrc, self.env.ctrTauPt)
    #     else :  #estimate COP as center of both feet body node com locations         
    #         #find average foot location (idx 0)
    #         avgFootLocAra = self.calcAvgFootBodyLoc()
    #         COPval = np.copy(avgFootLocAra[0])
    #         #put it on the ground 
    #         COPval[1]= 0

    #     return contactDict, COPval  

    #Take total moments, total force, return a suitable point of application of that force to provide that moment - more than 1 answer, want answer on ground plane
    #COP_tau = COPval cross COP_ttlFrc ==> need to constrain possible COPvals
    #so set COPval.y == 0 since we want the COP at the ground, and then solve eqs :
    #ctrTauPt is point far away around which torques are initially calculated, then summed, and then reversed to find COP loc      
    #ctrTauPt[1] should be 0, since can be chosen arbitrarily    
    def calcCOPFromTauFrc(self,COP_ttlTau, COP_ttlFrc, ctrTauPt):
        #v_Ocop : vector from ctrTauPt to COP loc == n_floor x ttlTau / n_floor . ttlFrc
        #n_floor : floor normal
        #ttlTau : total torque around ctrTauPt
        #ttlFrc : total contact force
        #given a cross b (==c : total tau) and b (total force), find a (location vector) : a = (bxc)/(b.b)  + tb, where t is any real scalar
        COPvalTmp = np.cross(COP_ttlFrc,COP_ttlTau)/np.dot(COP_ttlFrc,COP_ttlFrc)#soln for t==0
        #solve for t by finding frc[1]*t - COPval[1]=0 since we want COP val to lie on y=0 plane
        try:
            t = -float(COPvalTmp[1])/COP_ttlFrc[1]
            COPval = COPvalTmp + t*COP_ttlFrc
            #print('Cop Val : {} from COPvalTmp {} W/(t = {}) + ctrTauPt == {}'.format(COPval, COPvalTmp,t,(COPval+ctrTauPt)))
        except ZeroDivisionError:
            #should never hit this -> COP ttl force in y dir is not allowed to be 0 to call this - if it is then we use avg loc of feet as COP
            #print('Cop Val w/ttlFrc[1] == 0 : {} + ctrTauPt : {} '.format(COPvalTmp,ctrTauPt))
            COPval = COPvalTmp 
        #displacing COPval by ctrTauPt
        COPval += ctrTauPt
        #print('COP Value : {}'.format(COPval))
        return COPval       
    
    #return the end effector location in world coords and the constraint location in  world coords
    def getCnstEffLocs(self):
        return  self.cnstrntBody.to_world(x=self.cnstrntOnBallLoc), self.reachBody.to_world(x=self.reachBodyOffset)    
        
    #return jacobaian inv of trans, jacobian transpose and jacobian (either linear or world) of passed body at passed offset, along with dynamically consistent inverse with passed mass matrix, if specified
    def getEefJacobians(self, useLinJacob, body, offset, calcDynInv=False, M=None):
        if (useLinJacob) :             
            Jpull = body.linear_jacobian(offset=offset)            
        else : 
            Jpull = body.world_jacobian(offset=offset) 

        JTrans = np.transpose(Jpull)
        JTransInv = np.linalg.pinv(JTrans)
        DynInv = None
        #calc dynamically consistent inverse - Minv * Jt * (J * Minv * Jt)^-1
        if calcDynInv :
            Minv = np.linalg.inv(M)         #ndof x ndof
            MinvJt = Minv.dot(JTrans)       #ndof x 3
            JMinvJtinv = np.linalg.inv(Jpull.dot(MinvJt)) #3x3
            DynInv = np.transpose(MinvJt.dot(JMinvJtinv))
        return JTransInv, JTrans, Jpull, DynInv      

    #return constraint force as seen at eef
    def getEefCnstrntFrc(self, useLinJacob):
        #        #jacobian to end effector 
        JTransInv, JTrans, Jpull, _ = self.getEefJacobians(useLinJacob, body=self.reachBody, offset=self.reachBodyOffset)
        #convert constraint force as seen at dofs to world force seen at eef
        worldEefCnstrntFrc = JTransInv.dot(self.skel.constraint_forces())
        return worldEefCnstrntFrc

    #this will calculate external force as seen at eef, not counting contributions from contact with ground
    def getExternalForceAtEef(self, useLinJacob):
        #        #jacobian to end effector 
        JTransInv, JTrans, Jpull, _ = self.getEefJacobians(useLinJacob, body=self.reachBody, offset=self.reachBodyOffset)
        #convert constraint force as seen at dofs to world force seen at eef
        cnstFrc = self.skel.constraint_forces()
        #convert constraint force as seen at dofs to world force seen at eef
        allCnstrntFrcAtEef = JTransInv.dot(cnstFrc)
        #now determine contributions from contact with ground
        #get current contact info - dict of per-body contacts
        cntctInfo = self.getMyContactInfo()
        #use bdy.calcTtlBdyTrqs() to get total body torque from contacts 
        ttlCntctBdyTrque = np.zeros(np.size(self.skel.q,axis=0))
        for k,v in cntctInfo.items():
            ttlCntctBdyTrque += v.calcTtlBdyTrqs()
        ttlCntctBdyFrcAtEef = JTransInv.dot(ttlCntctBdyTrque)
        #subtract cntct frc @ eef from ttl cnstrnt force @ eef to get external force due to assist
        #add due to sign of contact forces? different collision detectors seem to have different signs/directions for the contacts they return
        extFrcAtEef = allCnstrntFrcAtEef + ttlCntctBdyFrcAtEef

        return extFrcAtEef, allCnstrntFrcAtEef, ttlCntctBdyFrcAtEef
    
    #return body torques to provide self.desExtFrcVal at toWorld(self.constraintLoc)
    #provides JtransFpull component of equation
    def getPullTau(self, useLinJacob, debug=False):  
        JTransInv, JTrans, Jpull, _ = self.getEefJacobians(useLinJacob, body=self.reachBody, offset=self.reachBodyOffset)
        if (useLinJacob) :             
            self.useForce = self.desExtFrcVal
        else : 
            #wrench TODO verify target orientation should be 0,0,0
            self.useForce = np.zeros(6)
            self.useForce[3:]=self.desExtFrcVal
        if(debug):
            print('getPullTau : pull force being used : {} '.format(self.useForce))
            
        resTau = JTrans.dot(self.useForce)
        #last 3 rows as lin component
        return resTau, JTransInv, Jpull, Jpull[-3:,:]
    
    #build force dictionary to be used to verify pull force after sim step 
    #rwd is reward generated for the motion of this step of sim
    #indivCalcFlag : passed to child class impelementation
    def _buildPostStepFrcDict(self, calcTauMag=True, indivCalcFlag=True):
        skel = self.skel
        ma = skel.M.dot(skel.ddq )    
        cg = skel.coriolis_and_gravity_forces() 
        #torque cntrol desired to provide pulling force at contact location on reaching hand    
        JtPullPInv_new, _, _, _ = self.getEefJacobians(self.useLinJacob, body=self.reachBody, offset=self.reachBodyOffset)
        
        frcD = {}      
        t=self.tau
        frcD['tau']=np.copy(t)
        if calcTauMag :
            frcD['tauMag']= np.sqrt(t.dot(t))
        frcD['ma']=ma
        frcD['cg']=cg
        #jacobian pemrose inverse to pull contact point
        frcD['JtPullPInv_new'] = JtPullPInv_new        
        frcD['jtDotCGrav'] = JtPullPInv_new.dot(cg)
        frcD['jtDotMA'] = JtPullPInv_new.dot(ma)
        frcD['jtDotTau'] = JtPullPInv_new.dot(t)
        
        #handle individual skel's frc components that may affect calculation of totPullFrc, 
        #also calculate totPullFrc appropriately for this skel (might include constraint force or not)       
        frcD = self._bldPstFrcDictPriv(frcD,indivCalcFlag)
        
        #if monitoring generated force over the life of rollout
        if(self.monitorGenForce):
            self._checkiMinMaxVals(frcD['totPullFrc'], self.minMaxFrcDict)
            #self._checkiMinMaxVals(frcD['totPullFrcCnst'], self.minMaxFrcDict, self.maxGenFrc)
            self.totGenFrc.append(frcD['totPullFrc'])
            #self.totGenFrc.append(frcD['totPullFrcCnst'])        
        return frcD
  
    def _IK_setSkelAndCompare(self, skel, reachBody, q, pos):
        skel.set_positions(q)
        eefWorldPos = reachBody.to_world(x=self.reachBodyOffset)
        diff = pos - eefWorldPos
        return diff, diff.dot(diff), eefWorldPos 
    
    #IK eef to world location of constraint
    def IKtoCnstrntLoc(self):
        self.IKtoPassedPos(self.trackBodyCurPos, self.skel, self.reachBody)

    #track passed skel to passed position
    def IKtoPassedPos(self, pos, skel, reachBody):
        maxIKIters = self.IKMaxIters
        offset=self.reachBodyOffset
        minIKAlpha=self.minIKAlpha
        if(self.debug_IK):
            print('\nIK to pos : {}'.format(pos))
        #eefWorldPos is only used locally
        delPt, distSqP, eefWorldPos= self._IK_setSkelAndCompare(skel, reachBody, skel.q, pos)
        iters = 0
        #oldQ is always last good state of skeleton
        oldQ = skel.q
        oldDistSq = distSqP
        iters = 0
        while ((distSqP > 0.000001) and (iters < maxIKIters)):
            #reset learning rate
            alpha = 5.0
            #jacobian to end effector 
            JPullPInv = np.linalg.pinv(reachBody.linear_jacobian(offset=offset))
            #dq = J^-1 * dx
            delQ = JPullPInv.dot(delPt)
            #apply current jacobian repeatedly until doesn't improve
            while True :
                #find new skeleton dofs using q_new == q_old + alpha * dq
                #newQ = skel.q + alpha*delQ #TODO verify oldq preserves skeleton state properly
                newQ = oldQ + alpha*delQ
                #set skeleton to new q and compare position of eef to target
                delPt, distSqP, eefWorldPos = self._IK_setSkelAndCompare(skel, reachBody, newQ, pos)

                if(self.debug_IK):
                    print('iters : {} | alpha : {} | distSqP (sqDist) : {} | old dist : {}'.format(iters, alpha, distSqP,oldDistSq))
                
                #if got worse, reset skel and try again with smaller alpha *= .5 or just break out
                if (distSqP > oldDistSq):
                    #return to previous state (oldQ), and reset lcl refs to dist to target and end effector world position
                    delPt, oldDistSq, eefWorldPos = self._IK_setSkelAndCompare(skel, reachBody, oldQ, pos)        
                    #alpha not too small, make smaller               
                    if(alpha > minIKAlpha):
                        alpha *= .5                    
                    #alpha too small and getting worse, find new jacobian
                    else :
                        distSqP = oldDistSq
                        break           
                #got better, continue, saving lcl refs to best res
                else:
                    #last good distance measure of skeleton eef
                    oldDistSq = distSqP
                    #oldQ is always last good state of skeleton
                    # poll skeleton to make sure we haven't exceeded joint lims after last application of jacobian, instead of using newQ
                    oldQ = skel.q
                    
            if(self.debug_IK):
                print('iter:{} delPt : {} | delQ {} | distSqP : {} | eff world pos {}'.format(iters,delPt, delQ, distSqP, eefWorldPos))
            iters +=1
        self.trackBodyCurPos = pos
        if(self.debug_IK):
            print('IK Done : final world position {}\n'.format(eefWorldPos))
    

    ######################################################
    #   debug, testing and helper functions
    ######################################################
    
    #display results of fwd simulation - force generated at end effector
    def dispFrcEefRes(self, dispDetail):    
        print('\nForce gen results for {} at eef @ sim step {} (# of torque applications) :'.format(self.name, self.numTauApplied))
        if dispDetail: 
            for k,v in self.frcD.items():
                if ('jt' in k) :#or ('tot' in k):
                    print('\t{} : \t{}'.format(k,v))
        #print('\t\tDiff between optguess dq and actual dq : \t{:5f}'.format(self.frcD['dqDiff'])) #always 0
        print('\t\tEEF Meas Force : \t{}'.format(self.frcD['totPullFrc'][-3:]))
        print('\t\tTarget Force   : \t{}'.format(self.frcD['targetEEFFrc'][-3:]))
        #print('\t\tAbove W/Cnst Frc : \t{}'.format(self.frcD['totPullFrcCnst']))
        #print('\n\tPull Force : \t{}'.format(self.frcD['jtDotTau']))
        #display individual skeleton-related results         
        #frcDiffOld =self.frcD['totPullFrcOld'] - self.useForce
        #print('Difference between proposed force and actual : {} \n\tdesFrc : {} \n\tjtDotTau  : {}\n\tcntctFrc : {} \n\tskelGrav : {}'.format(frcDiff,self.useForce,jtDotTau, cntctFrcTtl,skelGrav))
        #print('\nDifference between proposed force and actual : \n{} \n\tdesFrc : \t{}\n\ttotPullFrc  : \t{}\n\tSum of all components :\t{}'.format(frcDiff,self.wrenchFrc, totPullFrc, sumTtl))
        #print('\nDifference between proposed force and actual : \n\t\t\t{} \n\tdesFrc : \t{}\n\ttotPullFrc  : \t{}\n\n'.format(frcDiffNew,self.useForce, self.frcD['totPullFrc']))

        #display skeleton-specific force results
        self._dispFrcEefResIndiv()
        print('\n')
   
    #called externally to debug end effector location
    def dbg_getEffLocWorld(self):
        return self.reachBody.to_world(x=self.reachBodyOffset)   
    
    #align sequence of values
    def dotAligned(self, seq):
        strNums = ['{:.5f}'.format(n) for n in seq]
        dots = [s.find('.') for s in strNums]
        m = max(dots)
        return [' '*(m - d) + s for s, d in zip(strNums, dots)]    
    
    #will display current torque vector and RL-policy action vector values formatted with names of joints being applied to
    def dbgShowTauAndA(self, name=' '):
        print('\n{}Torques and causing Actions : '.format(name))
        #should have a dof corresponding to each component of tau
        dofs = self.skel.dofs
        alignTauStr = self.dotAligned(self.tau)
        alignAStr = self.dotAligned(self.a)
        numDofs = len(dofs)
        if(numDofs != len(self.tau)):
            print('!!!!!! attempting to print torques that do not align with skeleton dofs')
            return
        for i in range(0,self.stTauIdx):
            print('\tDof : {:20s} | Torque : {} | Action : {:.5f}'.format(dofs[i].name, alignTauStr[i],0))   
        for i in range(self.stTauIdx,len(dofs)):
            print('\tAct Dof : {:20s} | Torque : {} | Action : {}'.format(dofs[i].name, alignTauStr[i],alignAStr[(i-self.stTauIdx)]))   
    
    def dbgShowState(self, name=' '):
        print('\n{}Q and Qdot : '.format(name))
        #should have a dof corresponding to each component of tau
        dofs = self.skel.dofs
        align_QStr = self.dotAligned(self.skel.q)
        align_dQStr = self.dotAligned(self.skel.dq)
        numDofs = len(dofs)
        if(numDofs != len(self.tau)):
            print('!!!!!! attempting to print torques that do not align with skeleton dofs')
            return
        for i in range(0,len(dofs)):
            print('\tDof : {:20s} | Q : {} | Qdot : {:.5f}'.format(dofs[i].name, align_QStr[i],align_dQStr[i]))   
 

    #will display passed torque values formatted with names of joints being applied to
    def dbgShowTorques(self, tau, name=' '):
        print('\n{}Torques : '.format(name))
        #should have a dof corresponding to each component of tau
        dofs = self.skel.dofs
        alignTauStr = self.dotAligned(tau)
        if(len(dofs) != len(tau)):
            print('!!!!!! attempting to print torques that do not align with skeleton dofs')
            return
        for i in range(len(dofs)):
            print('\tDof : {:20s} | Value : {}'.format(dofs[i].name, alignTauStr[i]))            
            
    #display min/max torques seen so far
    def dbgDispMinMaxTorques(self):
        if(self.monitorTorques):        
            #min/max torques seen
            self.dbgShowTorques(self.monTrqDict['min'],name='Minimum Seen ')
            print('\n')
            self.dbgShowTorques(self.monTrqDict['max'],name='Maximum Seen ')
            print('\n')
        else:
            print('Torques Not Monitored')

    #called only to clear derived torques
    def dbgResetTau(self):
        #set tau to 0
        self.tau = np.zeros(self.ndofs)
    
    #debug functionality - show skel body names, joint names and dof names
    def dbgShowSkelVals(self):
        self.dbgShowSkelNames(self.skel.bodynodes, 'Body')
        self.dbgShowSkelNames(self.skel.joints, 'Joint')
        self.dbgShowSkelNames(self.skel.dofs, 'Dof')

    #display skel-related object names
    def dbgShowSkelNames(self, objs, typeStr=''):
        numObjs = len(objs)
        print('\n{} {} names : '.format(numObjs,typeStr))
        for bidx in range(numObjs):
            d = objs[bidx]
            print('\t{}'.format(d.name))

    def dbgShowDofLims(self):
        lims = self.getObsLimits()
        numDofs = self.ndofs
        print('\n{} Dof names and limit values : '.format(numDofs))
        for bidx in range(numDofs):
            d = self.skel.dof(bidx)
            print('{} :\tMin : {:.3f}\tMax : {:.3f}\tMin Vel : {:.3f}\tMax Vel : {:.3f}'.format(d.name, lims['lowQLims'][bidx], lims['highQLims'][bidx], lims['lowDQLims'][bidx], lims['highDQLims'][bidx]))    

    #display min and max forces seen at skel's eef        
    def dbg_dispMinMaxForce(self):
        if(self.monitorGenForce):
            print('Desired Force value : \t{}'.format(self.useForce) )
            print('Min Force Value Seen : \t{}'.format(self.minMaxFrcDict['min']))
            print('Max Force Value Seen : \t{}'.format(self.minMaxFrcDict['max']))
            # print('Min Force Value Seen : \t{}'.format(self.minGenFrc))
            # print('Max Force Value Seen : \t{}'.format(self.maxGenFrc))
            mean = np.mean(self.totGenFrc, axis=0)
            print('Mean Force Seen : \t{}'.format(mean))
            stdVal = np.std(self.totGenFrc, axis=0)
            print('Std of Force Seen : \t{}\n'.format(stdVal))
        else:
            print('Min/Max Force generated not monitored.  Set self.monitorGenForce to true in constructor' )

    #get dictionary of environmental/simulation variables used for optimization at reach body constraint location (where applicable)
    def getOptVars(self):
        res = {}        
        res['M']=self.skel.M
        res['CfrcG']=self.skel.coriolis_and_gravity_forces()
        res['jacobian']=self.reachBody.jacobian(offset=self.reachBodyOffset)
        res['world_jacobian']=self.reachBody.world_jacobian(offset=self.reachBodyOffset)
        res['linear_jacobian']=self.reachBody.linear_jacobian(offset=self.reachBodyOffset)
        res['angular_jacobian']=self.reachBody.angular_jacobian()       
        
        return res
    
    #this returns heading string for state-aciton-state'-reward 
    def _SASVecGetDofNames(self, names, t):
        qNameStr = ','.join( ['S{} {}_pos'.format(t,name) for name in names])
        qdotNameStr = ','.join( ['S{} {}_vel'.format(t,name) for name in names])
        strRes = ','.join([qNameStr,qdotNameStr])
        obsExt = self._SASVecGetDofNamesPriv(t) 
        if len(obsExt) > 0 :
            strRes = ','.join([strRes, obsExt])
        return strRes
    
    #this returns heading string for state-aciton-state'-reward 
    def getSASVecHdrNames(self):
        numDofs = self.ndofs
        names = [self.skel.dof(bidx).name.replace('j_','') for bidx in range(numDofs)]
        action = ','.join(['A {}'.format(name) for name in names[self.stTauIdx:]])
        initStrRes = self._SASVecGetDofNames(names, 0)
        endStateRes = self._SASVecGetDofNames(names, 1)
        strRes = ','.join([initStrRes,action,endStateRes])
        indivSASRes = self._SASVecGetHdrNamesPriv()        
        if len(indivSASRes) > 0:
            strRes = ','.join([strRes,indivSASRes])
        return strRes

    #turn on saving of state-action-state' info to passed file name
    def setStateSaving(self, saveStates=True, fileName='RO_SASprimeData.csv'):
        self.recStateActionRes = saveStates
        if not self.recStateActionRes :
            self.savedStateFilename = ''
            self.SASTrajNum = 0
            return

        fileName = '{}_{}'.format(self.shortName, fileName)
        baseDirName = '~/rolloutCSVData/{}_{}/'.format(self.shortName, 'RO_SASprime')        
        import os
        dirName = self.env.getRunDirName()[:-3]
        directory = os.path.join(os.path.expanduser(baseDirName) + dirName)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.savedStateFilename = os.path.join(directory, fileName) 
        #if file doesn't exist, create it, write header, and save
        if not os.path.exists(self.savedStateFilename):
            #open file and write header string corresponding to column names
            hdrString = self.getSASVecHdrNames() + ',Traj Step, Traj Progress, Traj #'
            self.SASTrajNum = 0
            f = open(self.savedStateFilename, 'w')
            f.write('{}\n'.format(hdrString))
            f.close()

    def _saveStateData(self):
        numSteps = len (self.listOfSASVec)
        
        if (numSteps > 0):
            #vectors available to save to file - append to given file name
            f = open(self.savedStateFilename, 'a')
            step=0
            for state in self.listOfSASVec:
                prg = step/(1.0*numSteps)
                resStr='{},{},{},{}'.format(state.replace('\n', '').replace('\r', '')[1:-1],step,prg,self.SASTrajNum)
                step+=1
                #print('------------\n{}\n--------------\n'.format(resStr))
                f.write('{}\n'.format(resStr))#remove beginning and ending brackets from string rep of list
            f.close()
            self.SASTrajNum += 1
        self.listOfSASVec = []

    #called externally when a rollout is terminated
    def checkRecSaveState(self):
        #if saving state-action-state' info, save here and then reinitialize array holding states
        if self.recStateActionRes : 
            self._saveStateData()
    
    ######################################################
    #   abstract methods 
    ######################################################    
        
    #builds a dictionary describing the idxs in each SAS' record
    @abstractmethod
    def _initSaveStateRecPriv(self, endIdx): pass    
    #add holder-instance specific state values to save to file - only called if actually saving to file - returns a list of values
    @abstractmethod
    def _addStateVecToSASlistPriv(self): pass  
    #return idnvidual holder's obs dof description
    @abstractmethod
    def _SASVecGetDofNamesPriv(self, t): pass
    #return any exta components beyond SAS' to be saved in SAS' csv file
    @abstractmethod
    def _SASVecGetHdrNamesPriv(self): pass
    #build the configuration of the initial pose of the figure
    #@abstractmethod
    #def _makeInitPoseIndiv(self): pass          
    #individual skeleton handling for calculating post-step dynamic state dictionary
    @abstractmethod
    def _bldPstFrcDictPriv(self, frd,indivCalcFlag): pass    
    #need to apply tau every step  of sim since dart clears all forces afterward;
    # this is for conducting any other per-sim step functions specific to skelHldr (like reapplying assist force)
    @abstractmethod
    def applyTauPriv(self): pass            
    #special considerations for init pose setting - this is called every time skeleton is rest to initial pose
    @abstractmethod
    def _setToInitPosePriv(self): pass
    #setup initial constructions for reward value calculations
    @abstractmethod
    def _setInitRWDValsPriv(self): pass
    #init to be called after skeleton pose is set
    @abstractmethod
    def _postPoseInit(self): pass    
    #functionality necessary before simulation step is executed    @abstractmethod
    def _preStepIndiv(self, a): pass    
    #individual instance class reset functionality called at end of reset
    @abstractmethod
    def _resetIndiv(self, dispDebug): pass    
    #get the state observation from this skeleton - concatenate to whatever extra info we are sending as observation
    @abstractmethod
    def getObs(self): pass    
    #test results, display calculated vs simulated force results at end effector
    @abstractmethod
    def _dispFrcEefResIndiv(self): pass         
    #calculate reward for this agent, see if it is done, and return informational dictionary (holding components of reward for example)
    @abstractmethod
    def calcRewardAndCheckDone(self, obs, debug, dbgStructs): pass
    

#class for skeleton holder specifically for the human needing assistance
class ANASkelHolder(skelHolder):#, ABC):
    #Static list of names of implemented reward components - whenever a new reward component is implemented, put it's name/tag in here
    #add new entries at end of list -MUST to maintain consistency with experiments that did not have these entries
    #WARNING :any removals or re-orderings of this list will invalidate all old trained policy file configurations.
    rwdNames = ['eefDist','action', 'height','footMovDist','lFootMovDist','rFootMovDist','comcop','contacts','UP_COMVEL','X_COMVEL','Z_COMVEL','GAE_getUp','kneeAction','matchGoalPose','assistFrcPen']
                 
    def __init__(self, env, skel, widx, stIdx, fTipOffset, rtLocDofs):          
        skelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset)
        self.name = 'ANA : Agent Needing Assistance'
        self.shortName = 'ANA'
        #dof idxs for root location
        self.rootLocDofs = rtLocDofs

        #set to true to initialize assist force in apply_tau for training, set to false when using robot assistant
        self.setAssistFrcEveryTauApply = False
        #this is # of sim steps in rollout before displaying debug info about reward
        self.numStepsDBGDisp = 201
        #init reward matrix base for action minimization - set joint dof idxs that we don't care as much about for minimization to values between 0 and 1
        self.actPenDofWts = np.identity(self.numActDofs)        

        #which com key to use for height calculations - currently 'com' and 'head'
        #self.comHtKey = 'com' 
        # or 
        self.comHtKey = 'head'
        
        #target goal pose for standing - only set actuated dofs, q and qdot
        self.goalPose=np.zeros(self.numActDofs * 2)

        #whether or not to use best state from previous rollout as initial state on this rollout TODO this needs further development!
        #bestStateMon : 
        #0 : donot monitor best states (default)
        #1 : whether to check for best state after each reward - this is to keep around the best state from the previous rollout to use as next rollout's mean state, or 
        #2 : whether to remember all states and build a CDF of them - set true in ANA if used
        self.bestStateMon = 0
        # self.checkBestState = False
        # #whether or not to use CDF of all previously seen states in CDF to determine initial state - only this or checkBestState should ever be set to true, never both
        # self.buildStateCDF = False  

        #use 3dim jacobians for frc calcs
        self.useLinJacob = True
        
        #sqrt of numActuated dofs, for action reward scaling
        self.sqrtNumActDofs = np.sqrt(self.numActDofs)

        #whether or not we should calculate a force to compensate for eef motion orthogonal to assist force dir
        self.calcCompensateEefFrc = False
        self.orthoCompensateEefFrc = np.array([0.0,0.0,0.0])
        
        #this is list of components used used for reward function - REWRDS TO BE USED MUST BE SPECIFIED IN dart_env_2bot.py file
        self.setRwdsToUse(self.env.rwdCompsUsed)
        #this is whether or not ANA should clamp action proposal before it is used to derive Tau.  normalized environments do this already - might be necessary for non-normed envs
        self.clampA = self.env.clampA
        #this is a list to hold debug messages
        self.msgScrCnstrnts = []

        #self.setRwdWtsVars(names=ANASkelHolder.rwdNames, wts=wtVals, varVals=varVals, tolVals=tolVals, rwdsToUse=rwdFuncsToUse)
    
    #set which reward components ANA will use
    def setRwdsToUse(self, rwdFuncsToUse):
        ##########################################
        #   reward function weights, var/scales and tolerances for each component        
        #specify non-default values in default dict - not using default dict because it might hide bugs where wrong or unknown reward is called for wt/var
        #each foot move dst should be weighted 1/2 of avg all foot dist
        wtVals = defaultdict(lambda:1.0,{'eefDist':10.0,'assistFrcPen':.1, 'kneeAction':10.0, 'matchGoalPose':10.0,'action':1.0,'height':10.0,'footMovDist':10.0,'lFootMovDist':10.0,'rFootMovDist':10.0,'UP_COMVEL':10.0, 'comcop':20.0, 'contacts':10})
        #scales how quickly or shallowly the exponential grows - larger value has more shallow slope of exponent, smaller value has more severe slope - only used for exponential reward
        varVals = defaultdict(lambda:.1,{'action' : (1.0*self.sqrtNumActDofs), 'height':.5,'footMovDist':.2,'lFootMovDist':.2,'rFootMovDist':.2, 'comcop':0.7})
        #non-zero positive value for tol allows for range of values max reward/ min penalty  :must be >= 0
        tolVals = defaultdict(float, {'eefDist':.01, 'comcom':.1,'footMovDist':.1,'lFootMovDist':.1,'rFootMovDist':.1})

        self.setRwdWtsVars(names=ANASkelHolder.rwdNames, wts=wtVals, varVals=varVals, tolVals=tolVals, rwdsToUse=rwdFuncsToUse)
        
    #set target pose - only actuated dofs
    def setGoalPose(self, q):
        if len(q)>self.numActDofs:
            q = q[self.stTauIdx:]
        qdot = np.zeros(self.numActDofs)
        self.goalPose[:] = np.concatenate([q, qdot])

    #initialize descriptors used to describe format of State-action-state' data set
    def _initSaveStateRecPriv(self, endIdx):
        #format of ANA's SAS vec - idxs of components - add per-step reward to each SAS' vec        
        self.SASVecFmt['reward']=(endIdx,endIdx+1)
       
    #return idnvidual holder's obs description for SAS vec saving to disk
    def _SASVecGetDofNamesPriv(self, t):
        assistRes = self.env.getSkelAssistObsNames()
        if len(assistRes) > 0:
            return assistRes
        else :
            return ''
    #return any exta components beyond SAS' to be saved in SAS' csv file
    def _SASVecGetHdrNamesPriv(self): 
        return 'reward'
        

    #set weighting vectors for reward wts, variance/scales and tols - 
    # uses dictionary and not default dict, so that any unrecognized weight values throw an exception - we don't want any unintended values here - they would show some bug in reward code
    #names are the names of the rwd components getting set
    #wts is list of wt mods different than default, varVals and tolVals are same
    def setRwdWtsVars(self, names, wts, varVals, tolVals, rwdsToUse):
        self.rwdWts = {}
        self.rwdVars = {}
        self.rwdTols = {}
        self.comVelDictNameToIDX={'UP_COMVEL':1,'X_COMVEL':0,'Z_COMVEL':2}
        self.comFootDictNameToIDX={'footMovDist':0,'lFootMovDist':1,'rFootMovDist':2}
        #reward functions used for each type of reward - treat as lambdas
        #eval('lambda v: ' + expression)
        self.rwdFuncEvals = {
            'GAE_getUp':self.calcRwdVal_GAEgetUp,
            'eefDist':self.calcRwdVal_EefDist,
            'action':self.calcRwdVal_ActionMin,
            'matchGoalPose': self.calcRwdVal_GoalPoseMatch,
            'assistFrcPen' : self.calcRwdVal_CnstrntFrcPen,
            'kneeAction':self.calcRwdVal_KneeAction,
            'height':self.calcRwdVal_Height,
            'footMovDist':self.calcRwdVal_curFtLoc,
            'lFootMovDist':self.calcRwdVal_curFtLoc,
            'rFootMovDist':self.calcRwdVal_curFtLoc,
            'comcop':self.calcRwdVal_curComCopRew,
            'contacts':self.calcRwdVal_curContactRew,
            'UP_COMVEL':self.calcRwdVal_comVel,
            'X_COMVEL':self.calcRwdVal_comVel,
            'Z_COMVEL':self.calcRwdVal_comVel
        }

        #specify reward components to check
        self.rwdsToCheck = {}
        self.rwdsToCheckAra = []
        for i in range(len(names)) :
            k = names[i]
            self.rwdWts[k] = wts[k]
            self.rwdVars[k] = varVals[k]
            self.rwdTols[k] = tolVals[k] if tolVals[k] >= 0 else 0            
            self.rwdsToCheck[k] = (k in rwdsToUse)
            if (k in rwdsToUse) : 
                self.rwdsToCheckAra.append(k)

    #setup initial constructions for reward value calculations - called before pose is set, 
    # assume skeleton is upright in file -oriented as standing, but not actually upon ground plane (lower than plane), this uses initial skeleton configuration in file
    def _setInitRWDValsPriv(self):
        #this is used only to determine height of COM's above avg foot locs - BEFORE SKELETON IS INITIALIZED/MOVED!!
        avgInitFootLocAra = self.calcAvgFootBodyLoc()
        #dict of bodies to be used for height calcs
        self.comHtBodyDict = {}        
        self.comHtBodyDict['com'] = self.skel
        self.comHtBodyDict['head'] = self.skel.body(self.headBodyName)
        #height of COM above ground plane when standing - raw pose is within ground plane, this is location if standing at starting position of feet
        self.standOnGndCOMLoc = np.array([ 0.5102 ,  0.91379,  0.00104])
        #heights for standing of various body coms over avg foot heigth
        self.standHtOvFtDict = {}
        #vectors for COM to avg Foot loc
        self.StndCOMToFtVecDictList = defaultdict(list)
        #dicts to hold min and max com velocity vectors for the various tracked skel bodies
        self.minCOMVelVals = {}
        self.maxCOMVelVals = {}
    
        #dict of heights standing of various bodies to be used for height calcs
        for k,v in self.comHtBodyDict.items():
            for i in range(len(avgInitFootLocAra)):
                self.StndCOMToFtVecDictList[k].append(v.com() - avgInitFootLocAra[i])
            self.standHtOvFtDict[k] = np.linalg.norm(self.StndCOMToFtVecDictList[k][0])#0.87790457356624751 for body COM
            print('Init StandCOMHeight before pose (skel expected to be upright from skel file) - height of {} com above avg COM foot location : {} : COMCOP vec : {}'.format(k, self.standHtOvFtDict[k],self.StndCOMToFtVecDictList[k]))  
        
        #min and max com vel vals to receive positive rewards - roots of parabola
        #these should not be the same (min==max) ever
        self.minCOMVelVals['com'] = np.array([-.5,.001,-.3])
        self.minCOMVelVals['head'] = np.array([-.5,.001,-.3])

        self.maxCOMVelVals['com'] = np.array([.9,2.75,.3])        
        self.maxCOMVelVals['head'] = np.array([.9,2.75,.3])  

       
    #modify action penalty weight matrix to be val for passed dof idxs
    def buildActPenDofWtMat(self, idxs, val=.1): 
        minWtVals = np.ones(idxs.shape)*val
        i=0
        for idx in idxs :
            self.actPenDofWts[idx,idx] = minWtVals[i]
            i+=1
            
    #return current human fingertip + offset in world coordinates
    def getHumanCnstrntLocOffsetWorld(self, offsetAra=None):
        #want further along negative y axis
        loc = np.copy(self.reachBodyOffset)
        if(offsetAra != None):
            loc += offsetAra
        return self.reachBody.to_world(x=loc)
    
    #ANA does nothing here    
    def _setToInitPosePriv(self):        
        pass
#        #IK's to initial eff position in world, to set up initial pose
#        if(self.initEffPosInWorld is not None):
#            print('skelhldr:_setToInitPosePriv :: {} init eff pose exists, IK to it'.format(self.name))
#            self.skel.set_positions(self.initPose)
#            #IK eff to constraint location ( only done initially to set up robot!)
#            self.IKtoPassedPos(self.initEffPosInWorld,self.skel, self.reachBody)
#            self.initPose = np.copy(self.skel.q) 
    
    #use this to set trajectory ball location self.comHtKey - determines ratio of current height to target/standing height for trajectory location
    #clipped so traj tar doesn't extend past expected trajectory
    def getRaiseProgress(self, useNextStep=False):
        k = self.comHtKey#what body to use for COM height calc
        d = (self.comHtBodyDict[k].com()[1] - self.stSeatHtOvFtDict[k])/self.htDistToMoveDict[k]
        if d < 0:            
            d = 0
        elif d > 1 :
            d = 1             
        return d    

    #called after pose is set - set up initial heights for seated COM and seated head COM
    def _postPoseInit(self):
        #will fail if run for kr5 arm robot
        self.initFootLocAra = self.calcAvgFootBodyLoc()
        #initial seated height above avg foot loc
        self.stSeatHtOvFtDict = {}
        self.htDistToMoveDict = {}
        for k,v in self.standHtOvFtDict.items():
            self.stSeatHtOvFtDict[k] = self.comHtBodyDict[k].com()[1] - self.initFootLocAra[0][1]
            self.htDistToMoveDict[k] = (.9999 * v) - self.stSeatHtOvFtDict[k]#slightly less so that it can be reached
            if(self.htDistToMoveDict[k] <= 0) :
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!! ANASkelHolder::_postPoseInit : Problem with height difference calc in ANASkelHolder {}._postPoseInit() : key:{} seated Height : {} standing height : {}'.format(self.name,k, self.stSeatHtOvFtDict[k],v))
        #print('ANASkelHolder::_postPoseInit : Seated Pose Initial body COM height above avg foot loc {}'.format(self.stSeatHtOvFtDict['com']))    

    def getObs(self):
        #get assist component from environment - different environment files will have different assist components
        assistComp = self.env.getSkelAssistObs(self)
        # #target on traj element
        # tarLoc = self.cnstrntBody.to_world(x=self.cnstrntOnBallLoc)
        state =  np.concatenate([
            self.skel.q,
            self.getClippedSkelqDot(),
            assistComp
            # #need location and force as part of observation!
            # tarLoc,
            # frcObs
        ])
        #print('obs : {}'.format(state))
        #assign COM to state in idxs of root dofs world location
        state[self.rootLocDofs] = self.skel.com()
        return state

    #get-up skel's individual per-reset settings
    def _resetIndiv(self, dispDebug):
        #initial copcom distance (COM projected on ground to COP on ground) - want this to always get smaller
        #set initial COMCOPDist for COMCOP loc and velocity calcs         
        self.COMCOPVecAra = self.getCurCOMCOPData()
        # self.COMCOPDist = vals[0]
        # self.COMCOPVecAra = vals[1]

    #set all relevant COMCOP data used for rewards
    #avg loc of feet coms is treated as COP.  Since we want to match standing COM->COP vector, this should result in standing motion because of gravity
    def getCurCOMCOPData(self, com=None, AVGFootLoc=None):
        #instead of COP, use avg foot loc?
        if AVGFootLoc is None:
            AVGFootLoc = self.calcAvgFootBodyLoc()
        #use COM being tracked by reward
        if com is None:
            com = self.getRewardCOM()
        #vectors for COM to avg Foot loc - compare against these vectors
        comToAvgFtVec = []
        #vectors to all foot locations
        for i in range(len(AVGFootLoc)):
            comToAvgFtVec.append(com - AVGFootLoc[i])
        #ht to average between both feet - not currently using, so don't calculate
        #COMCOPDist = np.linalg.norm(comToAvgFtVec[0])#0.87790457356624751 for body COM
        return comToAvgFtVec # [COMCOPDist,comToAvgFtVec]

    #need to apply tau every step  of sim since dart clears all forces afterward
    def applyTauPriv(self):
        #show ANA's torques
        #self.dbgShowTorques(tau=self.tau,name=self.name)

        #must reapply external force every step as well, if not being assisted - use only when robot not applying force
        if(self.setAssistFrcEveryTauApply) :             
            #print('Apply assist frc : {}'.format(self.desExtFrcVal))
            self.reachBody.set_ext_force(self.desExtFrcVal, _offset=self.reachBodyOffset)
            if self.calcCompensateEefFrc :
                self.reachBody.add_ext_force(self.orthoCompensateEefFrc, _offset=self.reachBodyOffset)                
            
            if (self.setReciprocal):
                self.reachBody.add_ext_force(-1 * self.desExtFrcVal, _offset=self.reachBodyOffset)

    #build dictionary of all components of f=ma for current skeleton having been frwrd simmed with tau
    def _calFrcCmpsAtEefForTau(self, tau, debug):
        skel = self.skel
        M = skel.M
        #torques from mass, coriolis and gravity
        ma = M.dot(skel.accelerations())    
        cg = skel.coriolis_and_gravity_forces()  
        JtPullPInv_new, _, _, JDynInv = self.getEefJacobians(self.useLinJacob, body=self.reachBody, offset=self.reachBodyOffset, calcDynInv=True, M=M)        
        #get torques from contacts
        cntctTau, cntctDict = self.calcCntctTau(self.useLinJacob)
        #build data vals ->cntctTau is external force
        tauArm = tau - ma - cg - cntctTau
        #tauDynArm = 
        frcDbg = []
        frcDbg.append(JtPullPInv_new.dot(tau))#force component at hand solely due to torque
        frcDbg.append(JtPullPInv_new.dot(ma))#force component at hand due to ma
        frcDbg.append(JtPullPInv_new.dot(cg))#force component at hand due to coriolis and gravity
        frcDbg.append(JtPullPInv_new.dot(cntctTau))#force component at hand due to ground collisions reactive forces
        frcDbg.append(JtPullPInv_new.dot(tauArm))
        frcDbg.append(tauArm)
        frcDbg.append(JDynInv.dot(tau))#force component at hand solely due to torque using dynamically consistent inv of J
        frcDbg.append(JDynInv.dot(tauArm))
        if(debug):
            lbl=['tau','ma','cg','cntct','JttauArm','armTorque', 'dyn_tau', 'dync_frc']
            for i in range(len(lbl)) : 
                print('{} : frc : {}'.format(lbl[i],frcDbg[i]))
        return frcDbg, cntctTau, cntctDict
                
    #calculate all force and torque components for current state of skel -TODO verify this
    def _calcCmpFrcAtEefForTau(self, assist, tau, debug):
        frcDbg, cntctTau, cntctDict = self._calFrcCmpsAtEefForTau(tau, debug)
        
        eefPull=frcDbg[0][-3:] - frcDbg[1][-3:] - frcDbg[2][-3:] -frcDbg[3][-3:]
        #unit vector in assistance direction
        assistDir = self.desExtFrcDir

        eefPullTan =  eefPull.dot(assistDir) * assistDir
        eefPullOrtho = eefPull - eefPullTan
        orthoCompensateEefFrc = -eefPullOrtho

        print('_calcCmpFrcAtEefForTau : w/applied assist : {} :\n\teefPull force : {} assist dir : {} assist Frc : {} eef in assist dir : {} ortho to assist : {} applied : {}'.format(assist, eefPull, assistDir, self.desExtFrcVal, eefPullTan, eefPullOrtho, orthoCompensateEefFrc))
        eefPull = frcDbg[-1]
        eefPullTan =  eefPull.dot(assistDir) * assistDir
        eefPullOrtho = eefPull - eefPullTan
        orthoCompensateEefFrc = -eefPullOrtho

        print('_calcCmpFrcAtEefForTau : w/dyn J applied assist : {} :\n\teefPull force : {} assist dir : {} assist Frc : {} eef in assist dir : {} ortho to assist : {} applied : {}'.format(assist, eefPull, assistDir, self.desExtFrcVal, eefPullTan, eefPullOrtho, orthoCompensateEefFrc))

        res={}
        res['eefPull']=eefPull
        res['ortho'] = orthoCompensateEefFrc
        res['eefPullCmps']={'eefPullTan':eefPullTan,'eefPullOrtho':eefPullOrtho}
        res['cntctCmps'] ={'cntctTau':cntctTau,'cntctDict':cntctDict}
        res['frcDbgList']=frcDbg
        return res

    #save sim state for this skel, step sim forward for this skel with current tau only, examine quantities and then restore previous sim state for this skel  
    def _stepFrwrdStepBackEefCompFrc(self, skel, tau, assist, debug):        
        #save ANA's current state
        q = skel.positions()
        dq = skel.velocities()
        ddq = skel.accelerations()      
        #apply control and step forward
        self.reachBody.set_ext_force(assist, _offset=self.reachBodyOffset)    
        skel.set_forces(tau)
        self._stepNoneButMe() 

        resDict = self._calcCmpFrcAtEefForTau(assist, tau, debug)
       
        #restore ANA's state
        skel.set_accelerations(ddq)          
        skel.set_positions(q)
        skel.set_velocities(dq)  
        
        return resDict
    
    #deterine ANA's eef force that is orthogonal to the assistance force
    #DEPRECATED
    def _calcOrthoEefFrc(self, tau, debug=True):
        skel = self.skel
        rb = self.reachBody
        rbOff = self.reachBodyOffset
        #verified that undoing the simulation as I am doing is working
        #frwd sim ANA - first clear current ext force
        assist = np.array([0.0,0.0,0.0])
        resDict = self._stepFrwrdStepBackEefCompFrc(skel, tau, assist, debug)
        #try with added assist force
        #assist=self.desExtFrcVal
        #resDict1 = self._stepFrwrdStepBackEefCompFrc(skel, tau,assist)
             

        #below is for testing
#        self.reachBody.set_ext_force(self.desExtFrcVal, _offset=self.reachBodyOffset) 
#        self.reachBody.add_ext_force(self.orthoCompensateEefFrc, _offset=self.reachBodyOffset)
#        ma1 = skel.M.dot(ddq )    
#        cg1 = skel.coriolis_and_gravity_forces()
#        #eef's contribution to torque
#        eefTau1 = (tau - ma1 - cg1 + cntctTau)
#        #torque cntrol desired to provide pulling force at contact location on reaching hand    
#        JtPullPInv_new1, _, _, _ = self.getEefJacobians(self.useLinJacob, body=self.reachBody, offset=self.reachBodyOffset)
#        #find jacobian at eef and then calculate force at eef
#        #find component of eef force orthogonal to assist force.  apply neg of this to ANA's eef, and penalize torque causing it
#        newEefPull = JtPullPInv_new1.dot(eefTau1)


        #print('eefPull force : {} : assist dir : {} eef in assist dir : {} ortho to assist : {} applied : {}  new pull : {}'.format(eefPull, assistDir, eefPullTan, eefPullOrtho, self.orthoCompensateEefFrc, newEefPull))
#        print('eefPull force : {} assist dir : {} assist Frc : {} eef in assist dir : {} ortho to assist : {} applied : {}'.format(eefPull, assistDir, self.desExtFrcVal, eefPullTan, eefPullOrtho, self.orthoCompensateEefFrc))
        #self.env.pauseForInput("ANASkelHolder::_calcOrthoEefFrc")
    
    #functionality necessary before control application step is executed for the human needing assistance
    def _preStepIndiv(self, a):
        #get measured com values before forward step   
        self.com_b4 = self.getRewardCOM()   
        #set torques  - clamping occurs in normalized env
        self.tau=self.setClampedTau(a, doClamp=self.clampA) 
        #set up aggregation quantities for per-sim-step (Frameskip) reward calcs
        self.initPerSimStepRWDQuantities()
        #calculate force exerted by ANA's end effector that is orthogonal to assistance force
        #use norm of assist force 
        if self.calcCompensateEefFrc :
            self._calcOrthoEefFrc(self.tau)
    

    #returns com of body used in reward calculation
    def getRewardCOM(self):
        return self.comHtBodyDict[self.comHtKey].com()

    #returns exponential calculation with chkVal == 0 returning highest magnitude value
    #chkVal needs to be >= 0, tol needs to be >= 0
    def getRwd_expCalc(self, chkVal, typ, offset):
        wt = self.rwdWts[typ] 
        tol = self.rwdTols[typ]
        var = self.rwdVars[typ]
        #if chkVal is < tol, then this is negative, so result is <= wt(1-offset), assymptotically approaching wt * -offset
        if np.abs(chkVal) < tol  :
            return wt * (1-offset)
        rew = wt * (np.exp((chkVal-tol)/var)  -offset)
        return rew

    #linear reward formulation - chkVal is expected to vary between 0 and 1
    def getRwd_linCalc(self, chkVal, typ):
        if(chkVal <= 0): return 0
        wt = self.rwdWts[typ]
        if(chkVal >= 1.0) : return wt 
        return wt * chkVal
 
    #reward that minimizes action and maximizes left hand COM height
    #uses old formulation of rewards
    def calcMinActionLeftHandHigh(self, numItersDBG=200):
        #actionRew, actMag = self.getRwd_actionMin(constVal=1,optVal=self.a, mult=1.0) 
        ##weighting reach elbow and both knees much less in contribution (to allow for higher magnitude torque on those joints) 
        optVal = self.a
        #either weight optimization by dof or weight every dof equally
        #actSqMag = np.transpose(optVal).dot(self.actPenDofWts.dot(optVal))
        #equal weighting below - inequal weighting didn't seem to make much difference
        actSqMag = optVal.dot(optVal) 
        #needs to be negative to be appropriate - increases magnitude as actSqMag increases
        actionPen = self.getRwd_expCalc(chkVal = (-1*actSqMag), typ='action', offset=1)

        lHandCOM = self.skel.body(self.handBodyNames[0]).com()
        handRew = (10.0*lHandCOM[1]*lHandCOM[1])+5
        reward = -actionPen +  handRew
        # if(self.env.sim_steps % 1 == 0):
        #     print('Only Action And Hand ht Reward @ step {} : {}\tActionPen : -{:.3f}(action sq magnitude : {:.3f})\tHand ht Reward : {:.3f}'.format( self.env.sim_steps, reward, actionPen, actMag, handRew))
         #never done from action, just keep going until 
        done = (self.env.sim_steps >= 200)

        dct = defaultdict(list) 
        #dct : dictionary of lists of reward components, holding reward type as key and reward component and reward vals watched as value in list (idx 0 is reward value, idx 1 is list of rwrd values watched)  
        dbgDict=[dct]
        return reward, done, dbgDict
    
    #process results from reward component check
    #rwdCompsDict : dict holding : current total reward as accumulated value, and a dictionary of dicts of good or bad result completion keyed by rwrd type
    #rwdType : type of this reward
    #rwdComp : reward value
    #rwdValWatched : list of tuples of descriptor and value used by rwd calculation to determine reward
    #succCond, failCond : reward component specific termination conditions (success or failure)
    #debug : whether this is in debug mode or not
    #dct : dictionary of reward types used holding reward components and list of values used to calculate reward, (dict of lists, with first idx == reward and 2nd idx == list of values watched)
    #dbgStructs : dictionary holding lists of values used for debugging
    def procRwdCmpRes(self, rwdStateDict, rwdType, rwdComp, rwdValWatched, succCond, failCond, debug, dbgStructs):
        rwdStateDict['reward'] += rwdComp
        rwdStateDict['isDone']['good'][rwdType]=succCond
        rwdStateDict['isDone']['bad'][rwdType]=failCond
        #checks for success or failure - should be boolean eqs - set to false if no conditions impact done
        rwdStateDict['isGoodDone'] = rwdStateDict['isGoodDone'] or succCond
        rwdStateDict['isBadDone'] = rwdStateDict['isBadDone'] or failCond
        if (debug):
            try :#do this if not numpy vectors in rwdValWatched
                strVal = '{0} rwd/pen: {1:3f} | obs vals: [{2}] | gdone : {3} | bdone :{4} ||'.format(rwdType, rwdComp,'; '.join(['{}:{:3f}'.format(x[0],x[1]) for x in rwdValWatched]),succCond,failCond)
                csvStrVal = '{0},{1:3f},[{2}]'.format(rwdType, rwdComp,';'.join(['{}:{:3f}'.format(x[0],x[1]) for x in rwdValWatched]))
            except : #handle numpy constructs in rwdValWatched
                strVal = '{0} rwd/pen: {1:3f} | obs vals: [{2}] | gdone : {3} | bdone :{4} ||'.format(rwdType, rwdComp,'; '.join(['{}:{}'.format(x[0],x[1]) for x in rwdValWatched]),succCond,failCond)
                csvStrVal = '{0},{1:3f},[{2}]'.format(rwdType, rwdComp,';'.join(['{}:{}'.format(x[0],x[1]) for x in rwdValWatched]))
            dbgStructs['rwdTyps'].append(rwdType)
            dbgStructs['rwdComps'].append(rwdComp)
            dbgStructs['successDone'].append(succCond)
            dbgStructs['failDone'].append(failCond)
            dbgStructs['dbgStrList'][0] += '\t{}\n'.format(strVal)
            dbgStructs['dbgStrList_csv'][0] += '{},'.format(csvStrVal)
            dbgStructs['dbgStrList'].append(strVal)  
            dbgStructs[rwdType] = [rwdComp, rwdValWatched]

    #this reward function calculates reward given in GAE paper for getup - test
    def calcRwdVal_GAEgetUp(self, name):
        #cost = - height from standing ^2  - 1e-5 * ||a||^2
        optVal = self.tau
        #either weight optimization by dof or weight every dof equally
        #actSqMag = np.transpose(optVal).dot(self.actPenDofWts.dot(optVal))
        #equal weighting below - inequal weighting didn't seem to make much difference
        actSqMag = optVal.dot(optVal) 
        actCost = 1e-6 * actSqMag #paper used 1e-5 but had fewer dofs

        bComHeight = self.com_now[1]     
        #determine reward component and done condition
        heightAboveAvgFoot = bComHeight - self.curAvgFootLocAra[0][1] 
        heightDiffStand = heightAboveAvgFoot - self.standHtOvFtDict[self.comHtKey]
        htCost = heightDiffStand * heightDiffStand
        #can add scalar, doing so to keep all rewards positive - never less than -6 or thereabouts
        rewVal = 10 -htCost - actCost

        #95% of standing COM ht means success
        #self.startSeatedCOMHeight #avg center of feet at start when feet on ground
        #ratioToStand = (heightAboveAvgFoot - self.stSeatHtOvFtDict[self.comHtKey])/self.htDistToMoveDict[self.comHtKey]
        ratioToStand = heightAboveAvgFoot /self.standHtOvFtDict[self.comHtKey]
        succCond = ratioToStand >= .95
        failCond = False

        rwdValWatched = [('htCost',htCost),('actCost',actCost), ('htDiffFrStand',heightDiffStand),('heightAboveAvgFoot',heightAboveAvgFoot),('ratioToStand',ratioToStand),('actSqMag',actSqMag)]
        return rewVal, rwdValWatched, succCond, failCond
    
    #penalty term to minimize assistive force - to be used with constraint-based observation/world
    def calcRwdVal_CnstrntFrcPen(self, name):
        cnstrntFrcAra = self.RWDCnstrntFrcVals
        magCFrc = 0
        numFrames = len(cnstrntFrcAra)
        if numFrames > 0 :
            #make size of constraint force in 0 idx of cnstrntFrcAra
            cFrc= np.zeros(cnstrntFrcAra[0].shape)
            for cnstFrc in cnstrntFrcAra:
                # #sclFact scales from 0 to 1, where 0 is the beginning of traj and 1 is near goal - this will ignore reward components early but increase their impact later
                # #sclFact = self.htSclFact
                # #cFrc is force seen on constraint body
                # #penalize avg constraint forces from every sim step per control step
                # cFrc = self.cnstrntBody.skeleton.constraint_forces()
                # cFrcMod = np.array(cFrc)
                # #add assist force on constraint body (meaning we don't penalize desired assist force) - this is 0 unless we are using frc as a component of observation
                # cFrcMod[3:] = cFrc[3:] +self.desExtFrcVal
                # #compensate for mg from constraint force ? - find force only used to pull ANA
                # #cFrcMod[3:] -= self.cnstrntBodyMG
                # #print('CFrc : {} Modded CFrc : {}'.format(cFrc,cFrcMod))
                # #magFrc = np.linalg.norm(cFrc)
                cFrc += cnstFrc
            magCFrc = np.linalg.norm(cFrc)#/(1.0*numFrames)
        #average constraint force exerted through frameskips
        rwdComp = -1.0 * self.rwdWts[name] *  magCFrc
        rwdValWatched = [('numFrames',numFrames), ('avgCFrcMag',magCFrc),('cBodyMG',self.cnstrntBodyMG),('wt',self.rwdWts[name])]
        for i in range(len(cnstrntFrcAra)):
            if ( i % 4 == 0) : 
                stStr = "\n\t"
            else :
                stStr = "|"
            rwdValWatched.append(("{}Step {}".format(stStr,i), 'CnstrFrc : {} '.format(cnstrntFrcAra[i][3:])))

        #self.env.pauseForInput("ANASkelHolder::calcRwdVal_CnstrntFrcPen")
        succCond = False
        failCond = False # magFrcMod > self.mg #pulling by more than ANA weighs is bad, should fail
        return rwdComp, rwdValWatched, succCond, failCond

    #get reward for how close the goal pose (with qdot == 0)  is matched.  Scale by how high COM is, so we don't lay on the ground
    def calcRwdVal_GoalPoseMatch(self, name):
        #sclFact scales from 0 to 1, where 0 is the beginning of traj and 1 is near goal - this will ignore reward components early but increase their impact later
        sclFact = self.htSclFact
        curPose = np.concatenate([self.skel.q[self.stTauIdx:],self.skel.dq[self.stTauIdx:]])
        poseDel = curPose - self.goalPose
        #goalPoseVariance is 0 when matched perfectly - want max value when 0, decreasing value as increases
        goalPoseVariance = poseDel.dot(poseDel)
        rwdComp = self.rwdWts[name] * sclFact * (1.0/(1+goalPoseVariance)) 
        rwdValWatched = [('goalPoseVariance',goalPoseVariance),('htsclFact',sclFact),('wt',self.rwdWts[name])]
        succCond = ( goalPoseVariance < .05)
        failCond = False
        return rwdComp, rwdValWatched, succCond, failCond

#    #temp rwd component to try to penalize hand going far away from target traj
#    def calcRwdVal_EefDistNeg(self, name):
#        #distance of end effector from constraint loc 
#        ballLoc,handLoc = self.getCnstEffLocs()
#        #find vector from target to current location - want to reward very highly distance closer than tolerance (tolerance accounted for in expCalc)
#        eefDist = np.linalg.norm((handLoc - ballLoc))
#        #max dist to receive a positive reward from this component
#        distThresh = .25
#        wt = self.rwdWts[name] 
#        #tol = self.rwdTols[name]       
#        #max rwrd when eefDist <= tol; 0 rwrd when eefDist >= thresh (.5?)
#        #val will always be between (0, 1), where 0 means eefDist == thresh, 1 means eefDist == tol
#        val = (distThresh - eefDist)/distThresh
#        rwdComp = wt * (val*val*val)
#
#        succCond = False
#        failCond = eefDist > 2.0 * distThresh#too far away this should just stop the sim
#        rwdValWatched = [('eefDist',eefDist, 'rwdComp', rwdComp)]
#        return rwdComp, rwdValWatched, succCond, failCond
    #reward component for keeping hand by target location

    def calcRwdVal_EefDist(self, name):
        #distance of end effector from constraint loc 
        ballLoc,handLoc = self.getCnstEffLocs()
        #find vector from target to current location - want to reward very highly distance closer than tolerance (tolerance accounted for in expCalc)
        eefDist = np.linalg.norm((handLoc - ballLoc))
        wt = self.rwdWts[name]
        tol = self.rwdTols[name]
        rwdComp = wt * (tol - eefDist) #if eefDist< tol, this is positive

        succCond = False
        failCond = eefDist > .5#too far away (half a meter) this should just stop the sim
        rwdValWatched = [('eefDist',eefDist),('rwdComp', rwdComp),('wt',wt), ('tol',tol)]
        return rwdComp, rwdValWatched, succCond, failCond

    # #reward component for keeping hand by target location
    # def calcRwdVal_EefDistOld(self, name):
    #     #distance of end effector from constraint loc 
    #     ballLoc,handLoc = self.getCnstEffLocs()
    #     #find vector from target to current location - want to reward very highly distance closer than tolerance (tolerance accounted for in expCalc)
    #     eefDist = np.linalg.norm((handLoc - ballLoc))
    #     #max dist to receive a non-zero reward from this component
    #     distThresh = .25
    #     wt = self.rwdWts[name] 
    #     tol = self.rwdTols[name]

    #     if (eefDist >= distThresh) : 
    #         rwdComp = 0.0
    #     elif (eefDist <= tol):
    #         rwdComp = wt
    #     else :#dist is < thresh and > tol
    #         #max rwrd when eefDist <= tol; 0 rwrd when eefDist >= thresh (.5?)
    #         #val will always be between (0, 1), where 0 means eefDist == thresh, 1 means eefDist == tol
    #         val = (distThresh - eefDist)/(distThresh - tol) 
    #         rwdComp = wt * (val*val)

    #     succCond = False
    #     failCond = eefDist > 2.0 * distThresh#too far away this should just stop the sim
    #     rwdValWatched = [('eefDist',eefDist, 'rwdComp', rwdComp)]
    #     return rwdComp, rwdValWatched, succCond, failCond

    def calcRwdVal_ActionMin(self, name):
        ##weighting reach elbow and both knees much less in contribution (to allow for higher magnitude torque on those joints) -weigthing knees and elbow has minimal effect with current weights - perhaps need to decrease weigthing values
        #either minimize torque or action from controller
        optVal = self.a
        #either weight optimization by dof or weight every dof equally
        #actSqMag = np.transpose(optVal).dot(self.actPenDofWts.dot(optVal))
        #equal weighting below - inequal weighting didn't seem to make much difference, probably related to scale of overall reward value
        actSqMag = optVal.dot(optVal) + .00001
        #making negative causes decaying curve, setting offset to 0 causes results to vary 0<res<-1
        rwdComp = self.getRwd_expCalc(chkVal=(-1*actSqMag), typ=name, offset=0.0)
        #no instant success/fail conditions predicated on action
        succCond = False
        failCond = False
        
        rwdValWatched = [('actSqMag',actSqMag), ('wt',self.rwdWts[name]), ('tol',self.rwdTols[name]), ('var',self.rwdVars[name] )]
        return rwdComp, rwdValWatched, succCond, failCond
    
    def dbgTestRwdVals(self, name, minval, maxval, offset):
        vals = np.linspace( minval, maxval, num=51, endpoint=True)
        rwds = []
        for x in vals:
            rwds.append(self.getRwd_expCalc(chkVal=(-1*x), typ=name, offset=offset))
        return vals,rwds
    
    def dbgTestHtRwdVals(self):
        maxVal = self.StndCOMToFtVecDictList[self.comHtKey][0][1]#idx 0 is avg foot vec to test com, idx 1 is y component
        vals = np.linspace( 0.0, maxVal, num=51, endpoint=True)
        rwds = []
        for x in vals:
            ratioToStand = (x - self.stSeatHtOvFtDict[self.comHtKey])/self.htDistToMoveDict[self.comHtKey]            
            rwds.append(self.getRwd_linCalc(chkVal=ratioToStand, typ='height'))
        return vals,rwds

    def dbgTestCOMCOPRwdVals(self):     
        new_COMCOPVecAra = self.getCurCOMCOPData(com=self.com_now, AVGFootLoc=self.curAvgFootLocAra)
        standCOMToAvgFtVec = self.StndCOMToFtVecDictList[self.comHtKey][0] #vector from feet to head com
        htStanding = self.standHtOvFtDict[self.comHtKey]

        #dot prod will vary between 0 and 1, with 1 being best and 0 being worst
        comCopDotProd = new_COMCOPVecAra[0].dot(standCOMToAvgFtVec)/(htStanding*htStanding)
        rwdComp = self.rwdWts['comcop'] * comCopDotProd * comCopDotProd

        succCond = comCopDotProd > .95
        failCond = False

        rwdValWatched=[('comCopDotProd',comCopDotProd), ('wt',self.rwdWts['comcop'])]
        return rwdComp, rwdValWatched, succCond, failCond

    #calc reward for positive knee action - want to give a small reward for using the knees
    def calcRwdVal_KneeAction(self, name):
        #sclFact scales from 0 to 1, where 0 is the beginning of traj and 1 is near goal - this will ignore reward components early but increase their impact later
        sclFact = self.htSclFact
        lKneeA = self.a[self.kneeDOFActIdxs[0]]
        rKneeA = self.a[self.kneeDOFActIdxs[1]]
        #diff = (lKneeA - rKneeA)
        #negative action value is pushing knee straight

        #optVal = self.a[self.kneeDOFActIdxs]
        #action may not be limited to +/- 1 - this may be removed when we move away from normalized environments
        #negative knee action is pushing, positive knee action is pulling.  
        #knee action - varies from 0 to 1, with 1 meaning both knees are pushing with a mag of -1, and 0 means both are pulling with a mag of 0
        
        #kneeAct = (-1 * (np.sum(optVal)) +2.0)/4.0
        #penalize difference 
        #kneeDiff = (diff * diff)
        kneeAct = (-1.0 * (lKneeA + rKneeA))# - kneeDiff 
        rwdComp = self.rwdWts[name] * sclFact * kneeAct

        succCond = False
        failCond = False #sclFact*kneeAct < .01    #means both knees are pulling hard - maybe check to see that both knees are pushing?

        rwdValWatched=[('htsclFact',sclFact),('kneeActVal',kneeAct),('leftKnee_A',lKneeA), ('rightKnee_A',rKneeA), ('wt',self.rwdWts[name])]#, ('kneeDiff',diff)]
        return rwdComp, rwdValWatched, succCond, failCond

    def calcRwdVal_Height(self, name):   
        #determine reward component and done condition
        heightAboveAvgFoot = self.com_now[1] - self.curAvgFootLocAra[0][1] #self.startSeatedCOMHeight #center of COP on ground/feet

        #find ratio of heightabovefoot/ standAboveFoot 
        #ratioToStand = heightAboveAvgFoot/self.standHtOvFtDict[self.comHtKey]
        #find ratio of heightabovefoot-seatedaboveFt / standAboveFoot-seatedAboveFoot == ratio of how far we have to go
        ratioToStand = (heightAboveAvgFoot - self.stSeatHtOvFtDict[self.comHtKey])/self.htDistToMoveDict[self.comHtKey]
        #negative value denoting height to go to be standing - greater magnitude means further to go, max is 0
        heightDiffStand = heightAboveAvgFoot - self.standHtOvFtDict[self.comHtKey]
        #gives 0 reward if ratioToStand < 0 - means moving backwards.  might need to do this to get up, but doesn't get reward for it
        heightRew = self.getRwd_linCalc(chkVal=ratioToStand, typ=name)
        #95% of standing COM ht means success
        succCond = ratioToStand >= .95
        failCond = False
        
        rwdValWatched = [('htDiffFrStand',heightDiffStand),('heightAboveAvgFoot',heightAboveAvgFoot),('ratioToStand',ratioToStand), ('wt',self.rwdWts[name]), ('tol',self.rwdTols[name]), ('var',self.rwdVars[name] )]
        return heightRew, rwdValWatched, succCond, failCond

    #calculate com velocity reward measure based on passed idx
    def calcRwdVal_comVel(self, name):
        idx = self.comVelDictNameToIDX[name]
        comVelComp = self.bComVel[idx]
        htKey = self.comHtKey
        rwdComp = self.rwdWts[name] * self.calcRwdRootMethod(comVelComp,self.minCOMVelVals[htKey][idx], self.maxCOMVelVals[htKey][idx])
        
        succCond = False
        #?com moving too fast in certain directions should fail?
        failCond = False
        
        rwdValWatched=[(name,comVelComp), ('dir(idx)', idx), ('minCOMVel', self.minCOMVelVals[htKey][idx]), ('maxCOMVel', self.maxCOMVelVals[htKey][idx]), ('wt',self.rwdWts[name])]
        return rwdComp, rwdValWatched, succCond, failCond

    #calculate avg foot distance from start reward
    def calcRwdVal_curFtLoc(self, name):
        idx = self.comFootDictNameToIDX[name]
        footMovDist = np.linalg.norm(self.curAvgFootLocAra[idx] - self.initFootLocAra[idx])
        #needs to be negative foot move distance for exponential reward, otherwise will try to move feet as far as possible
        rwdComp = self.getRwd_expCalc(chkVal=(-1 * footMovDist), typ=name, offset=0.0)
        
        succCond = False
        #feet have moved past x m from start - instant fail
        failCond = footMovDist > 2

        rwdValWatched=[(name,footMovDist)]
        return rwdComp, rwdValWatched, succCond, failCond

    #calculate contact-based rewards/penalties
    #TODO best way to monitor this?  - this needs to be determined per frame skip
    def calcRwdVal_curContactRew(self, name):
        cntctDictAra = self.RWDContactDicts
        rwdComp = 0
        numFrames = len(cntctDictAra)
        if numFrames > 0 :
            #for all contact dicts saved
            for contactDict in cntctDictAra:        
                if(contactDict['footGroundContacts'] != 0):
                    #TODO calculate contact contributions to reward if being used - keep from kicking bot - maybe not necessary if using feet distance reward/penalty
                    #contacts to be rewarded
                    gCntcts = contactDict['GoodContacts']
                    #contacts to be avoided
                    bCntcts = contactDict['BadContacts']
                    #rwdComp = max(0,sclVal *(gCntcts-bCntcts))
                    rwdComp += self.rwdWts[name] * (gCntcts-bCntcts)
                    #penalize feet off ground
            rwdComp /= 1.0*numFrames        
        succCond = False
        failCond = False

        rwdValWatched=[("Step {}".format(i),[('cDict:{}'.format(k),v) for k,v in cntctDictAra[i].items()]) for i in range(len(cntctDictAra)) ]
        rwdValWatched.append( ('wt',self.rwdWts[name]))
        return rwdComp, rwdValWatched, succCond, failCond

    # #calculate reward based on COM-COP distance compared to standing height
    # def calcRwdVal_curComCopRew_old(self, name):
    #     # tolerance represents area around COP -> find if distance is within tolerance value COMCOPDist-tol == 0 will yield 0 penalty
    #     #ratio of distance from COM COP to height when standing - larger this is, the further we are from standing over COP, varies from 0->1
    #     COMCOPDistRatio = self.new_COMCOPDist/self.standHtOvFtDict[self.comHtKey]
    #     #ratio goes from 0 -> 1; 0 is best, 1 is worst
    #     #COMCOP_DistPen = self.getRwd_expCalc(chkVal= -COMCOPDist, typ='comcop', offset=0)  
    #     rwdComp = self.rwdWts[name] * (1.0 - COMCOPDistRatio)**2.0

    #     succCond = False
    #     failCond = False

    #     rwdValWatched=[('comcopDist',self.new_COMCOPDist), ('ratioOfDistToStandHt',COMCOPDistRatio)]
    #     return rwdComp, rwdValWatched, succCond, failCond
    
    
    #calculate reward based on maximizing COM-COP vector similarity to initial COM-COP vector - use dot product
    #calculate over control step application (no need to consider individual sim steps)
    def calcRwdVal_curComCopRew(self, name):
        new_COMCOPVecAra = self.getCurCOMCOPData(com=self.com_now, AVGFootLoc=self.curAvgFootLocAra)
        #COM COP dot prod measures similarity between current and target com->avg foot loc vectors
        #dictionary per body of com vectors to foot com locs (idx 0 is avg of both foot loc)
        standCOMToAvgFtVec = self.StndCOMToFtVecDictList[self.comHtKey][0]
        htStanding = self.standHtOvFtDict[self.comHtKey] #already calculated to be ||standCOMToAvgFtVec||_2
 
        #dot prod will vary between 0 and 1, with 1 being best and 0 being worst
        #idx 0 is 
        comCopDotProd = new_COMCOPVecAra[0].dot(standCOMToAvgFtVec)/(htStanding*htStanding)
        rwdComp = self.rwdWts[name] * comCopDotProd * comCopDotProd

        succCond = comCopDotProd > .95
        failCond = False

        rwdValWatched=[('comCopDotProd',comCopDotProd),('wt',self.rwdWts[name])]
        #set for next control step so not necessary to recalculate
        self.COMCOPVecAra = new_COMCOPVecAra        
        return rwdComp, rwdValWatched, succCond, failCond

    #set up entities to use to aggregate per simstep data points used for reward calc, for multi-sim-step-per-control step calculation
    #rwdNames = ['eefDist','action', 'height','footMovDist','lFootMovDist','rFootMovDist','comcop','contacts','UP_COMVEL','X_COMVEL','Z_COMVEL','GAE_getUp','kneeAction','matchGoalPose','assistFrcPen']
    def initPerSimStepRWDQuantities(self):
        if (self.rwdsToCheck['contacts']):
            self.RWDContactDicts = []
        if (self.rwdsToCheck['assistFrcPen']):
            self.RWDCnstrntFrcVals = []
            self.msgScrCnstrnts = []
       
    #calculate reward based on what rwrd funcs are specified
    def calcRewardAndCheckDone(self, obs, debug, dbgStructs):

        rwdComps = {}
        rwdComps['reward'] = 0.0
        rwdComps['isDone'] = {'good':{}, 'bad':{}}

        rwdComps['isGoodDone'] = False
        rwdComps['isBadDone'] = False

        #many reward components use foot location and current com and scale of current height compared to standing (0-1)
        #precalc 
        self.htSclFact = self.getRaiseProgress()
        self.com_now = self.getRewardCOM()#com of only tracked body   #self.comHtBodyDict[self.comHtKey].com()
        self.curAvgFootLocAra = self.calcAvgFootBodyLoc()#array of 3 entries
        #calculate tracked com velocity if using reward that includes those terms - over entire control step application
        if (self.rwdsToCheck['UP_COMVEL']) or (self.rwdsToCheck['X_COMVEL']) or (self.rwdsToCheck['Z_COMVEL']):
            self.bComVel = (self.com_now - self.com_b4) / self.perFrameSkip_dt
        
        #reward function calculation - for each specified reward component calculation, execute proper function
        for rwdCompName in self.rwdsToCheckAra :
            rwdComp, rwdValWatched, sc, fc = self.rwdFuncEvals[rwdCompName](rwdCompName)
            self.procRwdCmpRes(rwdComps, rwdType=rwdCompName, rwdComp=rwdComp, rwdValWatched=rwdValWatched, succCond=sc, failCond=fc, debug=debug, dbgStructs=dbgStructs)      
        
        #if counting # of frames, check against done here
        done = rwdComps['isGoodDone'] or rwdComps['isBadDone']
        #can be both good and bad, bad takes precedence
        if rwdComps['isBadDone']:
            rwdComps['reward'] -= 1000.0 #want to make reward much more negative
        elif rwdComps['isGoodDone'] : 
            rwdComps['reward'] += 1000.0 #extra reward for success

        #if (debug):
        #set display if debug, clear otherwise
        self.env.setRewardDisplay(rwdComps, done, dbgStructs, debug)
        #send constraint debug messages
        self.env.setConstraintDisplay(self.msgScrCnstrnts, debug)
            # numSteps = self.env.sim_steps
            # assistCmp = self.env.getSkelAssistObs(self)
            # print('Step : {} : AssistComponent {} : Reward : {:.7} Done : {}'.format(numSteps,assistCmp, rwdComps['reward'], done)),
            # print('{}'.format(dbgStructs['dbgStrList'][0]))
            
        self.mostRecentRwd = rwdComps['reward']
        
        #self.env.pauseForInput("ANASkelHolder::calcRewardAndCheckDone")
        #print ('reward : {}\tdone :{}'.format(reward, done))
        return rwdComps['reward'], done, dbgStructs     
    
    #calc end effector force
    def tmpCalcEEF_Frc(self):
        skel = self.skel
        ma = skel.M.dot(skel.ddq )    
        cg = skel.coriolis_and_gravity_forces() 
        cnstrntF = skel.constraint_forces()
        #torque cntrol desired to provide pulling force at contact location on reaching hand    
        JtPullPInv_new, _, _, _ = self.getEefJacobians(self.useLinJacob, body=self.reachBody, offset=self.reachBodyOffset)
        
        frcD = {}      
        t=self.tau        
        totEEfTau = ma + cg + cnstrntF - t
        totEEfFrc = JtPullPInv_new.dot(totEEfTau)             
        
        return totEEfFrc    
    
    #based upon which reward functions are being used, aggregate per-sim step (frameskip) quantities, so they can each be considered in reward function calculation
    def aggregatePerSimStepRWDQuantities(self, debug):
        #if contacts, aggregate contact profile
        if (self.rwdsToCheck['contacts']):
            contactDict = self.calcAllContactDataNoCOP()
            self.RWDContactDicts.append(contactDict)
        #if assistFrcPen, aggregate all constraint forces from each sim frame
        if (self.rwdsToCheck['assistFrcPen']):
            cFrc = self.cnstrntBody.skeleton.constraint_forces()            
            #meCFrc = self.tmpCalcEEF_Frc()
            cFrcMod = np.array(cFrc)
            #subtract assist force from constraint force (meaning we don't penalize desired assist force) - this is 0 unless we are using frc as a component of observation
            #compensate for mg from constraint force -> find force only used to "satisfy constraint" -> pulling force
            perStepCnstrntMoveExtFrc = self.env.getPerStepMoveExtForce()
            dispStr = 'Ext Frc Moving Body (ma) : {} | CFrc From Dart : {} | CnstrntBody MG : {}'.format(perStepCnstrntMoveExtFrc,cFrc[3:],self.cnstrntBodyMG)
            self.msgScrCnstrnts.append(dispStr)
            if(debug):
                print("skelHolders::aggregatePerSimStepRWDQuantities : {}".format(dispStr))
            cFrcMod[3:] = cFrc[3:] - self.desExtFrcVal + self.cnstrntBodyMG #- perStepCnstrntMoveExtFrc
            #magFrc = np.linalg.norm(cFrcMod)
            self.RWDCnstrntFrcVals.append(cFrcMod)
            #print('ANAs eef frc : {} balls cfrc : {}'.format(meCFrc,cFrc))
        
            #print("aggregate constraint forces")
        
    #add holder-instance specific state values to save to file - return a list of values
    def _addStateVecToSASlistPriv(self):
        return [self.mostRecentRwd]        
    
    #individual skeleton handling for calculating post-step dynamic state dictionary
    def _bldPstFrcDictPriv(self, frcD, calcCnstrntVals):   
        if (calcCnstrntVals):
            cnstrntFrc = self.skel.constraint_forces()
            #monitor constraint forces to see if force seen by human is appropriate pull force        
            frcD['cnstf'] = cnstrntFrc
            #store JtPullPInv_new in dict - get constraint forces observed at eef
            frcD['jtDotCnstFrc'] = frcD['JtPullPInv_new'].dot(cnstrntFrc)        
            #total pull force w/constraint force
            frcD['totPullFrcNoCnstrnt'] = frcD['jtDotTau'] - frcD['jtDotCGrav'] - frcD['jtDotMA']
            #total pull force of listed forces 
            frcD['totPullFrc'] = frcD['totPullFrcNoCnstrnt']  + frcD['jtDotCnstFrc']
            #target force to be seen at eef is -1 * self.use force
            frcD['targetEEFFrc'] = -1*self.useForce
            #save state ref - get this elsewhere?s
            frcD['state']=self.getObs()
        else :
            #total pull force of listed forces (doesn't include contact force calc if present)
            frcD['totPullFrc'] = frcD['jtDotTau'] - frcD['jtDotCGrav'] - frcD['jtDotMA']
        
        return frcD
    
    #test results, display calculated vs simulated force results at end effector
    #sumTtl : total force seen at eef by ANA
    def _dispFrcEefResIndiv(self):  
        #print('\t\tEEF w/o Cnstrnt F : \t{}'.format(self.frcD['totPullFrcNoCnstrnt'][-3:]))
        pass


#abstract base class for helper robot (full body or fixed arm)
class helperBotSkelHolder(skelHolder, ABC):
                 
    def __init__(self, env, skel, widx, stIdx, fTipOffset):          
        skelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset)
        # number of optimization runs for optimization process (overridden in instancing class)
        self.numOptIters = 10
        #the holder of the human to be helped - 
        self.helpedSkelH = None
        #default to 0 contact dimensions
        self.numCntctDims = 0
        #to hold constraint body
        self.cnstrntBody = None
        #limit for qdot proposals only if box constraints are set to true
        self.qdotLim = 5.0   
        #set in child classes
        self.bndCnst = -1             
        
        ########################
        ##  runtime flags
        #do not use randomized initial state - use init state that has been evolved & IKed to be in contact with constraint
        self.randomizeInitState = False
        #Monitor decision vals seen
        self.monitorOptGuess = True
        #Monitor generated force at pull hand
        self.monitorGenForce = True
        #if matching a pose this calls function initOptPoseData
        self.doMatchPose = False
        #whether or not to include box constraints for allowable values in optimization
        self.useBoxConstraints = False
        #whether to use operational space controller
        self.useOSControl = False
        #the force being used for optimization process
        numDim = 3 if (self.useLinJacob) else 6
        self.useForce = np.zeros(numDim)
        #sets dimension of optimization vector : function lives in child class
        self.nOptDims = self.getNumOptDims()
        #this is mimic skeleton used to perform calculations without needing to break constraints on actual bot.  
        # set in simulation
        self.mimicBot = None 
        #old dq, to calculate ddq
        self.oldDq = np.zeros(self.ndofs)
      
    #helper bot would only use this if it was solving its action using RL
    def _setInitRWDValsPriv(self):
        pass
        
    #need to apply tau every step  of sim since dart clears all forces afterward - this is any class-specific actions necessary for every single sim step/frame
    def applyTauPriv(self):
        pass    
    
    #initialize descriptors used to describe format of State-action-state' data set - TODO implement this for individual helper bot configs
    def _initSaveStateRecPriv(self, endIdx):
        pass
    #add holder-instance specific state values to save to file - returns a list of values
    def _addStateVecToSASlistPriv(self):
        return []
    #return individual holder's obs description for SAS vec saving to disk
    def _SASVecGetDofNamesPriv(self, t):
        return ''
    #return any exta components beyond SAS' to be saved in SAS' csv file
    def _SASVecGetHdrNamesPriv(self): 
        return 'reward'

    
    #helper bot will IK to appropriate world position here, and reset init pose    
    def _setToInitPosePriv(self):
        #IK's to initial eff position in world, to set up initial pose, if solving either IK, IK-SPD, or dynamics for helper
        if(self.initEffPosInWorld is not None) and (self.env.botIKtoInitPose()):
            print('helperBotSkelHolder::_setToInitPosePriv :: {} init eff pose exists, IK to it'.format(self.name))
            #init pose here is base configuration to start IK from
            self.skel.set_positions(self.initPose)
            #IK eff to constraint location ( only done initially to set up robot!)
            self.IKtoPassedPos(self.initEffPosInWorld,self.skel, self.reachBody)
            #after IK'ing successfully, set init pose to be resultant state of bot
            self.initPose = np.copy(self.skel.q)        
    
    #called after initial pose is set 
    def _postPoseInit(self):
        self.initHelperBot()
        
    #this is the skelHolder for the assisting robot, so set the human to be nonNull
    def setHelpedSkelH(self, skelH):
        self.helpedSkelH = skelH
        #reset observation dim to match this skel and human skel's combined obs
        self.setObsDim(self.obs_dim + skelH.obs_dim)   

    #robot uses combined human and robot state, along with force observation (from human)
    def getObs(self):
        stateHuman = self.helpedSkelH.getObs()
        #stateHuman has current force as part of observation, need to replace with local desired force TODO    
        state =  np.concatenate([
            self.skel.q,
            self.getClippedSkelqDot(),
            stateHuman,
        ])            
        
        return state        
    
    #####################################################
    #       optimization initialization     

    #method to initialize robot, after pose set, including optimization parameters and variables used in all assisting bot skel holders 
    #one-time init
    def initHelperBot(self):  
        #build index range lists
        self.initDecVarRanges()    

        #for optimization controller
        if(self.useBoxConstraints):
            #build box bounds for values for optiziation
            self.buildOptBounds()

        self.initQdotWts()
        
        #for pose matching
        if(self.doMatchPose):
            self.initOptPoseData()

        #for derivs for accel term and tau
        self.dofOnes = np.ones(self.ndofs)
        
        #negative because subttracted Tau in MA equality constraint
        self.tauDot = self.dofOnes * -1
        self.tauDot[0:self.stTauIdx] *=0
        #matrix of tauDot, for vector constraint gradient
        self.tauDotMat = np.diag(self.tauDot)
         #tolerance for F=MA vector constraint eq -> 1 constraint calc per dof
        self.cnstTolMA = np.ones(self.ndofs) * 1e-8        
        #tolerance for scalar constraints and/or obj func
        self.cnstTolSclr = 1e-6                
        #optimize any individual components
        self.initOptIndiv()
        
#        if (self.useBoxConstraints):
#            #set initial guess for velocity to be all ones TODO sample between box bounds
#            self.nextGuess=np.random.uniform(self.lbndVec, self.ubndVec, self.nOptDims)     
#        else :
        self.nextGuess=np.zeros(self.nOptDims) 
        #set root dof initial guess of taus to be 0
        stTauRoot=(self.ndofs+self.numCntctDims)
        self.nextGuess[stTauRoot:(stTauRoot+self.stTauIdx)] = np.zeros(self.stTauIdx)
        #these are to hold last iteration values, in case opt throws exception
        self.prevGuess = np.copy(self.nextGuess)
        self.currGuess = np.copy(self.nextGuess)  
        #monitor all decision values seen so far
        if(self.monitorOptGuess):
            self.minMaxGuessDict = {} #self.nOptDims
            self._resetMinMaxMonitors(self.minMaxGuessDict,self.nOptDims)
            
        #create optimizer - uses default LN_COBYLA unless alg is passed as arg
        self.initOptimizer()           
        
        #for operational space controller : gains values from dart example
        Kp = 5.0
        Kd = 0.01
        self.osc_Kp = np.diag(np.ones(6)*Kp)
        self.osc_Kd = np.diag(np.ones(self.ndofs)*Kd)
        
        
    #method to initialize optimization parameters and variables specific to skeleton
    @abstractmethod
    def initOptIndiv(self):
        pass  
            
    #set up variables for which dof idxs we want to match if using pose matching objective
    def initOptPoseData(self):
        #build weighting vector/matrix for pose matching
        self.poseMatchWtsVec = np.zeros(self.ndofs)
        #set up self.optPoseUseIDXs array
        self.optPoseUseIDXs = self.initOptPoseData_Indiv()
        #actual pose values - use initial pose values as matching targets
        self.matchPose = self.initPose[(self.optPoseUseIDXs)]
        #only dofs of interest should be matched,other dofs have 0 wt
        self.poseMatchWtsVec[self.optPoseUseIDXs] = 1.0
        self.poseMatchWts = np.diag(self.poseMatchWtsVec)
        
        #TODO magic number, no idea if this is correct choice - tightness of pose matching
        self.kPose = 100
        self.tSqrtKpose = 2 * sqrt(self.kPose)
    
    #set up variables for which dof idxs we want to match if using pose matching objective
    @abstractmethod
    def initOptPoseData_Indiv(self):
        pass
    
    #initialize lists of idxs for individual decision variables in x vector
    def initDecVarRanges(self):
        #idx aras to isolate variables in constraint and optimization functions
        endIdx = self.ndofs
        self.qdotIDXs = np.arange(0, endIdx)
        #if uses contacts, put contacts here.
        if(self.numCntctDims > 0):
            stIdx = endIdx
            endIdx = stIdx + self.numCntctDims
            self.fcntctIDXs = np.arange(stIdx,endIdx)
        stIdx = endIdx
        endIdx = stIdx + self.ndofs
        self.tauIDXs = np.arange(stIdx, endIdx)
        
        
    #set values for decision variable bounds
    #upIdxAra : idxs of up direction for contacts, if contacts are decision values
    #idxTorqueSt : idx where torques start
    def buildOptBounds(self):
        #set bounds for optimizer, if necessary/useful
        #basic box constraint bounds for optimization
        
        self.lbndVec = np.ones(self.nOptDims)*-self.bndCnst
        self.ubndVec = np.ones(self.nOptDims)*self.bndCnst

        #limit qdot vals
        self.lbndVec[self.qdotIDXs] = np.ones(self.qdotIDXs.shape)*-self.qdotLim
        self.ubndVec[self.qdotIDXs] = np.ones(self.qdotIDXs.shape)*self.qdotLim
        
        tolVal= 1e-12
        #find where torques start
        idxTorqueSt = self.tauIDXs[0]

        #root (uncontrolled) dofs should be very close to 0 (# of root dofs held in self.stTauIdx)
        if (self.stTauIdx > 0):        
            self.lbndVec[idxTorqueSt:(idxTorqueSt+self.stTauIdx)] = -tolVal
            self.ubndVec[idxTorqueSt:(idxTorqueSt+self.stTauIdx)] = tolVal


    #build list of tuples of idx and multiplier to modify wts array/wts matrix for qdot weighting
    @abstractmethod
    def buildQdotWtModAra(self):
        pass

    #build wts vector and matrix for velocity matching objective
    #modAra is ara of tuples of idx and multiplicative mod for that idx
    def initQdotWts(self):
        modAra = self.buildQdotWtModAra()
        #obj func deriv for sqrt of sqrd obj func == weights multiplier for obj func using velocities
        self.qdotWtsVec = np.ones(self.ndofs)
        #don't look at root dof velocities -WRONG
        #self.qdotWtsVec[0:6]*=0
        #look at waist/spine velocities
        for tup in modAra:
            self.qdotWtsVec[tup[0]] *= tup[1]
            
        #qdot weights for velocity matching-based objective func
        #obj func deriv for sqrd obj funcs
        self.qdotWts = np.diag(self.qdotWtsVec)       
    #instance optimizer and set optimizer box constraints/bounds
    def initOptimizer(self, alg=nlopt.LN_COBYLA):
        #create optimizer
        #only LD_MMA and LD_SLSQP support nonlinear inequality constraints
        #only LD_SLSQP supports nonlinear equality cosntraints
        #
        self.optimizer = nlopt.opt(alg, self.nOptDims)  #NLOPT_LN_COBYLA is default - gradient free method      
        if(self.useBoxConstraints):
            #set min bounds
            self.optimizer.set_lower_bounds(self.lbndVec)
            #set max bounds
            self.optimizer.set_upper_bounds(self.ubndVec)   
        
    #method to determine # of optimization calculation parameters (dim of decision vector) used for the optimzer for this skel holder    
    @abstractmethod
    def getNumOptDims(self):
        pass   
    
    #calculate pseudo PM inverse of M, with final dim of d x d
    #MPinv = Mt * (M * Mt + lamda* I)^-1
    #same result as numpy.linalg.pinv() in limit of lambda
    def calcPInv(self, M, lam=0.000125):
        #lambda * I - size is # rows of M x # rows of M
        nRows = np.size(M, axis=0)
        tmpIdent = np.diag(np.ones(nRows)*lam)
        MTrans = np.transpose(M)        
        return MTrans.dot(np.linalg.inv(M.dot(MTrans) + tmpIdent)) 
    
    ####################################################
    # per-step body values TODO
    
    #get jacobian deriv for eef
    def getEffJdot(self, useLinJacob, debug=False):
        if (useLinJacob) :
            return self.reachBody.linear_jacobian_deriv(offset=self.reachBodyOffset) 
        else :
            djAng = self.reachBody.angular_jacobian_deriv()
            djLin = self.reachBody.linear_jacobian_deriv(offset=self.reachBodyOffset) 
            if(debug):
                print('shape of djAng : {} shape of djLin : {}'.format(djAng.shape, djLin.shape))
            #derivative and pinv of derivative
            return np.concatenate((djAng, djLin), axis=0)
        
    
    #called from prestep, set important values used to track trajectory ball and retain original state info before step
    def setPerStepStateVals(self, calcDynQuants):        
        #tracking body curent location and last location, for vel vector
        self.trackBodyLastPos = self.trackBodyCurPos
        #world position of tracked position on ball
        self.trackBodyCurPos = self.cnstrntBody.to_world(x=self.cnstrntOnBallLoc) #self.cnstrntBody.com()
        self.trackBodyVel = (self.trackBodyCurPos - self.trackBodyLastPos)/self.perSimStep_dt
        #print('setSimVals : Tracked Body Position : {} |\tLast position : {} |\tVel : {}'.format(self.trackBodyCurPos, self.trackBodyLastPos,self.trackBodyVel ))
        #save current state to restore for forward sim-dependent objectives/constraints, and to use for cost/objective functions
        self.curQ = self.skel.q
        self.curQdot = self.skel.dq
        #current end effector position in world space -
        self.curEffPos = self.reachBody.to_world(x=self.reachBodyOffset)
        #whether or not ot calculate dynamic jacobians and other quantities
        #torque cntrol desired to provide pulling force at contact location on reaching hand    
        self.Tau_JtFpull, self.JtPullPInv, self.Jpull, self.JpullLin = self.getPullTau(self.useLinJacob)    
        if(calcDynQuants):                   
            #current end effector velocity in world space
            self.currEffVel = self.Jpull.dot(self.curQdot)
            #current end effector acceleration in world space : Jdot * ddq + Jddot * dq
            JdotPull = self.getEffJdot(self.useLinJacob)
            self.currEffAccel = JdotPull.dot(self.curQdot) + self.Jpull.dot(self.skel.accelerations())#self.Jpull.dot(self.curQdot) + self.Jpull.dot(self.curQdot)   
    
    # end per-step body values TODO
    ####################################################
        

    #####################################################
    #       end optimization initialization    
    #
    #   per-step optimization/torque/force calc functions
        
    #build operational space controller as per DART tutorial
    #perhaps this would work?
    def compOSCntrllr(self):
        
        #end effector jacobian        
        J = self.Jpull
        qDot = self.curQdot
        #linear and angular (if present) components of error
        osc_error = np.zeros(self.useForce.shape)
        #error gradient
        osc_dError = np.zeros(self.useForce.shape)
        linVel = np.zeros(self.useForce.shape)
        #vector from eff to constraint
        vecToCnstrnt = self.getWorldPosCnstrntToFinger()
        #distance error
        osc_error[-3:]=vecToCnstrnt
        if (self.useLinJacob) :
            #find end effector velocity -> xdot = J * qdot --> want relative velocity between ball@constLoc and eff
            #Since ball is being moved manually every timestep, need to use constraint vel in world space
            JcnstCntct = self.cnstrntBody.linear_jacobian(offset=self.cnstrntOnBallLoc)
        else :
            #angular components (osc_error[:3]) are 0 here : angular components are differences between orientation of reachbody and desired orientation
            #in c++ this part would be, where the linear part of the transform is the rotation component of the matrix (linear as in system of linear eqs):            
            #Eigen::AngleAxisd aa(mTarget->getTransform(mEndEffector).linear()); 
            #       mTarget is target frame(loc, orientation); 
            #       getransform here is returning endeffector's world transform inverse * target's transform -> the rotation of the target w/respect to eff
            #e.head<3>() = aa.angle() * aa.axis() -> this is angle error in each axis (total angle error * unit rot axis)
            #TODO can we approximate this with inv world transform of body node  x unit vector(representing target orientation) with 0 in 4th part?
            #unit vector would be osc_error[3:]/mag(osc_error[3:])
            mag = np.linalg.norm(vecToCnstrnt)
            if(mag > 0.0):
                #this is desired orientation - point at target
                orientVec_v =(1.0/mag)*vecToCnstrnt
                orientVecRot = np.append(orientVec_v, [0.0])
                TbodyInv = np.linalg.inv(self.reachBody.transform())
                invRotRes=TbodyInv.dot(orientVecRot)
                osc_error[:3]=invRotRes[:3]            
            #find end effector velocity -> xdot = J * qdot --> want relative velocity between ball@constLoc and 
            JcnstCntct =  self.cnstrntBody.world_jacobian(offset=self.cnstrntOnBallLoc)
        
        #error vel
        osc_dError = JcnstCntct.dot(self.cnstrntBody.skel.dq) - J.dot(qDot)
        #J dot @ eef
        derivJ = self.getEffJdot(self.useLinJacob)
        #pseudo inverse of jacobian and J dot
        pinv_J = np.linalg.pinv(J)
        pinv_dJ = np.linalg.pinv(derivJ)
        #def in ctor
#        self.osc_Kp = np.diag(np.ones(6)*Kp)
#        self.osc_Kd = np.diag(np.ones(self.ndofs)*Kd)
        resKpE = self.osc_Kp.dot(osc_error)
        res1 = pinv_J.dot(self.osc_Kp.dot(osc_dError))
        res2 = pinv_dJ.dot(resKpE)
        Kd_dq = self.osc_Kd.dot(qDot)
        Kd_JinvKp_e = self.osc_Kd.dot(pinv_J.dot(resKpE))
        
        desTorques = self.M.dot(res1 + res2) - Kd_dq + Kd_JinvKp_e + self.CfG + self.Tau_JtFpull        
        return desTorques      

    
    #keep around last step's decision variable
    def setPrevCurrGuess(self, x):
        self.prevGuess = self.currGuess 
        self.currGuess = x  
    
    #initialize these values every time step - as per Abe' paper they are constant per timestep
    def setSimVals(self):        
        self.oneAOvTS = self.dofOnes/self.perSimStep_dt
        #per timestep constraint and eef locations and various essential jacobians
        self.setPerStepStateVals(True)
        #Mass Matrix == self.skel.M
        #M is dof x dof 
        self.M = self.skel.M
        #M * qd / dt - precalc for MAconst
        #M / dt = precalc for MAconst - multiply by qdotPrime
        self.M_ovDt =self.M/self.perSimStep_dt
        #derivative of MA eq w/respect to qdotprime
        self.M_dqdotPrime = self.M.dot(self.oneAOvTS)
        #precalc -Mqdot/dt
        self.Mqd = -self.M_ovDt.dot(self.skel.dq)
        #CfG == C(q,dq) + G; it is dof x 1 vector
        self.CfG = self.skel.coriolis_and_gravity_forces() 
        #TODO constraint forces - investigate
        #self.CntrntFrc = self.skel.constraint_forces()    
        # number of iterations of optimization
        self.optIters = 0        
        #specific settings for instancing class - specific for skeleton configuration
        self.setSimValsPriv()          
        #if pose matching, using q/qdot for pose elements being matched
        if self.doMatchPose :
            self.curMatchPose = self.curQ[(self.optPoseUseIDXs)]
            self.curMatchPoseDot = self.curQdot[(self.optPoseUseIDXs)] 


    #solving quadratic objectives : 
    # min (a, F(cntct), tau)
    
    #   w1 * | pose_fwd_accel - pose_des_accel | + 
    #   w2 * | COM_fwd_accel - COM_des_accel |
    
    #subject to 
    # : M a + (C(q,dq) + G) + Tau_JtF(cntct) + Tau_JtF(grab) = tau 
    # : ground forces being in the coulomb cone
    # : tau being within torque limits
    # : subject to no slip contacts on ground


    #prestep for kinematic sim of bot - ik to appropriate location
    def preStepKin(self):
        if (self.debug):
            print('helperBotSkelHolder::preStepKin : {} robot set to not mobile, so no optimization being executed, but IK to constraint position performed'.format(self.skel.name))
        self.setPerStepStateVals(False)
        self.IKtoCnstrntLoc()
       
        
    #pre-sim step for dynamic simulation - initialize and solve optimal control
    def preStepDyn(self):
        #save world state - this is so that the state can be restored after fwd sim in opt
        #self.env.saveSimState()
        #set sim values used by optimization routine
        self.setSimVals()
                
        #determine control
        if(self.useOSControl):
            tauOSC = self.compOSCntrllr()
            self.dbgShowTorques(tauOSC)
            self.tau = tauOSC
            
        else :            
            #build optimizer with LD_SLSQP alg - remake every time to handle static baloney seems to be happening
            self.initOptimizer(nlopt.LD_SLSQP)
            #mma opt formulat only handles inequality constraints - use pos and neg ineq for eq constraint
            #self.initOptimizer(nlopt.LD_MMA)
            #initialize optimizer - objective function is minimizer
            #TODO set which objectives we wish to use
            self.optimizer.set_min_objective(self.objFunc)                
            #set constraints F=MA and fCntctCnst
            #need to set tol as vector        
            #self.optimizer.add_equality_constraint(self.MAcnst, self.cnstTol[0])
            self.optimizer.add_equality_mconstraint(self.MAcnstVec, self.cnstTolMA)
            #mma only handles inequality constraints so use dual-bound constraints
            #self.initOptimizer(nlopt.LD_MMA)
            #self.optimizer.add_inequality_constraint(self.MAconstPos, self.cnstTol[0])
            #self.optimizer.add_inequality_constraint(self.MAconstNeg, self.cnstTol[0])
            #expects inequality constraint f(x) <= 0   
            #stop if less than tol changes happen to objective function eval - needs to be small
            self.optimizer.set_ftol_rel(1e-13)
            #stop after numOptIters
            self.optimizer.set_maxeval((int)(self.numOptIters))
            #instance specific inequality constraints - ground contact forces, for example
            self.setInstanceConstrnts()           
            
            #print('stopping if obj func <= {}'.format(self.optimizer.get_stopval()))
            #run optimizer - use last result as guess
            #guess = self.nextGuess.tolist()
            try:
                self.nextGuess = self.optimizer.optimize(self.nextGuess.tolist())
            except Exception as e:
                print('Exception {} thrown, using previous iteration value.'.format(e))
                self.nextGuess = np.copy(self.prevGuess)

            #if monitoring guesses                
            if(self.monitorOptGuess):
                self._checkiMinMaxVals(self.nextGuess, self.minMaxGuessDict)
                
            self.doStepTestsAndSetTau_OptCntrl()        
    
    #dt set to timestep * frameskips functionality before sim step is executed on this skel
    #here is where we calculate the robot's control torques
    def _preStepIndiv(self, actions):   

        #if not mobile (dynamic) skeleton, perform IK on position
        if (not self.isFrwrdSim):  #if not dynamic and this is called then we want to IK to position
            self.preStepKin()
        else :                      #dynamic, and prestep called, means we want to solve opt control
            self.preStepDyn()       

    #set up mimic bot for IK calculation
    def setMimicBot(self, _mimicSkel):
        self.mimicBot = _mimicSkel
        self.mimicReachBody = self.mimicBot.body(self.reach_hand)
        #use self.reachBodyOffset to find actual location

    #use force ana sees at eef using proposed optimal action to derive assist force for bot 
    def frwrdSimBot_DispToFrc(self, frcDbgDict, dbgFrwrdStep=False):
        # ulj = self.useLinJacob
        # ana = self.helpedSkelH
        # anaSkel = ana.skel
        # #get current cnstrntFrc on ANA
        # #extFrcAtEef, allCnstrntFrcAtEef, ttlCntctBdyFrcAtEef
        # anaEefPullFrc, anaEefCnstrntFrc, anaCntctBdyFrcAtEef = ana.getExternalForceAtEef(ulj)


        #use ANA's eef force here to derive control for bot using optimization - use secondary bot to derive control to generate eef force ANA sees
        print("\n\n!!!!!!!!!!!!!!!!!!!!helperBotSkelHolder::frwrdSimBot_DispToFrc : Not Implemented!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")


    #first use IK to find new pose for bot to enable desired displacement, then use SPD to move to this pose
    def frwrdSimBot_DispIKSPD(self, desDisp, dbgFrwrdStep=False):
        skel = self.skel
        #first find pose by IK'ing mimic skeleton - des displacement is relative to main helper bot, find instead location of mimic's eef
        #solve for IK to des displacement (desDisp) from current constraint location
        #IK part 
        #set up initial quantities necessary for IK, along with dynamic quantities used for later SPD
        self.setPerStepStateVals(False)
        #initialize mimic skeleton to perform IK - first mimic current pose and vel of helper bot        
        self.mimicBot.set_positions(skel.q)
        self.mimicBot.set_velocities(skel.dq)
        mimicBotEefPos = self.mimicReachBody.to_world(x=self.reachBodyOffset)
        #need to find appropriate IK target position by transferring displacement in bot frame to mimic bot frame 
        mimicBotNewPos = mimicBotEefPos + desDisp
        #solve IK  - IK end effector to passed position (world coords) 
        self.IKtoPassedPos(mimicBotNewPos, skel=self.mimicBot, reachBody=self.mimicReachBody)
        mimicBotIKEefPos = self.mimicReachBody.to_world(x=self.reachBodyOffset)
        print("mimicBotEefPos : {} to go to {} : new mimic Eef Pos after IK : {} ".format(mimicBotEefPos,mimicBotNewPos,mimicBotIKEefPos))
        #solve and frwrd integrate helper bot to move the specified delta amount, using IK to find desired q/qdot and then SPD to solver for torques
        #SPD part
        # SPD
        ulj =self.useLinJacob
        ana = self.helpedSkelH
        botEefCnstrntFrc = self.getEefCnstrntFrc(ulj)
        #extFrcAtEef, allCnstrntFrcAtEef, ttlCntctBdyFrcAtEef
        anaEefPullFrc, anaEefCnstrntFrc, anaCntctBdyFrcAtEef = ana.getExternalForceAtEef(ulj)
        print("helperBotSkelHolder::frwrdSimBot_DispIKSPD :\n\tbot c frc @ eef : \t\t{}\tana c frc @ eef : \t\t{}\n\tana cntct frc @ eef : \t\t{}\t!!!ana frc Pull @ eef : \t{}".format(botEefCnstrntFrc, anaEefCnstrntFrc, anaCntctBdyFrcAtEef, anaEefPullFrc))
#        #jacobian to end effector  
#        JTransInv, JTrans, Jpull, _ = self.getEefJacobians(useLinJacob, body=self.reachBody, offset=self.reachBodyOffset)
#        bot_JtPullPInv, botJTrans = self.getJInvTrans()
#        ana_JtPullPInv, _ = self.helpedSkelH.getJInvTrans()
#    
#        trajCnstrntFrc = self.cnstrntBody.skeleton.constraint_forces()#
#        trajJtPullPInv = np.transpose(self.cnstrntBody.skeleton.bodynodes[0].world_jacobian(offset=self.reachBodyOffset))
#        worldTrajCnstrntFrc = trajJtPullPInv.dot(trajCnstrntFrc)
#   
#        botCnstrntFrc = skel.constraint_forces()
#        worldBotCnstrntsFrc = bot_JtPullPInv.dot(botCnstrntFrc)
#        worldAnaCnstrntsFrc = ana_JtPullPInv.dot(self.helpedSkelH.skel.constraint_forces())
#        anaToBotCFrc = botJTrans.dot(-worldAnaCnstrntsFrc)
#        
#        
#        print("cbody c frc : \t\t{} \nwrld cbody c frc : \t{} \nbot c frc : \t\t{}\nana c frc in world : \t{}\nbot c frc in world : \t{}\nana c frc in bot : \t{} ".format(trajCnstrntFrc, worldTrajCnstrntFrc, botCnstrntFrc,worldAnaCnstrntsFrc, worldBotCnstrntsFrc, anaToBotCFrc))

        q = skel.q
        dq = skel.dq
        #sim quantities
        M = skel.M
        CorGrav = skel.coriolis_and_gravity_forces()
        kd_ts = self.Kd_ts_SPD
        mKp = self.Kp_SPD
        mKd = self.Kd_SPD
        dt = self.perSimStep_dt
        
        #get current pose - this is the pose we want to SPD to - NOTE we do not want root dof locations from mimic bot, if any exist
        #qBar = np.zeros(self.ndofs)
        qBar = np.copy(q)#retain root to world dofs
        qBar[self.stTauIdx:] = self.mimicBot.q[self.stTauIdx:]
        #we need to derive desired dq by taking starting dq and average dq (Which will result in desired eef displacement == (qBar - orig q)/dt )
        avgDq = (qBar - q)/dt
        #avgDq == (dq + dq1)/2 -? dq1 = (2*avgDq) - dq
        dqBar = (2*avgDq) - dq
        #calc accel resulting in current vel
        #ddqSim_dt = (dq - self.oldDq)#/dt
        #print("skel accel : {} | calc : {}".format(skel.ddq,ddqSim_dt))        

        cnstrntFrc = skel.constraint_forces() 

        #spd formulation
        invM = np.linalg.inv(M + kd_ts)
        nextPos = q + dq * dt
        p = -mKp.dot(nextPos - qBar)
        #nextVel = dq + ddq * dt
        d = self.zeroKD # == -mKd.dot(dq - dq)     #setting this makes matching IK stable.
        #d = -mKd.dot(dq - dqBar)               
        #d = -mKd.dot(dq)# - dqBar)            
        print("helperBotSkelHolder::frwrdSimBot_DispIKSPD : d : {}".format(d))            
        #d = -mKd.dot(avgDq)# - dqBar)                        
        ddq = invM.dot(-CorGrav + p + d + cnstrntFrc)
        #cntrl = p + d - mKd.dot(ddq * dt)
        cntrl = p + d - kd_ts.dot(ddq)
        #set tau
        self.tau = cntrl
        #clear control for unactuated root dofs
        self.tau[:self.stTauIdx] = 0
        #??? TODO only step me forward ???
        #self._stepNoneButMe
        #self.oldDq = np.copy(dq)
        #self.env.pauseForInput("helperBotSkelHolder::frwrdSimBot_DispIKSPD")

    #solve and frwrd integrate helper bot, to find actual force generated solving for optimization process
    #traj is expected to be evolved before this is called
    #BOT'S STATE IS NOT RESTORED if  restoreBotSt=False
    def frwrdStepBotForFrcVal(self, desFrc, recip, restoreBotSt=True, dbgFrwrdStep=False):
        #turn off bot debugging, will still display overall force generated
        self.debug=False        
        #find dir of force
        norm=np.linalg.norm(desFrc)
        if (norm == 0) :
            desFrcDir = np.array([0,0,0]) 
        else : 
            desFrcDir = (desFrc / norm)
            
        #init and solve for bot optimal control for current target force
        self.setDesiredExtAssist(desFrc, desFrcDir, setReciprocal=recip) 
        #initialize solve - always dynamic for this
        self.preStepDyn()            
        #step bot forward to get f_hat 
        #save state, if we wish to restore state
        if(restoreBotSt):
            saved_state = self.skel.states()
        #torque isn't relevant if we are frwrd integrating here 
        self.applyTau()
        #frwd sim robot
        self._stepNoneButMe()
       
        #build dictionary of skeleton dynamic data related to frwrd step
        self.frcD = self._buildPostStepFrcDict()
        #display results of frwrd step
        self.dispFrcEefRes(dbgFrwrdStep)
        #get generated force
        f_hat = self.frcD['totPullFrc'][3:]       
        #restore bot state to previous state
        if(restoreBotSt):
            self.skel.set_states(saved_state)
            
        return f_hat, self.frcD    
    
    ######################################################
    #   cost/reward methods 
    ###################################################### 
    
    #generic objective/cost function calculation
    #returns .5 * diff^T * wtMat * diff and gradient (wtMat * diff)
    #assumes d(newVal) == np.ones(newVal.shape) (all 1s for individual newVal gradient, so newval is not product of actual indep variable and some other object, like jacobian)
    def genericObj(self, costwt, newVal, oldVal, wts):
        diff = newVal-oldVal
        wtDotDiff =  wts.dot(diff)
        y = costwt * (.5 * np.transpose(diff).dot(wtDotDiff))
        dy = costwt * wtDotDiff
        return y, dy
        

    #calculate pose matching objective - performed in generalized coordinate space
    #minimize difference between some preset pose config and the current velocity fwd integrated
    #requires optPoseUseIDXs to be defined (idxs of dofs to match)
    def pose_objFunc(self, poseWt, qdotPrime):
        #pose matching via accelerations
        poseAccel = np.zeros(self.ndofs)
        desPoseAccel = np.zeros(self.ndofs)
        poseAccel[(self.optPoseUseIDXs)] = (qdotPrime[(self.optPoseUseIDXs)] - self.curMatchPoseDot)/self.perSimStep_dt
        #desire pose acceleration TODO : this is in pose space, not world space, and only of dofs we are trying to match
        desPoseAccel[(self.optPoseUseIDXs)] = (self.matchPose - self.curMatchPose) * self.kPose - self.tSqrtKpose * self.curMatchPoseDot
        poseAccelDiff = poseAccel-desPoseAccel
#        #Only Pose here
        posePart = poseWt * .5 * (poseAccelDiff.dot(poseAccelDiff))
        poseGrad = poseWt * poseAccelDiff
        return posePart, poseGrad
    
#    #minimize exerted force objective - requires JtPullPInv to be calculated, to find force
#    def force_objFunc(self, frcWt, tau):
#        #difference between pull force as determined by torques at point of application and actual desired force
#        #JtPullPInv dot tau is actual force at application point
#        #derivative is JtPullPInv 
#        #
#        jtDotTau = self.JtPullPInv.dot(tau)
#        frcDiff = jtDotTau - self.useForce        
##        #isn't this below torques of whole body, vs Torques responsible for external force
##        #really is tau difference
#        #frcDiff = tau - self.Tau_JtFpull
#        #print('difference between actual and desired force {}'.format(frcDiff))
#        
#        frcPart = frcWt * (.5 * (frcDiff.dot(frcDiff)))
#        #torque gradient here is derivative w/respect to tau
#        #deriv of f(g(x)) is f'(g(x))*g'(x)
#        #f(x)== frcWt * .5 * x^2 :: f'(x)= frcWt * x
#        #g(x) == jtDotTau - self.useForce -> JtPullPInv.dot(tau) - self.useForce ::  g'(x) == JtPullPInv
#        frcGrad = frcWt * frcDiff.dot(self.JtPullPInv)
#        return frcPart, frcGrad
    
    #minimize end effector distance from sphere location - for tracking
    #using future velocity to fwd integrate current position
    def effPos_objFunc(self, locWt, qdotPrime):
        curLoc = self.curEffPos
        curWrldVel = self.JpullLin.dot(qdotPrime)
        #new location in world will be current world location + timestep* (worldVel == J_eff * new qdot)
        newLoc = curLoc + self.perSimStep_dt * curWrldVel
        #trackBody needs to be set for this objective
        locDiff = newLoc - self.trackBodyCurPos
        locPart = locWt * (.5 * (locDiff.dot(locDiff)))
        #gradient of locPrt = locDiff * d(locDiff) 
        #d(locDiff) = timestep * self.JpullLin
        locGrad = locWt * locDiff.dot(self.perSimStep_dt*self.JpullLin)
        return locPart, locGrad
    
    #match a world-space velocity with end effector velocity
    def effVel_objFunc(self, pVelWt, qdotPrime):
        curWrldVel = self.JpullLin.dot(qdotPrime)
        pVelDiff = curWrldVel - self.trackBodyVel
        pVelPart = pVelWt *(.5*(pVelDiff.dot(pVelDiff)))
        #gradient of vel part == pVelDiff * d(pVelDiff)
        #d(pVelDiff) == self.JpullLin
        pVelGrad = pVelWt * pVelDiff.dot(self.JpullLin)
        return pVelPart,pVelGrad    
    
    @abstractmethod
    def doStepTestsAndSetTau_OptCntrl(self):
        pass
            
    @abstractmethod         
    def setSimValsPriv(self):
        pass
    
    #objective function referenced by optimization process
    @abstractmethod
    def objFunc(self, x, grad):
        pass
       
    #MA constraint equation
    @abstractmethod
    def MAcnstVec(self, result, x, grad):
        pass
    
    def MAconstVecPos(self, result, x, grad):
        return self.MAcnstVec(result, x, grad)
    def MAconstVecNeg(self, result, x, grad):
        return -self.MAcnstVec(result, x, grad) 
    
    
    #set all instance-specific constraints
    @abstractmethod
    def setInstanceConstrnts(self):
        pass
    
    #test a proposed optimization solution to see if it satisfies constraint
    def dbg_testMAconstVec(self, x, debug=False):
        result = np.zeros(np.size(self.cnstTolMA))
        tar = np.zeros(np.size(self.cnstTolMA))
        self.MAcnstVec(result, x, np.empty(shape=(0,0)))
        passed = True
        #result should be all zeros
        if (not np.allclose(result,  tar, self.cnstTolMA)):
            print('!!!!!!!!!!!!!!!!! MAcnstVec constraint violated : ')
            passed = False
            if(not debug):
                for x in result:
                    print('\t{}'.format(x))  
        #else:
        #    print('MAcnstVec satisfied by result')
        if(debug):
            for x in result:
                print('\t{}'.format(x))  
        return passed
        
        
    def dbg_dispMinMaxGuesses(self):
        self.dbgDispGuessPriv(self.minMaxGuessDict['min'], 'Min Guess Vals Seen ')
        print('\n')
        self.dbgDispGuessPriv(self.minMaxGuessDict['max'], 'Max Guess Vals Seen ')
        print('\n')
    
    #display instance classes partition of guess values for debugging
    @abstractmethod
    def dbgDispGuessPriv(self, guess, name=' '):   pass
    
    #perform post-step calculations for robot - no reward for Inv Dyn
    def calcRewardAndCheckDone(self, obs, debug, dbgStructs): 
        
        if (not self.isFrwrdSim) and (self.debug):
            print('helperBotSkelHolder::calcRewardAndCheckDone : No Optimization since {} set to not mobile'.format(self.skel.name))

        done=False
        reward=0
        #dct : dictionary of lists of reward components, holding reward type as key and reward component and reward vals watched as value in list (idx 0 is reward value, idx 1 is list of rwrd values watched)  
        return reward, done, dbgStructs

#class to hold assisting robot
class robotSkelHolder(helperBotSkelHolder):
                 
    def __init__(self, env, skel, widx, stIdx, fTipOffset):          
        helperBotSkelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset)
        self.name = 'Humanoid Helper Bot'
        self.shortName = 'Helper Bot'
 
        self.numOptIters = 1000
        #const bound magnitude
        self.bndCnst = 200
        #robot biped uses contacts in optimization process
        self.numCntctDims = 12
        #robot optimization attempts to match pose
        self.doMatchPose = True
        #self.nOptDims = self.getNumOptDims()

        
    def getNumOptDims(self):        
        # dim of optimization parameters/decision vars-> 2 * ndofs + 12 for full body bot : qdot, cntctFrcs, Tau
        return 2*self.ndofs + 12

    #return idx's of dofs that are to be matched if pose matching objective is used
    def initOptPoseData_Indiv(self):
        #dof idxs to be used for pose matching optimization calculations
        #ignore waist, and reach hand dofs and root location
        #root orientation 0,1,2; root location 3,4,5
        #left thigh 6,7,8 : 6 is bend toward chest, 7 is twist along thigh axis, 8 is spread
        #left shin, left heel(2), left toe 9,10,11,12
        #right thigh ,13,14,15: 13 is bend toward chest, 14 is twist along thigh axis, 15 is spread
        #right shin, right heel(2), right toe 16,17,18,19
        #abdoment(2), spine 20,21,22,; head 23,24
        #scap left, bicep left(3) 25,26,27,28 ; forearm left, hand left 29,30
        #scap right, bicep right (3) 31,32,33,34 ; forearm right.,hand right 35,36
        if('h_hand_right' in self.reach_hand) :
            return np.array([0,1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,23,24,25,26,27,28,29,30])
        else : #left hand reaching, match pose of right hand
            return np.array([0,1,2,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,23,24,31,32,33,34,35,36])
        
    #build modification array for building qdot wts->vector of tuples of idx and multiplier
    def buildQdotWtModAra(self):
        #abdoment(2), spine 20,21,22,; head 23,24
        #scap left, bicep left(3) 25,26,27,28 ; forearm left, hand left 29,30
        #scap right, bicep right (3) 31,32,33,34 ; forearm right,hand right 35,36

        modAra = list()
        #waist/spine
        modAra.append((20,2.0))
        modAra.append((21,2.0))
        modAra.append((22,2.0))
        if('h_hand_right' in self.reach_hand) :
            #array of importance weights for right hand need to be zero
            armIdxs = np.arange(31,37)
        else :
            #left hand reaching, importance of left hand dofs needs to be zero
            armIdxs = np.arange(25,31)        
        for i in armIdxs:
            modAra.append((i,0.1))        
        return modAra

    #initialize nlopt optimizer variables specific to this robot - contact vars
    def initOptIndiv(self):
        #index array of locations in fcntct array of y component of contact force - should always be positive
        fcntct_y_idxAra = np.array([1,4,7,10])
        #modify lower bound to handle contact bounds - no negative y contact force
        tolVal= 1e-12
        offsetYIdxs = (fcntct_y_idxAra + self.ndofs)
        if(self.useBoxConstraints):
            self.lbndVec[offsetYIdxs] = tolVal     
        
        #up vector for all 4 contact forces
        self.fcntctUp = np.zeros(self.numCntctDims)
        self.fcntctUp[fcntct_y_idxAra] = 1
#        #derive of contact force eq - setting to negative since cone calc uses difference in y dir
        self.fcntctDot = np.ones(self.numCntctDims)
        self.fcntctDot[fcntct_y_idxAra] *= -1
        #tolerance of contact constraints (4 constraints) - 1 scalar value per contact
        # 1/3 # of contact dims
        szCntctTol = int(self.numCntctDims/3)
        self.cnstTolCNTCT = np.ones(szCntctTol)*1e-8
                                
    #assistant robot skel's individual per-reset settings
    def _resetIndiv(self, dispDebug):
#        print('bot head height : {} '.format(self.skel.body(self.headBodyName).com()[1] ))
#        print('bot foot height : {} '.format(self.skel.body('h_heel_left').com()[1] ))
#        print('bot com  : {} '.format(self.skel.com() ))
        pass

    #return jacobian transpose for all 4 contact bodies
    #this will be a single matrix of n x 3m dim where n is # of dofs
    #and m is # of contact forces (4)
    def getCntctJTranspose(self):
        #get 
        #3 values for each force (4 forces)  x numdofs
        cntctJ = np.zeros([12, self.ndofs])
        idx = 0
        COPPerBody = np.zeros([12])
        for bodyName in self.feetBodyNames:
            body = self.skel.bodynode(bodyName)
            #this is the jacobian at an estimate of the COP of a foot-related
            #body in contact with the ground - projecting the body's COM into world 
            #coords, then setting y==0, and finding jacobian at that location
            COMinWorld = body.com()
            COMinWorld[1] = 0
            #J is 3 rows x ndofs cols - get com of each body in world, moved to ground (approx position of contact) and move to local space
            J = body.linear_jacobian(offset=body.to_local(COMinWorld))
            #TODO use world jacobian if wanting contact torque and force (6 x ndof)
            #fill by rows
            nextIdx = idx+np.size(J, 0)
            cntctJ[idx:nextIdx, : ] = J
            COPPerBody[idx:nextIdx] = COMinWorld
            idx = nextIdx
        return np.transpose(cntctJ), COPPerBody             
    
    #initialize these values every time step - as per Abe' paper they are constant per timestep
    def setSimValsPriv(self):
        #get contact profile for this skeleton from previous step
        self.oldCntctData = self.getMyContactInfo()        
        #Jacobian for all contact forces (appended)
        #world location of contact force - vector of 12 x 1 dims
        self.JtCntctTrans, self.wrldCOPPerFootBody = self.getCntctJTranspose()
        #set contact gradient
        self.JtCntctGrad = np.full_like(self.fcntctIDXs,0)
        for i in range(len(self.fcntctIDXs)):
            self.JtCntctGrad[i]=self.JtCntctTrans[:,i].sum(axis=0)        
                                     
        
    #call this to return objective function value that minimizes velocity and 
    #returns scalar value
    #grad is dim n
    def objFunc(self, x, grad):
        self.setPrevCurrGuess(x)
        self.optIters +=1
        qdotPrime = x[self.qdotIDXs]
        tau = x[self.tauIDXs]
        #weight of each component of objective function
        velWt = 2.0
        frcWt = 1.0
        poseWt = 1.0
        #pose matching        
        posePart, poseGrad = self.pose_objFunc(poseWt, qdotPrime)
        #weighted qdot min
        velPart, velGrad = self.genericObj(velWt, qdotPrime, self.curQdot, self.qdotWts)       
        #frcPart, frcGrad = self.force_objFunc(frcWt, tau)           
        funcRes = velPart + posePart ##+ frcPart 
        if grad.size > 0:
            grad[:]=np.zeros(grad.shape())
            #if gradient exists, set it here.  Must be set to 0 for all terms not modified
            #grad calc - put grad in place here
            #func is .5 * xT * W * x
            #func' is .5 * (xT * W * WT); W*WT * .5 == W
            grad[self.qdotIDXs] = velGrad  + poseGrad#self.qdotWts.dot(velDiff)
            #torque gradient here is derivative w/respect to tau
            #deriv of f(g(x)) is f'(g(x))*g'(x)
            #f(x)== frcWt * .5 * x^2 :: f'(x)= frcWt * x
            #g(x) == jtDotTau - self.useForce -> JtPullPInv.dot(tau) - self.useForce ::  g'(x) == JtPullPInv
            #grad[self.tauIDXs] = frcWt *frcDiff.dot(self.JtPullPInv)
            #$grad[self.tauIDXs] = frcGrad
        #funcRes = sqrt(np.transpose(velDiff).dot(self.qdotWts.dot(velDiff)))
        if(self.optIters % 100 == 1):
            print('objectiveFunc iter : {} res : {}'.format(self.optIters,funcRes))
        
        return funcRes
    

    #equation of motion constraint
    def MAcnstVec(self, result, x, grad):
        #print('MAcnst iter')
        qdotPrime = x[self.qdotIDXs]
        fcntct = x[self.fcntctIDXs]
        tau = x[self.tauIDXs]
        
        #To use new cfg to provide coriolis, apply new qdot and get new cfg, then reapply old qdot to return to old cfg
        #this doesn't make much difference
#        self.skel.set_velocities(qdotPrime)
#        CfG = self.skel.coriolis_and_gravity_forces()
#        self.skel.set_velocities(self.curQdot)
        #TODO Add constraint force?
        maRes = self.M_ovDt.dot(qdotPrime) + self.Mqd + self.CfG #+ self.CntrntFrc
        
        #cntctRes is positive, because force is up (reactive collision force)
        cntctRes = self.JtCntctTrans.dot(fcntct)
        #build pull force and subtract tau estimate
        #pulling force relative to external force is negative - want opposite direction
        #in other words, we want the resultant torques to generate this force 
        tauPull = self.Tau_JtFpull - tau
        
        result[:] = maRes + cntctRes + tauPull     
        
        if grad.size > 0:
            #if gradient exists, set it here.  Must be set to 0 for all terms not modified
            grad[:,:] = np.zeros(grad.shape)
            #if gradient exists, set it here.  Must be set to 0 for all terms not modified
            #grad is matrix of len(result) rows (37) x len(x) cols (86)
            #gradient for all qdot terms
            grad[:,self.qdotIDXs] = self.M_ovDt
            #gradient for all contact force terms
            grad[:,self.fcntctIDXs] = self.JtCntctTrans
            #gradient for all taus - diagonal mat with -1 on diags            
            grad[:,self.tauIDXs] = self.tauDotMat
        #return ttlRes#np.sum(np.absolute(ttlVec))        

            
    def testFcntctCnstVec(self, x):
        result = np.zeros(np.size(self.cnstTolCNTCT, axis=0))
        self.fCntctCnstVec(result, x, np.empty(shape=(0,0)))
        #result should be all zeros
        res = np.sum(np.abs(result))
        passed = (res > 0)
        if(not passed):
            print('testFcntctCnstVec Not 0 sum result')
            for x in result:
                print('\t{}'.format(x))
        return passed
       
    #vector value contact contraint, returns a value for each contact point
    def fCntctCnstVec(self, result, x, grad):
        fcntct = x[self.fcntctIDXs]
        #fcntct == 12elements, == 4 3 element vectors
        sqrtVals = [0,0,0,0]
        useLastVals = False
        for x in range(0,4):
            stIdx = 3*x
            #mag of tangent - mag of normal needs to be <= 0
            sqrtVals[x] = sqrt((fcntct[stIdx]*fcntct[stIdx]) + (fcntct[(stIdx+2)]*fcntct[(stIdx+2)]))
            if(sqrtVals[x] == 0):
                #sqrtVals[x]=1e-7
                useLastVals = True
            result[x] = sqrtVals[x] - fcntct[stIdx+1]   
        #print('fCntctCnst result size : {}'.format(np.size(result, axis=0)))
        if grad.size > 0:            
            #if gradient exists, set it here.  Must be set to 0 for all terms not modified
            grad[:,:] = np.zeros(grad.shape)
            #grad is matrix of 4 rows x size(x) cols 
            #self.fcntctDot is 1 for x and z of each contact point, and -1 for each y
            #grad[self.fcntctIDXs] = self.fcntctDot 
            # if sqrtVals[x] is 0 then use finite difference from last iteration over timestep for gradient 
            if useLastVals:
                oldFcntct = self.prevGuess[self.fcntctIDXs]
            
            for x in range(0,4):
                stIdx = 3*x
                gradColSt = self.fcntctIDXs[stIdx]
                if(sqrtVals[x] != 0):
                    grad[x,gradColSt] = fcntct[stIdx]/sqrtVals[x]
                    grad[x,(gradColSt+2)] = fcntct[stIdx+2]/sqrtVals[x]
                else :
                    #TODO:finite difference - appropriate? 
                    grad[x,gradColSt] = (fcntct[stIdx] - oldFcntct[stIdx])/self.perSimStep_dt
                    grad[x,(gradColSt+2)] = (fcntct[stIdx+2] - oldFcntct[stIdx+2])/self.perSimStep_dt
                grad[x,(gradColSt+1)] = -1
                
    #perform any tests if desired and populate tau                 
    def doStepTestsAndSetTau_OptCntrl(self):
        objFuncEval = self.objFunc(self.nextGuess, np.empty(shape=(0,0)))
        print('Opt Result Eval : {}\n'.format(objFuncEval))
        passMATest = self.dbg_testMAconstVec(self.nextGuess)
        passFcntctTest = self.testFcntctCnstVec(self.nextGuess)
        
        self.tau = self.nextGuess[self.tauIDXs]
        #leave nextGuess alone for these dofs, for gradient based calcs
        self.tau[:self.stTauIdx] *= 0        
    
    #constraints specific to this instance
    def setInstanceConstrnts(self):
        #expects inequality constraint f(x) <= 0
        #self.optimizer.add_inequality_constraint(self.fCntctCnst, self.cnstTolCNTCT[0])
        self.optimizer.add_inequality_mconstraint(self.fCntctCnstVec, self.cnstTolCNTCT) 

    #individual skeleton handling for calculating post-step dynamic state dictionary
    def _bldPstFrcDictPriv(self, frcD,indivCalcFlag):
        #seems to be always the same so far, although frwrd integrating 1st order using guess of dq yields inaccuracies in calculations
        frcD['dqDiff']=np.linalg.norm(self.nextGuess[self.qdotIDXs]-self.skel.dq)
        #total pull force of listed forces (doesn't include contact force calc if present)
        frcD['totPullFrc'] = frcD['jtDotTau'] - frcD['jtDotCGrav'] - frcD['jtDotMA']   
        cntctBdyTrqTtl = np.zeros(self.ndofs)
        contactRes = self.getMyContactInfo()
        cntctFrcTtl = np.zeros(self.useForce.shape)
        #find total body torque contribution from contacts
        for k,v in contactRes.items():
            cntctFrcTtl[3:] += v.ttlfrc
            cntctBdyTrqTtl += v.calcTtlBdyTrqs()
            
        #contact-related
        frcD['cntctBdTrq'] = cntctBdyTrqTtl
        frcD['jtDotCntct'] = frcD['JtPullPInv_new'].dot(cntctBdyTrqTtl)        
        frcD['cntctFrcTtl'] = cntctFrcTtl
        #target force to be seen at eef is self.use force
        frcD['targetEEFFrc'] = self.useForce

        return frcD
    
    #display results of fwd simulation - test results, display calculated vs simulated force results at end effector
    #sumTtl : total force used
    def _dispFrcEefResIndiv(self):
        #pull force should be == to jtpullpinv.dot(-MA - CG - JtcntctTrans.(cntctFrc) + Tau)
        #this is only internal force in opposite direction, need all contributing factors to get force in opposite direction
        #i.e. get all external forces, subtract gravity contribution of skeleton
        #calc body torques from all contacts -> Jtrans@cntctpt * fcntct
        print('\t\tTarget Force : \t\t{}\n'.format(self.useForce))

        totPullFrcNoCntct = self.frcD['totPullFrc']
        totPullFrc = totPullFrcNoCntct + self.frcD['jtDotCntct']
        totPullFrcCntct = totPullFrcNoCntct - self.frcD['cntctFrcTtl']
               
        print('\tContact Force : {}'.format(self.frcD['cntctFrcTtl']))
        frcDiffNow = self.frcD['totPullFrc'] - self.useForce
        #frcDiffOld =self.frcD['totPullFrcOld']- self.useForce    
        
        print('\nDifference (new) between proposed force and actual : {}'.format(frcDiffNow))
        #print('\nDifference (old) between proposed force and actual : {}'.format(frcDiffOld))
        print('\tdesFrc : {}'.format(self.useForce))
        print('\ttotPullFrcNoCntct:{}\n\ttotPullFrc  : {}\n\tcntctFrc : {}\n\ttotPullFrcCntct : {}'.format(totPullFrcNoCntct,totPullFrc, self.frcD['cntctFrcTtl'],totPullFrcCntct))

    #display this classes guess configuration
    def dbgDispGuessPriv(self, guess, name=' '):
        qdotPrime = guess[self.qdotIDXs]
        alignQdotStr = self.dotAligned(qdotPrime)
        fcntct = guess[self.fcntctIDXs]
        alignCntctStr = self.dotAligned(fcntct)
        tau = guess[self.tauIDXs]   
        alignTauStr = self.dotAligned(tau)
        #should have a dof corresponding to each component of qdot and tau
        dofs = self.skel.dofs
        print('\n{} : '.format(name))
        if(len(dofs) != len(qdotPrime)):
            print('!!!!!! attempting to print qdot values that do not align with skeleton dofs (different counts)')
            return
        if(len(dofs) != len(tau)):
            print('!!!!!! attempting to print torques that do not align with skeleton dofs (different counts)')
            return
        
        print('\tQdot for dofs : ')        
        #should have a dof corresponding to each component of tau
        dofs = self.skel.dofs
        alignTauStr = self.dotAligned(tau)
        for i in range(len(dofs)):
            print('\t\tDof : {:20s} | Qdot : {}'.format(dofs[i].name, alignQdotStr[i]))
        print('\n\tContacts : ')        
        for i in range(0,len(alignCntctStr),3):
            print('\t\tContact val : [{}, {}, {}]'.format(alignCntctStr[i],alignCntctStr[i+1],alignCntctStr[i+2]))
        print('\n\tTorques per dof : ')        
        for i in range(len(dofs)):
            print('\t\tDof : {:20s} | Torque : {}'.format(dofs[i].name, alignTauStr[i]))

#class to hold assisting robot
class robotArmSkelHolder(helperBotSkelHolder):
                 
    def __init__(self, env, skel, widx, stIdx, fTipOffset):          
        helperBotSkelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset)
        self.name = 'KR5 Helper Arm'
        self.shortName = 'KR5'

        #set to true to use box constraints
        self.numOptIters = 100       #bounds magnitude
        self.useBoxConstraints = False
        #bound for torques
        self.bndCnst = 500.0
        #limit for qdot proposals only if box constraints are set to true
        self.qdotLim = 5.0   
        self.doMatchPose = True
        #self.nOptDims = self.getNumOptDims()

    #build list of tuples of idx and multiplier to modify wts array/wts matrix for qdot weighting
    def buildQdotWtModAra(self):
        modAra = list()
        #put dofs to have wt multiplier modification here
        return modAra

    #initialize nlopt optimizer variables - nothing for this robot
    def initOptIndiv(self):
        pass

    #return idx's of dofs that are to be matched if pose matching objective is used
    def initOptPoseData_Indiv(self):
        #TODO find idxs to match pose - passing empty array
        return np.empty( shape=(0, 0), dtype=int)


    #constraints specific to this instance, if there are any - here they are assigned to optimization problem, as in comment
    #and they need to be defined in this class too
    def setInstanceConstrnts(self):
        #if this class had an inequality constraint it would be assigned to optimizer here, and defined elsewhere in class
#        self.optimizer.add_inequality_mconstraint(self.fCntctCnstVec, self.cnstTolCNTCT)
        pass

    def getNumOptDims(self):        
        # dim of optimization parameters/decision vars-> qdot and tau
        return 2*self.ndofs 

    #assistant robot skel's individual per-reset settings
    def _resetIndiv(self, dispDebug):

        pass
    
    #call this to return objective function value that minimizes velocity and 
    #returns scalar value
    #grad is dim n
    def objFunc(self, x, grad):
        self.setPrevCurrGuess(x)
        self.optIters +=1
        qdotPrime = x[self.qdotIDXs]
        tau = x[self.tauIDXs]
        #weight of each component of objective function
        velWt = 0.00001
        #matching location is assisted by matching target's velocity - 
        #helps smooth movement when ball has non-zero acceleration and locWt is not weighted very highly
        pVelWt= 0.00001        
        locWt = 0.99998        
        #match pose, if performed
        #posePart, poseGrad = self.pose_objFunc(poseWt, qdotPrime)
        #minimize velocity
        velPart, velGrad = self.genericObj(velWt, qdotPrime, self.curQdot ,self.qdotWts)        
        #minimize distance of end effector from constraint body
        locPart, locGrad = self.effPos_objFunc(locWt, qdotPrime)        
        #minimize difference in ball and end effector velocities
        pVelPart, pVelGrad = self.effVel_objFunc(pVelWt, qdotPrime)        

        funcRes = velPart + locPart + pVelPart #+ frcPart
        
        if grad.size > 0:
            #if gradient exists, set it here.  Must be set to 0 for all terms not modified (not initialized to 0 in nlopt)
            grad[:] = np.zeros(grad.shape)

            #grad calc - put grad in place here
            #func is .5 * xT * W * x : func' is .5 * (xT * W * WT); W*WT * .5 == W
            grad[self.qdotIDXs] = velGrad + locGrad + pVelGrad

#            if(frcWt > 0):
#                grad[self.tauIDXs] = frcGrad
            
            #grad[self.tauIDXs] = frcWt *frcDiff
        #funcRes = sqrt(np.transpose(velDiff).dot(self.qdotWts.dot(velDiff)))
#        if(self.optIters % (self.numOptIters/10) == 1):
#            print('objectiveFunc iter : {} velPart : {} | locPt : {} | pt_VelPt : {} |  res : {}'.format(self.optIters,velPart,locPart, pVelPart, funcRes))
        
        return funcRes    
    
    #per-timestep intializations not covered in setSimVals in parent class
    def setSimValsPriv(self):
        pass
    
    #equation of motion constraint
    def MAcnstVec(self, result, x, grad):
        #print('MAcnst iter')
        qdotPrime = x[self.qdotIDXs]
        tau = x[self.tauIDXs]
        #MA + c + g        
        #maRes = self.M_ovDt.dot(qdotPrime-self.skel.dq) + self.CfG        
        #M/dt dot qdotprime + ( -M/dt dot qdot + CfG)
#        self.skel.set_velocities(qdotPrime)
#        CfG = self.skel.coriolis_and_gravity_forces()
#        self.skel.set_velocities(self.curQdot)
#        if(np.allclose(CfG, self.CfG)):
#            print('CfG is not updated when qdot changed')
#        else:
#            print('Success!')
        maRes = self.M_ovDt.dot(qdotPrime) + self.Mqd + self.CfG #- self.CntrntFrc
        
        #build pull force and subtract tau estimate
        #pulling force relative to external force is negative - want opposite direction
        #in other words, we want the resultant torques to generate this force 
        tauPull = self.Tau_JtFpull - tau
        
        result[:] = maRes + tauPull     
        
        if grad.size > 0:
            #grad is matrix of len(result) rows (37) x len(x) cols (74)
            #if gradient exists, set it here.  Must be set to 0 for all terms not modified
            grad[:,:] = np.zeros(grad.shape)
            #gradient for all qdot terms
            #include change in coriollis forces
            grad[:,self.qdotIDXs] = self.M_ovDt #+ (CfG-self.CfG)
            #gradient for all taus - diagonal mat with -1 on diags            
            grad[:,self.tauIDXs] = self.tauDotMat
        #return ttlRes#np.sum(np.absolute(ttlVec))        

    #perform any tests if desired and populate tau                 
    def doStepTestsAndSetTau_OptCntrl(self):
        #set tau with best opt result
        guessTau = self.nextGuess[self.tauIDXs]
        #evaluate best guess
        objFuncEval = self.objFunc(self.nextGuess, np.empty(shape=(0,0)))
        #print('\nOpt Result Eval : {}\n'.format(objFuncEval))
        passMATest = self.dbg_testMAconstVec(self.nextGuess)
        #self.dbgShowTorques(guessTau)
        self.tau = guessTau


    #individual skeleton handling for calculating post-step dynamic state dictionary
    def _bldPstFrcDictPriv(self, frcD,indivCalcFlag):
        #seems to be always the same so far, although frwrd integrating 1st order using guess of dq yields inaccuracies in calculations
        frcD['dqDiff']=np.linalg.norm(self.nextGuess[self.qdotIDXs]-self.skel.dq)
        #total pull force of listed forces (doesn't include contact force calc if present)
        frcD['totPullFrc'] = frcD['jtDotTau'] - frcD['jtDotCGrav'] - frcD['jtDotMA']
        #target force to be seen at eef is self.use force
        frcD['targetEEFFrc'] = self.useForce
        
        #if bot is connected to constraint, include these values        
        #cnstrntFrc = self.skel.constraint_forces()
        #monitor constraint forces to see if force seen by human is appropriate pull force        
        #frcD['cnstf'] = cnstrntFrc
        #store JtPullPInv_new in dict - get constraint forces observed at eef
        #frcD['jtDotCnstFrc'] = frcD['JtPullPInv_new'].dot(cnstrntFrc)        
        #total pull force of listed forces (doesn't include contact force calc if present)
        #frcD['totPullFrcCnst'] = frcD['totPullFrc'] + frcD['jtDotCnstFrc'] 

        return frcD

    #display results of fwd simulation - test results, display calculated vs simulated force results at end effector
    def _dispFrcEefResIndiv(self):         
        pass

    #display this classes guess configuration 
    def dbgDispGuessPriv(self, guess, name=' '):
        qdotPrime = guess[self.qdotIDXs]
        alignQdotStr = self.dotAligned(qdotPrime)
        tau = guess[self.tauIDXs]   
        alignTauStr = self.dotAligned(tau)
        #should have a dof corresponding to each component of qdot and tau
        dofs = self.skel.dofs
        print('\n{} : '.format(name))
        if(len(dofs) != len(qdotPrime)):
            print('!!!!!! attempting to print qdot values that do not align with skeleton dofs (different counts)')
            return
        if(len(dofs) != len(tau)):
            print('!!!!!! attempting to print torques that do not align with skeleton dofs (different counts)')
            return
        
        print('\tQdot for dofs : ')        
        #should have a dof corresponding to each component of tau
        dofs = self.skel.dofs
        alignTauStr = self.dotAligned(tau)
        for i in range(len(dofs)):
            print('\t\tDof : {:20s} | Qdot : {}'.format(dofs[i].name, alignQdotStr[i]))
        print('\n\tTorques per dof : ')        
        for i in range(len(dofs)):
            print('\t\tDof : {:20s} | Torque : {}'.format(dofs[i].name, alignTauStr[i]))
#For KR5sixx robot
#7 Body names : 
#        base_link
#        shoulder
#        bicep
#        elbow
#        forearm
#        wrist
#        palm
#7 Joint names : 
#        arm_to_world_fixed
#        shoulder_yaw
#        shoulder_pitch
#        elbow_pitch
#        elbow_roll
#        wrist_pitch
#        wrist_roll
#6 Dof names : 
#        shoulder_yaw
#        shoulder_pitch
#        elbow_pitch
#        elbow_roll
#        wrist_pitch
#        wrist_roll    