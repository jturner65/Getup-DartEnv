#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:09:33 2018

@author: john
"""
#all skeleton holders used by dart_env_2bot

import numpy as np
import nlopt
from math import sqrt
from gym import error, spaces
#from gym.spaces import *

from abc import ABC, abstractmethod

from collections import defaultdict

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
        #ref to owning environment
        self.env = env
        #ref to skel object
        self.skel = skel
        #index in world skeleton array
        self.worldIdx = widx
        #start index for action application in tau array -> bypasses
        #dofs that are root/world related (And do not get tau applied)
        self.stTauIdx = stIdx
        #base action scale multiplier
        self.actionScaleBaseVal = 150.0
        #preloaded initial state to return to at reset, otherwise variation of initstate is used
        self.initQPos = None
        self.initQVel = None
        #idxs of dofs corresponding to world location of root - replace with com in observation, since x 
        self.rootLocDofs = np.array([3,4,5])
        #number of dofs - get rid of length calc every time needed
        self.ndofs = self.skel.ndofs
        #number of actuated dofs : number of dofs - world dofs
        self.numActDofs = self.ndofs - self.stTauIdx
        #sqrt of numActuated dofs, for reward scaling
        self.sqrtNumActDofs = np.sqrt(self.numActDofs)

        #ground friction from environment
        self.groundFric = self.env.groundFric
        #gravity of this world
        grav = self.env.dart_world.gravity()
        #print('gravity is  : {} '.format(grav))
        #precalc mg for force calculations - magnitude of grav * mass
        self.mg = np.linalg.norm(grav) * self.skel.mass()
        
        #to monitor torques seen : minimum and maximum seen torques saved in arrays
        self.monitorTorques = False
        #self._reset_monTorques()   
        self.monTrqDict = {}
        self._resetMinMaxMonitors(self.monTrqDict, self.ndofs)
        #bound on allowable qDot for training of policy
        self.qDotBnd = 10
        
        #state flags
        #use the mean state only if set to false, otherwise randomize initial state
        self.randomizeInitState = True
        #use these to show that we are holding preset initial states - only used if randomizeInitState is false
        self.loadedInitStateSet = False
        #desired force/torque has been set
        self.desForceTorqueSet = False
        #process actions and forward simulate (should match skel.is_mobile) - only use in prestep and poststep functions, if at all
        self.isFrwrdSim = True
        #whether to use linear or world jacobian for endeffector
        self.useLinJacob = False
        #Monitor generated force at pull point/hand contact
        self.monitorGenForce = False

        #display debug information for this skel holder
        self.debug = True

        #set baseline desired external forces and force mults
        self.desExtFrcVal = np.array([0,0,0])
        self.desExtFrcVal_mults = np.array([0,0,0])
        #using force multiplier instead of force as observation component of force in RL training/policy consumption
        self.useMultNotForce=False
   
        #initial torques - set to 0
        self.dbgResetTau()
        #TODO force and torque dimensions get from env
        
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
        self.numSimSteps = 0
        #timestep - do not include repeated steps 
        #since query sim for character state and sim state is step dependent        
        self.timestep = env.dart_world.dt  
        #tag to follow # of broken simulation steps
        self.env.numBrokeSims = 0
        #whether to check for best state after each reward - this is to keep around the best state from the previous rollout to use as next rollout's mean state
        self.checkBestState = False#set this true in ana
        #initialize to bogus vals
        self.bestState = None
    

    #moved from dart_env_2bot - this sets up the action and (initial) observation spaces for the skeleton.  
    # NOTE : the observation space does not initially include any external forces, 
    # and can be modified through external calls to setObsDim directly
    # this should be called once, only upon initial construction of skel handler
    def setInitialSkelParams(self, bodyNamesAra):
        #clips control to be within -1 and 1
        #min is idx 0, max is idx 1
        self.control_bounds = (np.array([[-1.0]*self.numActDofs,[1.0]*self.numActDofs])) 
        self.action_dim = self.numActDofs
        self.action_space = spaces.Box(self.control_bounds[0], self.control_bounds[1])

        #scaling value of action multiplier (action is from policy and in the range of approx +/- 2)
        action_scaleBase = np.array([self.actionScaleBaseVal]*self.numActDofs)
        #individual action scaling for different bot configurations
        action_scale = self._setupSkelSpecificActionSpace(action_scaleBase)

        print('action scale : {}'.format(action_scale))
        self.action_scale = action_scale
        #set initial observation dimension - NOTE : this does not include any external forces or observations
        self.setObsDim(2*self.ndofs)
        if (len(bodyNamesAra) > 0):
            self._setFootHandBodyNames(bodyNamesAra[0], bodyNamesAra[1], bodyNamesAra[2], bodyNamesAra[3]) 
        else :#not relevant for KR5 arm robot
            print("---- No foot/hand/head body names specified, no self.StandCOMHeight derived ----")
    
    #called for each skeleton type, to configure multipliers for skeleton-specific action space
    @abstractmethod
    def _setupSkelSpecificActionSpace(self, action_scale):
        pass

    #called initially before any pose modification is done - pose of skel by here is pose specified in skel/urdf file
    def _setFootHandBodyNames(self,lf_bdyNames, rf_bdyNames, h_bodyNames, headBodyName):
        self.feetBodyNames = lf_bdyNames + rf_bdyNames
        self.leftFootBodyNames = lf_bdyNames[:]
        self.rightFootBodyNames = rf_bdyNames[:]
        self.handBodyNames = h_bodyNames[:]
        self.headBodyName = headBodyName
        print('skelHolder::setFootHandBodyNames : set values for initial height above avg foot location  - ASSUMES CHARACTER IS UPRIGHT IN SKEL FILE. Must be performed before desired pose is set')   
        #specific to instancing class - only RL-involved skel holders should have code in this     
        self.setInitRWDValues()     
        
    def setSkelMobile(self, val):
        self.skel.set_mobile(val)
        #frwrdSim is boolean whether mobile/simulated or not
        self.isFrwrdSim = val
           
    #initialize constraint locations and references to constraint bodies
    def initConstraintLocs(self,  cPosInWorld, cBody):
        #this is initial end effector position desired in world coordinates, built relative to human pose fingertip
        self.initEffPosInWorld = np.copy(cPosInWorld)
        #initializing current location for finite diff velocity calc
        self.trackBodyCurPos = np.copy(self.initEffPosInWorld)#self.cnstrntBody.com()
        #body to be constrained to or track
        self.cnstrntBody = cBody
        #constraint location in ball local coords - convert to world for target location for inv dyn
        self.cnstrntOnBallLoc = self.cnstrntBody.to_local(x=self.initEffPosInWorld) 
        #initialize best state reward  
        if (self.checkBestState):
            self.bestStRwd = -100000000000
            self.bestState = self.state_vector()       
    
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
        
    #set the name of the body node reaching to other agent
    #called by initial init of pose
    def setReachHand(self, _rchHand):
        self.reach_hand = _rchHand
        self.reachBody = self.skel.body(self.reach_hand)
    
    #set the desired external force for this skel
    #(either the force applied to the human, or the force being applied by the robot)
    #set reciprocal is only used for debugging/displaying efficacy of force generation method for robot
    def setDesiredExtForce(self, desFrcTrqVal, desExtFrcVal_mults, setReciprocal, obsUseMultNotFrc):
        self.useMultNotForce = obsUseMultNotFrc
        #print('{} got force set to {}'.format(self.skel.name, desFrcTrqVal))
        #self.lenFrcVec = len(desFrcTrqVal)
        self.desExtFrcVal = np.copy(desFrcTrqVal)
        self.desExtFrcVal_mults = np.copy(desExtFrcVal_mults)
        self.desForceTorqueSet = True       
        #if true then apply reciprocal force to reach body
        if(setReciprocal):
            #counteracting force for debugging - remove when connected
            self.reachBody.set_ext_force(-1 * self.desExtFrcVal, _offset=self.reachBodyOffset)
    
    #set observation dimension
    def setObsDim(self, obsDim):
        self.obs_dim = obsDim
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        
    #set what this skeleton's init pose should be
    #only call this whenever initial base pose _actually chagnes_ (like upon loading)
    def setStartingPose(self):
        #initPose is base initial pose that gets varied upon reset if randomized - base pose is treated as constant 
        self.initPose = self._makeInitPoseIndiv()
        self.setToInitPose()
        
    #reset this skeleton to its initial base pose - uses preset self.initPose
    def setToInitPose(self):
        #priv method is so that helper bots can IK to appropriate location based on where ANA hand is
        self._setToInitPose_Priv()
        self.skel.set_positions(self.initPose)
        #set skel velocities
        self.skel.set_velocities(np.zeros(self.skel.dq.shape))

        #if monitoring the force/torques at pull application point, reset minmax arrays 
        if(self.monitorGenForce):
            self.totGenFrc = list()
            self.minMaxFrcDict = {}
            self._resetMinMaxMonitors(self.minMaxFrcDict, self.useForce.shape)
            #self._reset_minMaxGenForceVals()
       
        self.postPoseInit()

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
    def setClampedTau(self, a):
        self.a = np.copy(a)
        #print('pre clamped cntl : {}'.format(a))
        #idx 0 is min, idx 1 is max.
        clamped_control = np.clip(self.a, self.control_bounds[0],self.control_bounds[1])
        #print('clamped cntl : {}'.format(clamped_control))
        self.tau = np.zeros(self.ndofs)
        self.tau[self.stTauIdx:] = clamped_control * self.action_scale 
        #print('tau : {}'.format(self.tau))
        return self.tau  
       
    #send torques to skeleton
    def applyTau(self):
        if(self.monitorTorques):
            self._checkiMinMaxVals(self.tau, self.monTrqDict)               
        #apply torques every step since they are cleared after world steps
        self.skel.set_forces(self.tau)      
        #record # of tau applications
        self.numSimSteps += 1     
        #any private per-step torque application functionality - this is where the external force is applied to human skeleton during RL training to simulate robot assistant
        self.applyTau_priv()        
        
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
        frcBnds = self.env.frcBnds * self.mg
        #limits of location are arbitrary except for y, must be positive and less than ~2 - observation of rootLocDOFS is actually COM values, so won't be below ground
        jtlims['lowQLims'][self.rootLocDofs[1]] = 0
        jtlims['highQLims'][self.rootLocDofs[1]] = 2        
        
        jtlims['obs_lowBnds']=self._concat3Aras(jtlims['lowQLims'],jtlims['lowDQLims'], frcBnds[0])
        jtlims['obs_highBnds']=self._concat3Aras(jtlims['highQLims'],jtlims['highDQLims'], frcBnds[1])
        return jtlims
    
    def _concat3Aras(self, a, b,c):
        res1 = np.concatenate([a,b])
        res  = np.concatenate([res1,c])
        return res    
    
    #return a random initial state for this skeleton
    def getRandomInitState(self, poseDel=None):
        if(poseDel is None):
            poseDel=self.poseDel
        lPoseDel = -1*poseDel
        #checkBestState means instead of reusing initial state to begin rollout, we use the best state from last rollout.  NOTE need to make sure the traj ball is following it's own location, and not coupled to the hand.
        if((self.checkBestState) and (self.bestState is not None)):
            #set walker to be best state previously seen
            self.set_state_vector(self.bestState)
        else :
            #set walker to be laying on ground
            self.setToInitPose()
            
        #perturb init state and statedot
        qpos = self.skel.q + self.env.np_random.uniform(low= lPoseDel, high=poseDel, size=self.ndofs)
        qvel = self.skel.dq + self.env.np_random.uniform(low= lPoseDel, high=poseDel, size=self.ndofs)
        return qpos, qvel

    #returns a random observation, based on the full range of possible q and qdot, governed by joint limits and joint vel limits, if any exist
    def getRandomObservation(self):
        #get all known joint limits
        jtlims = self.getObsLimits()
        #print('{}'.format(jtlims))
        rndQ = self.env.np_random.uniform(low=jtlims['lowQLims'],high=jtlims['highQLims'])
#        if not (np.isfinite(jtlims['lowDQLims']).all()) or not (np.isfinite(jtlims['highDQLims']).all()) :
#            rndQDot = self.env.np_random.uniform(low=-self.qDotBnd, high= self.qDotBnd, size=self.ndofs)
#        else :
        rndQDot = self.env.np_random.uniform(low=jtlims['lowDQLims'],high=jtlims['highDQLims'])
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

    def dispResetDebug(self, notice=''):
        print('{} Notice : Setting specified init q/qdot/frc'.format(notice))
        print('initQPos : {}'.format(self.initQPos))
        print('initQVel : {}'.format(self.initQVel))
        print('initFrc : {}'.format(self.desExtFrcVal))

    #this will return the force, or force multiplier, currently being used as the final component of the observation
    def getObsForce(self):
        #this is specified in environment constructor
        if self.useMultNotForce : 
            return self.desExtFrcVal_mults
        else :
            return self.desExtFrcVal

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
            qpos, qvel = self.getRandomInitState()
            self.set_state(qpos, qvel)
        else:
            #reset to be in initial pose
            self.setToInitPose()
            #resetting to pre-set initial pose
            if (self.loadedInitStateSet):
                if(dispDebug):
                    self.dispResetDebug('skelHolder::reset_model')
                self.set_state(self.initQPos, self.initQVel)
                self.loadedInitStateSet = False
#            else:
#                print('skelHolder {}::reset_model Warning : init skel state not randomized nor set to precalced random state'.format(self.skel.name))
                
        if(self.monitorTorques):
            #self._reset_monTorques()
            self.monTrqDict = {}
            self._resetMinMaxMonitors(self.monTrqDict, self.ndofs)

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

    #handle if sim is broken
    def _handleBrokenSim(self, resDict):
        print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('{} has broken sim : reason : {}'.format(self.name, resDict['reason']))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        done = True
        rwd = 0
        d = {'broke_sim': True, 'raiseVelScore': -10000000,'height_rew':-10000000, 'actionPenalty':-10000000, 'done':True}
        self.env.numBrokeSims+=1
        #observation of current state
        obs = self.getObs()
        return obs, rwd, done, d, {}

    #functionality after sim step is taken - 
    #calculate reward, determine if done(temrination conditions) and return observation
    #and return informational dictionary
    def postStep(self, resDict):
        #resDict holds whether the sim was broken or not - if sim breaks, we need 
        #holds these values : {'broken':False, 'frame':n_frames, 'stableStates':stblState}   
        #check first if sim is broken - illegal actions or otherwise exploding   
        if(resDict['broken'] and (self.name in resDict['skelhndlr'])):
            return self._handleBrokenSim(resDict)

        debug = False
        #not broken                    
        rwd, done, d, dbgDict = self.calcRewardAndCheckDone(debug)
        #observation of current state
        obs = self.getObs()
        if (self.checkBestState) and (rwd > self.bestStRwd):
            self.bestStRwd = rwd
            self.bestState = self.state_vector()

        #ob, reward, done, infoDict
        return obs, rwd, done, d, dbgDict
    
    #calculate the distance between the COm projected onto the level of the COP and the COP itself
    #if no COP val passed, derive current COP val
    def calcCOMCOPDist(self, COM, COPVal):
        COMatCOPHt = np.copy(COM)
        COMatCOPHt[1] = COPVal[1]     
        #distance is just x/z
        distCOMCop = np.linalg.norm(COMatCOPHt - COPVal)
        #distance between COM projected to plane of foot and COP - want to be small
        return distCOMCop
    
    #return skeleton qdot - maybe clipped, maybe not
    def getSkelqDot(self):
        return np.clip(self.skel.dq, -self.qDotBnd , self.qDotBnd )
    
    #base check goal functionality - this should be same for all agents,
    #access by super()
    def checkSimIsBroken(self):
        s = self.state_vector()
        if not(self.isFrwrdSim):#if not frwrd simed then sim won't be broken
            return False, s, 'FINE'

        if not (np.isfinite(s).all()):
            return True, s, 'INFINITE'
        if not ((np.abs(s[2:]) < 1000).all()):
            return True, s, 'EXPLODE'

        return False, s, 'FINE'
    
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
        if ('ground' in bn1.name) :
            return (self.checkBNinNameList(bn2.name, self.feetBodyNames) and (self.skel.name == bn2.skel.name)), bn2
        elif ('ground' in bn2.name) :
            return (self.checkBNinNameList(bn1.name, self.feetBodyNames) and (self.skel.name == bn1.skel.name)), bn1        
        return False, None    
        
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
            avgLFootLoc /= len(self.rightFootBodyNames)           
            #for each body considered part of a "foot"
#            for ftBdyName in self.feetBodyNames:
#                avgFootLoc += self.skel.body(ftBdyName).com()
            avgFootLoc /= len(self.feetBodyNames)
        else :
            print('skelHolder::calcAvgFootBodyLoc : No feet for skelholder {} defined in self.feetBodyNames so avg loc is origin'.format(self.name))
        return [avgFootLoc, avgLFootLoc, avgRtFootLoc]

    #calculate contact's contribution to total contact torques around distant point for COP derivation
    #cntctBN : the body node in contact with the ground
    #contact : the contact object
    def calcSumTrqsForCOP(self, cntctBN, contact):
        trqPt = contact.point - self.env.ctrTauPt
#        if(contact.point[1] != 0):
#              print('!!!!!!!!!!!!!!!!!!!!!! non zero ground contact y : {}'.format(contact.point))
        new_COP_tau = np.cross(trqPt, contact.force)
        new_COP_ttlFrc = contact.force
        return new_COP_tau,new_COP_ttlFrc
        
    
    #calculate foot contact count and other terms if we want to use them for reward calc
    def calcAllContactData(self):
        contacts = self.env.dart_world.collision_result.contacts        
        contactDict = defaultdict(float)       
               
        COPval = np.zeros(3)
        COP_tau = np.zeros(3)
        COP_ttlFrc = np.zeros(3)
        #tau = loc x frc
        #COP calculation is the location that, when crossed with total force, will give total moment
        #we can calculate total moment by crossing all contact forces with all contact locations
        #we can get total force, and from this we can find the location that would produce the total 
        #torque given the total force (by first constraining one of the dimensions of the point in question)
        #we choose to constrain the y coordinate to be 0 since we want the cop on the ground
       
        for contact in contacts:
            if (self.skel.name != contact.bodynode1.skeleton.name ) and (self.skel.name != contact.bodynode2.skeleton.name ) :
                #contact not from this skeleton
                continue
            #penalize contact between the two skeletons - getup-human should not touch assistant bot except with assisting hand
            #only true if one skel contacts other and the other is not the ground - i.e. non-ground non-self contact
            if self.checkBNKickOtr(contact.bodynode1, contact.bodynode2):
                contactDict['kickBotContacts'] +=1            
            #only true if feet are contacting skeleton - kicking self - i.e. non-ground self contact
            elif (contact.bodynode1.skel.name == contact.bodynode2.skel.name):
                contactDict['selfContact'] +=1
            else:
                #this is a contact between this skeleton with the ground
                #cntctBN is skeleton body node responsible for ground contact, or none if no ground contact
                isGndCntctWFoot, cntctBN = self.checkMyFootWithGround(contact.bodynode1,contact.bodynode2)                 
                #save refs to each contact force and which body had contact with ground
                contactDict[cntctBN.name] +=1 
                if isGndCntctWFoot: #save for COP calc
                    #foot contact with ground     
                    contactDict['COPcontacts'] += 1
                    contactDict['footGroundContacts'] +=1
                    #print('With Ground : contact body 1 : {} skel : {} | contact body 2 : {} skel : {} : frc {}'.format(contact.bodynode1,contact.bodynode1.skeleton.name, contact.bodynode2,contact.bodynode2.skeleton.name, contact.force))
                    #COP shouldnt use butt contact in calculation - want a target to get up
                    #find total moment of all contacts
                    cTau, cTtlFrc = self.calcSumTrqsForCOP(cntctBN, contact)
                    COP_tau += cTau
                    COP_ttlFrc += cTtlFrc                    
                elif(self.checkBNinNameList(cntctBN.name, self.handBodyNames)):
                    #hand contact with ground
                    contactDict['COPcontacts'] += 1
                    contactDict['handGroundContacts']+=1  
                    #non-foot contact with ground - check if contact with non-reaching hand
                    #print('Not Foot With Ground : contact body 1 : {} skel : {} | contact body 2 : {} skel : {}'.format(contact.bodynode1,contact.bodynode1.skeleton.name, contact.bodynode2,contact.bodynode2.skeleton.name))
                    #find total moment of all contacts of hand to ground
                    cTau, cTtlFrc = self.calcSumTrqsForCOP(cntctBN, contact)
                    COP_tau += cTau
                    COP_ttlFrc += cTtlFrc                    
                else :
                    #nonHand nonFoot Contact with ground 
                    contactDict['badGroundContact']+=1
        #find average foot location (idx 0)
        avgFootLocAra = self.calcAvgFootBodyLoc()
        #determines COP based on foot contacts with ground - ignore non-foot contacts for COP calc
        #COP loc might be redundant - using foot location might be sufficient
        if((0 < contactDict['COPcontacts']) and (np.abs(COP_ttlFrc[1]) > 0)):#(COP_ttlFrc[1] > 0) and (COP_ttlFrc[0] > 0)):
            #COP_tau = COPval cross COP_ttlFrc ==> need to constrain possible COPvals -> set COPval.y == 0 since we want the COP at the ground, and then solve eqs : 
            COPval = self.calcCOPFromTauFrc(COP_tau, COP_ttlFrc, self.env.ctrTauPt)
        else :  #estimate COP as center of both feet body node com locations         
            COPval = np.copy(avgFootLocAra[0])
            #put it on the ground 
            COPval[1]= 0

        return contactDict, COPval, avgFootLocAra    

    #Take total moments, total force, return a suitable point of application of that force to provide that moment - more than 1 answer, want answer on ground plane
    #COP_tau = COPval cross COP_ttlFrc ==> need to constrain possible COPvals
    #so set COPval.y == 0 since we want the COP at the ground, and then solve eqs :
    #ctrTauPt is point far away around which torques are initially calculated, then summed, and then reversed to find COP loc      
    #ctrTauPt[1] should be 0, since can be chosen arbitrarily    
    def calcCOPFromTauFrc(self,COP_tau, COP_ttlFrc, ctrTauPt):
        #given a cross b (==c : total tau) and b (total force), find a (location vector) : a = (bxc)/(b.b)  + tb, where t is any real scalar
        COPvalTmp = np.cross(COP_ttlFrc,COP_tau)/np.dot(COP_ttlFrc,COP_ttlFrc)#soln for t==0
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
        return COPval       
    
#    def calcCOPFromTauFrc(self,COP_tau, COP_ttlFrc, ctrTauPt):
#        COPval = np.zeros(3)        
#        COPval[0] = COP_tau[2]/COP_ttlFrc[1] + ctrTauPt[0]
#        COPval[2] = COP_tau[1]/COP_ttlFrc[0] + (COP_ttlFrc[2] * COPval[0]/COP_ttlFrc[0]) + ctrTauPt[2]
#        return COPval       
    
    #return the end effector location in world coords and the constraint location in  world coords
    def getCnstEffLocs(self):
        trackBodyPos = self.cnstrntBody.to_world(x=self.cnstrntOnBallLoc) #self.cnstrntBody.com()
        curEffPos = self.reachBody.to_world(x=self.reachBodyOffset)
        return trackBodyPos, curEffPos
    
    
    #return body torques to provide self.desExtFrcVal at toWorld(self.constraintLoc)
    #provides JtransFpull component of equation
    def getPullTau(self, useLinJacob, debug=False):     
        if (useLinJacob) :             
            self.useForce = self.desExtFrcVal
            #using only linear : 3 rows x ndofs cols
            Jpull = self.reachBody.linear_jacobian(offset=self.reachBodyOffset)            
        else : 
            #wrench
            #TODO verify target orientation should be 0,0,0
            self.useForce = np.zeros(6)
            self.useForce[3:]=self.desExtFrcVal
             #using linear and rotational == world
            Jpull = self.reachBody.world_jacobian(offset=self.reachBodyOffset) 
        if(debug):
            print('getPullTau : pull force being used : {} '.format(self.useForce))

        JTrans = np.transpose(Jpull)
        res = JTrans.dot(self.useForce)
        JTransInv = np.linalg.pinv(JTrans)
        #last 3 rows as lin component
        return res, JTransInv, Jpull, Jpull[-3:,:]
    
    
    ######################################################
    #   abstract methods 
    ######################################################    
    
    #build the configuration of the initial pose of the figure
    @abstractmethod
    def _makeInitPoseIndiv(self):
        pass       
    
    #need to apply tau every step  of sim since dart clears all forces afterward;
    # this is for conducting any other per-sim step functions specific to skelHldr (like reapplying assist force)
    @abstractmethod
    def applyTau_priv(self):
        pass        
    
    #special considerations for init pose setting - this is called every time skeleton is rest to initial pose
    @abstractmethod
    def _setToInitPose_Priv(self):
        pass

    #setup initial constructions for reward value calculations
    @abstractmethod
    def setInitRWDValues(self):
        pass

    #init to be called after skeleton pose is set
    @abstractmethod
    def postPoseInit(self):
        pass
    
    #functionality necessary before simulation step is executed    @abstractmethod
    def preStep(self, a):
        pass        
    
    #individual instance class reset functionality called at end of reset
    @abstractmethod
    def _resetIndiv(self, dispDebug):
        pass    
    
    #get the state observation from this skeleton - concatenate to whatever extra info we are sending as observation
    @abstractmethod
    def getObs(self):
        pass
    
    #calculate reward for this agent, see if it is done, and return informational dictionary (holding components of reward for example)
    @abstractmethod
    def calcRewardAndCheckDone(self, debug):
        pass
    
    
    ######################################################
    #   debug and helper functions
    ######################################################
    
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
            print('\tDof : {:20s} | Value : {} | Action : {:.5f}'.format(dofs[i].name, alignTauStr[i],0))   
        for i in range(self.stTauIdx,len(dofs)):
            print('\tDof : {:20s} | Value : {} | Action : {}'.format(dofs[i].name, alignTauStr[i],alignAStr[(i-self.stTauIdx)]))   
    
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
        numDofs = self.skel.num_dofs()
        print('\n{} Dof names and limit values : '.format(numDofs))
        for bidx in range(numDofs):
            d = self.skel.dof(bidx)
            print('{} :\tMin : {:.3f}\tMax : {:.3f}\tMin Vel : {:.3f}\tMax Vel : {:.3f}'.format(d.name, lims['lowQLims'][bidx], lims['highQLims'][bidx], lims['lowDQLims'][bidx], lims['highDQLims'][bidx]))    


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

    
#class for skeleton holder specifically for the getup human
class ANASkelHolder(skelHolder, ABC):
    #Static list of names of implemented reward components
    rwdNames = ['eefDist','action','height','footMovDist','lFootMovDist','rFootMovDist','comcop','comcopvel','contacts','UP_COMVEL','X_COMVEL','Z_COMVEL']

                 
    def __init__(self, env, skel, widx, stIdx, fTipOffset):          
        skelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset)
        self.name = 'ANA : Agent Needing Assistance'
        #set this so that bot doesn't get stuck in limbo REMOVE
        #TODO : remove, repalce with vector ?
        self.minUpVel = .001
     
        #set to true to initialize assist force in apply_tau for training, set to false when using robot assistant
        self.setAssistFrcEveryTauApply = False
        #this is full time step - since human is only skel to be simulated with same actions multiple frames, human is only consumer of this
        self.dtForAllFrames = self.env.dt        
        #this is # of sim steps in rollout before displaying debug info about reward
        self.numStepsDBGDisp = 101

        #distance thresholds 
        self.rwdTols = defaultdict(float)
        # for end effector matching to ball location - distances above this are penalized harshly, distances below this are penalized very gently
        self.rwdTols['eefDist'] = .05
        # for com proj on ground dist to cop on ground - 10 cms
        self.rwdTols['comcop'] = .1
        #reward matrix base for action minimization - set joint dof idxs that we don't care as much about for minimization to values between 0 and 1
        self.actPenDofWts = np.identity(self.numActDofs)
        
        #which com key to use for height calculations - currently 'com' and 'head'
        self.comHtKey = 'com' 
        # or 
        #self.comHtKey = 'head'
        #whether or not to use best state from previous rollout as initial state on this rollout
        self.checkBestState = False
        
        ##########################################
        #   reward function weights and var/scales for each component
        #   default value is 1.0

        #specify weight and var dict names allowed
        #rwdNames = ['eefDist','action','height','footMovDist','lFootMovDist','rFootMovDist','comcop','comcopvel','contacts','UP_COMVEL','X_COMVEL','Z_COMVEL']
        #specify non-default values in default dict - not using default dict because it might hide bugs where wrong or unknown reward is called for wt/var
        wtVals = defaultdict(lambda:1.0,{'eefDist':6.0,'action':.1,'height':10.0,'footMovDist':6.0,'lFootMovDist':6.0,'rFootMovDist':6.0,'UP_COMVEL':10.0, 'comcop':.01})
        varVals = defaultdict(lambda:.1,{'action' : (1.0*self.sqrtNumActDofs), 'height':.5})
        #this is list of components used used for reward function
        rwdFuncsToUse = self.env.rwdCompsUsed
        self.setRwdWtsVars(names=ANASkelHolder.rwdNames, wts=wtVals, varVals=varVals, rwdsToUse=rwdFuncsToUse)

    #set weighting vectors for reward wts and variance/scales - 
    # uses dictionary and not default dict, so that any unrecognized weight values throw an exception - we don't want any unrecognized wt or var vals
    #names are the names of the rwd components getting set
    #wts is list of wt
    def setRwdWtsVars(self, names, wts, varVals, rwdsToUse):
        self.rwdWts = {}
        self.rwdVars = {}
        #specify reward components to check
        self.rwdsToCheck = {}
        for i in range(len(names)) :
            k = names[i]
            self.rwdWts[k] = wts[k]
            self.rwdVars[k] = varVals[k]
            self.rwdsToCheck[k] = (k in rwdsToUse)
        
    #setup initial constructions for reward value calculations - called before pose is set
    def setInitRWDValues(self):
        avgInitFootLocAra = self.calcAvgFootBodyLoc()
        #both feet avg location
        avgInitFootLoc = avgInitFootLocAra[0]
        #initial height - subtract this so we only reward increases in height -  0.87790457356624751
        #assumes unmodified base skeleton pose in skel file is standing upright
        #dict of bodies to be used for height calcs
        self.comHtBodyDict = {}        
        self.comHtBodyDict['com'] = self.skel
        self.comHtBodyDict['head'] = self.skel.body(self.headBodyName)
        self.standHtOvFtDict = {}
        #dict of heights standing of various bodies to be used for height calcs
        for k,v in self.comHtBodyDict.items():
            self.standHtOvFtDict[k] = v.com()[1] - avgInitFootLoc[1]#0.87790457356624751 for body COM
        print('Init StandCOMHeight before pose (skel expected to be standing upright from src file config) - height above avg foot location : {}'.format(self.standHtOvFtDict['com']))  
        
        #dicts to hold min and max com velocity vectors for the various tracked skel bodies
        self.minCOMVelVals = {}
        self.maxCOMVelVals = {}
        #min and max com vel vals to receive positive rewards - roots of parabola
        #these should not be the same (min==max) ever
        self.minCOMVelVals['com'] = np.array([-.3,.001,-.1])
        self.minCOMVelVals['head'] = np.array([-.3,.001,-.1])
        self.minCOMVelVals['comcop'] = .01
        self.maxCOMVelVals['com'] = np.array([.9,1.75,.1])        
        self.maxCOMVelVals['head'] = np.array([.9,1.75,.1])  
        self.maxCOMVelVals['comcop'] = 1.0
       
    #build action penalty weight matrix - set to 1 for full penalty vs dof, set to 0<= vals[dof] < 1 to minimally penalize actions in certain dofs
    def buildActPenDofWtMat(self, idxs, vals):
        i=0
        for idx in idxs :
            self.actPenDofWts[idx,idx] = vals[i]
            i+=1
            
    #return current human fingertip + offset in world coordinates
    def getHumanCnstrntLocOffsetWorld(self, offset=0):
        #want further along negative y axis
        loc = np.copy(self.reachBodyOffset)
        if(offset != 0):
            loc += np.array([0,-offset,0])
        return self.reachBody.to_world(x=loc)
    
    #ANA does nothing here    
    def _setToInitPose_Priv(self):
        pass    
    
    #use this to set trajectory ball location self.comHtKey
    def getRaiseProgress(self):
        k = self.comHtKey#what body to use for COM height calc
        d = (self.comHtBodyDict[k].com()[1] - self.stSeatHtOvFtDict[k])/self.htDistToMoveDict[k]
        if d < 0:            
            d = 0
        if d > 1 :
            d = 1             
        return d    

    #called after pose is set - set up initial heights for seated COM and seated head COM
    def postPoseInit(self):
        #will fail if run for kr5 arm robot
        self.initFootLocAra = self.calcAvgFootBodyLoc()
        #initial seated height above avg foot loc
        self.stSeatHtOvFtDict = {}
        self.htDistToMoveDict = {}
        for k,v in self.standHtOvFtDict.items():
            self.stSeatHtOvFtDict[k] = self.comHtBodyDict[k].com()[1] - self.initFootLocAra[0][1]
            self.htDistToMoveDict[k] = (.99 * v) - self.stSeatHtOvFtDict[k]
            if(self.htDistToMoveDict[k] < 0) :
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!! ANASkelHolder::postPoseInit : Problem with height difference calc in ANASkelHolder {}.postPoseInit() : key:{} seated Height : {} standing height : {}'.format(self.name,k, self.stSeatHtOvFtDict[k],v))
        #print('ANASkelHolder::postPoseInit : Seated Pose Initial body COM height above avg foot loc {}'.format(self.stSeatHtOvFtDict['com']))
    
    def getObs(self):
        frcObs = self.getObsForce()
        state =  np.concatenate([
            self.skel.q,
            self.getSkelqDot(),
            #need force as part of observation!
            frcObs
        ])
        #assign COM to state in idxs of root dofs world location
        state[self.rootLocDofs] = self.skel.com()
        return state

    #get-up skel's individual per-reset settings
    def _resetIndiv(self, dispDebug):
        #initial copcom distance (COM projected on ground to COP on ground) - want this to always get smaller
        #set initial COMCOPDist for COMCOP loc and velocity calcs
        _,COPVal, _= self.calcAllContactData()
        self.COMCOPDist = self.calcCOMCOPDist(self.comHtBodyDict[self.comHtKey].com(),COPVal)  

     
    #need to apply tau every step  of sim since dart clears all forces afterward
    def applyTau_priv(self):
        #must reapply external force every step as well, if not being assisted - use only when robot not applying force
        if(self.setAssistFrcEveryTauApply) : 
            #print('Apply assist frc : {}'.format(self.desExtFrcVal))
            self.reachBody.set_ext_force(self.desExtFrcVal, _offset=self.reachBodyOffset)
        pass
    
    #functionality necessary before simulation step is executed for the human needing assistance
    def preStep(self, a):
        #get all measured com values before forward step   
        #eventually only use 1
        self.com_b4 = np.copy(self.comHtBodyDict[self.comHtKey].com())    
        #set torques
        self.tau=self.setClampedTau(a) 
        
    #returns parabolioc score for each velocity - minXVal and maxXVal hold roots of parabola for each direction of velocity
    #any minX < x < maxX will have positive reward, outside that range reward will be negative
    def getRwd_calcVelScores(self, newCom, oldCom, minXVal, maxXVal, dt):
        wt = [self.rwdWts['X_COMVEL'], self.rwdWts['UP_COMVEL'], self.rwdWts['Z_COMVEL']]
        #var = [self.rwdVars['X_COMVEL'],self.rwdVars['UP_COMVEL'],self.rwdVars['Z_COMVEL']]       
        
        #idx 0 == x, idx 1 == y, idx 2 == z
        #com velocity over all frames since action applied
        comVel = (newCom - oldCom) / dt
        comHeight = newCom[1]        
        comVelScore = np.zeros(comVel.shape)        
        for i in range(len(comVel)):
            #max rwd now managed by wt multiplier  
            comVelScore[i] = wt[i] * self.calcRwdRootMethod(comVel[i],minXVal[i], maxXVal[i])
        return comHeight, comVel, comVelScore    
    
    #use exponential formation to determine reward for dof-weighted actions (optVal) - either actions or scaled torques (scale torques to be within +/-1 per dof)
    #var=sqrtNumActDofs seemed to scale appropriately compared to 1/a^2 reward
    #will return 0 if optVal.optVal is 0, rapidly increasing value if >0
    #TO USE TREAT AS PENALTY - make negative
    def getRwd_expActionMin(self, optVal):
        wt = self.rwdWts['action']
        var = self.rwdVars['action']
        #either minimize torque or action from controller
        val = optVal.dot(optVal)
        return self.getRwd_expCalc(val, var=var,offset=1, wt=wt), val  
    
    #use exponential formation to determine reward for dof-weighted actions (optVal) - either actions or scaled torques (scale torques to be within +/-1 per dof)
    #var=sqrtNumActDofs seemed to scale appropriately compared to 1/a^2 reward
    #will return 0 if optVal.W.optVal is 0, rapidly increasing value if >0 (the more action, the higher the value)
    #TO USE TREAT AS PENALTY - make negative
    def getRwd_expDofWtActionMin(self, optVal):
        wt = self.rwdWts['action']
        var = self.rwdVars['action']
        #either minimize torque or action from controller - actPenDofWts is diagonal matrix that weights penalty of different dofs.  most diags are 1, elbow and knees are .1
        tmpVal = self.actPenDofWts.dot(optVal)
        val = np.transpose(optVal).dot(tmpVal)
        return self.getRwd_expCalc(val, var=var,offset=1, wt=wt), val        
    
    #calculate penalties for distance of average foot location (both feet, left, and right) and put in array
    #var and wt are lists of values for each component of penalty list
    #will return 0 if footMovDist[i]==0, and rapidly increasingly positive value if footMovDist[i] > 0   
    #increasing var from .1 creates more gentle curve, less positive reward and less negative penalty
    #TO USE TREAT AS PENALTY - make negative
    def getRwd_expFootMoveRew(self, AvgFootLocAra):
        wt = [self.rwdWts['footMovDist'],self.rwdWts['lFootMovDist'],self.rwdWts['rFootMovDist']]  
        var = [self.rwdVars['footMovDist'],self.rwdVars['lFootMovDist'],self.rwdVars['rFootMovDist']]       
        footMovDist = [0,0,0]
        footMoveRwd = [0,0,0]
        for i in range(3):
            footMovDist[i] = np.linalg.norm(AvgFootLocAra[i] - self.initFootLocAra[i])
            footMoveRwd[i] = self.getRwd_expCalc(footMovDist[i], var=var[i], offset=1, wt=wt[i])
        return footMoveRwd, footMovDist
    
    # #returns reward for heightAboveAvgFoot being higher than initial seated height above avg foot loc   
    # #build exponential reward function based on height difference between height now and target height : MAKE SURE THESE ARE MEASURING THE SAME THING
    # #wt is weight of this reward function result
    # #will return slight value < 0 if heightDiff <0, 0 if heightDiff==0, and rapidly increasingly positive value if heightDiff > 0 
    # #increasing var from .1 creates more gentle curve, less positive reward and less negative penalty
    # #tarHeight is where we want to be
    # def getRwd_expHtBelowStandCOM(self, heightNow, tarHeight):
    #     wt = self.rwdWts['height']
    #     var = self.rwdVars['height']
    #     #get standing COM over feet - heightAboveAvgFoot - this shows distance to go
    #     heightDiff = (tarHeight - heightNow)
    #     return self.getRwd_expCalc(heightDiff, offset=1, var=var, wt=wt), heightDiff    

    #returns reward for heightAboveAvgFoot being higher than initial seated height above avg foot loc   
    #build exponential reward function based on height difference between height now and target height : MAKE SURE THESE ARE MEASURING THE SAME THING
    #wt is weight of this reward function result
    #will return slight value < 0 if heightDiff <0, 0 if heightDiff==0, and rapidly increasingly positive value if heightDiff > 0 
    #increasing var from .1 creates more gentle curve, less positive reward and less negative penalty
    #tarHeight is where we want to be
    def getRwd_expHtAboveInitCOM(self, heightNow, startHeight):
        wt = self.rwdWts['height']
        var = self.rwdVars['height']
        #get standing COM over feet - heightAboveAvgFoot - this shows distance to go
        heightDiff = (heightNow-startHeight)
        return self.getRwd_expCalc(heightDiff, offset=1, var=var, wt=wt), heightDiff   
    
    #calc reward for difference of end effector location from tracking ball location
    #tol is what part equates to 1 i.e. what part is considered close enough
    #will return slightly negative value (>=-wt) if locDiff <0, 0 if locDiff==0, and rapidly increasingly positive value if locDiff > 0 
    #increasing var from .1 creates more gentle curve, less positive reward and less negative penalty
    #TO USE TREAT AS PENALTY - make negative
    def getRwd_expEffLocDist(self):
        wt = self.rwdWts['eefDist'] 
        var = self.rwdVars['eefDist']
        tol = self.rwdTols['eefDist']
        ballLoc,handLoc = self.getCnstEffLocs()
        #find vector from target to current location - want to reward very highly distance closer than tolerance
        locDiff = np.linalg.norm((handLoc - ballLoc)) - tol
        return self.getRwd_expCalc(locDiff, var=var, offset=1, wt=wt), locDiff  
        
    #calculate penalty based on distance of COM projection at feet to center of feet
    #and reward based on velocity of convergence (decrease in this distance)
    #COM is com value to use
    #COP is current COP value
    #COMCOPDistRwd will return slightly positive value (>=-wt) if COMCOPDistVal <0 (closer than tol), 0 if COMCOPDistVal==0, and rapidly increasingly negative value if COMCOPDistVal > 0 
    #COMCOP_VelRwd gives positive reward if within min and max bnds, negative value if outside these bounds
    def getRwd_expCOMCOPDist(self, COM, COPVal):
        wt = self.rwdWts['comcop'] 
        var = self.rwdVars['comcop']
        tol = self.rwdTols['comcop']
        #new COM->COP distance, COM projected to ground and COP/avg foot loc projected to ground
        COMCOPDist = self.calcCOMCOPDist(COM, COPVal)
        #new dist needs to be smaller than old dist - this needs to be positive
        COMCOPDist_speed = (self.COMCOPDist - COMCOPDist)  / self.dtForAllFrames  
        #bounding convergence speed so we don't get the baseline diverging
        #positive reward if within min and max bnds (max is self.rwdWts['comcopvel']), negative value if outside these bounds
        COMCOP_VelRwd = self.rwdWts['comcopvel'] * self.calcRwdRootMethod(COMCOPDist_speed, self.minCOMVelVals['comcop'], self.maxCOMVelVals['comcop']) 
        #find if distance is within tolerance value
        COMCOPDistVal = COMCOPDist - tol
        #multiply by -1 to get appropriate sign - slightly positive reward if within tolerance, increasingly negative penalty if outside tolerance. 
        # tolerance represents area around COP
        COMCOP_DistRew = -self.getRwd_expCalc(COMCOPDistVal, var=var, offset=1, wt=wt)  
        #keep around for next timestep
        self.COMCOPDist = COMCOPDist        
        return COMCOP_VelRwd, COMCOP_DistRew, COMCOPDist_speed        
    
    #will return value <offset (>=-wt*offset) if chkVal/var = 1, 0 if chkVal==0, and rapidly increasingly positive value if chkVal > 0   
    #increasing var from .1 creates more gentle curve, less slope on both sides of 1
    def getRwd_expCalc(self, chkVal, wt, offset, var):
        rew = wt * (np.exp(chkVal/var)-offset)
        return rew

    #reward that minimizes action and maximizes left hand COM height
    def calcMinActionLeftHandHigh(self, numItersDBG=200):
        #actionRew, actMag = self.getRwd_actionMin(constVal=1,optVal=self.a, mult=1.0) 
        ##weighting reach elbow and both knees much less in contribution (to allow for higher magnitude torque on those joints) 
        actionRew, actSqMag = self.getRwd_expDofWtActionMin(optVal=self.a)
        lHandCOM = self.skel.body(self.handBodyNames[0]).com()
        handRew = (10.0*lHandCOM[1]*lHandCOM[1])+5
        reward = actionRew +  handRew
        # if(self.env.steps % 1 == 0):
        #     print('Only Action And Hand ht Reward @ step {} : {}\tActionRwd : {:.3f}(action sq magnitude : {:.3f})\tHand ht Reward : {:.3f}'.format( self.env.steps, reward, actionRew, actMag, handRew))
         #never done from action, just keep going until 
        done = (self.env.steps >= 200)
        dct = {'broke_sim': False, 'raiseVelScore': 0, 
            'height_rew':0, 'actionReward':actionRew,'done':done}
    
        return reward, done, dct 
    
    #use formulation from GAE paper
    def calcGAE_MinActionMaxHeadHtAboveFt(self,numItersDBG=100):
        trqSqMag = (self.tau.dot(self.tau))
        actionRew = 1e-5 * trqSqMag
        avgFootLocAra = self.calcAvgFootBodyLoc()
        #both feet avg location
        avgFootLoc = avgFootLocAra[0]

        headHeightOvFoot = self.comHtBodyDict['head'].com()[1] - avgFootLoc[1]
        heightRew =  (self.standHtOvFtDict['head'] - headHeightOvFoot)**2

        reward = -actionRew - heightRew        
        if(self.env.steps % 1 == 0):
             print('GAE based rwd (no assist force) @ step {} : {:.3f}\tActionRwd : {:.5f}\ttorque Mag : {:.5f}\tHead ht Reward : {:.5f}\thead ht over avg foot loc : {:.5f}'.format(self.env.steps,reward,actionRew,np.sqrt(trqSqMag),heightRew, headHeightOvFoot))  
             self.dbgShowTauAndA()
             print('\n')
        done = heightRew < .0025 #closer than 5 cm from full upright
        # success = (heightAboveAvgFoot > .95*self.standCOMHeightOvFt)
        # failure = (self.env.steps >= 100)
        # done = success or failure
        # if done and failure : 
        #     reward = 0
        dct = {'broke_sim': False, 'raiseVelScore': 0, 
           'height_rew':0, 'actionReward':actionRew,'done':done}    
        return reward, done, dct   

    #process results from reward component check
    #ttlReward : current total reward
    #isDone : current done condition
    #rwdType : type of this reward
    #rwdComp : reward component value
    #rwdValWatched : list of values used by rwd calculation to determine reward
    #succCond, failCond : reward component specific termination conditions (success or failure)
    #debug : whether this is in debug mode or not
    #dct : dictionary of reward types used holding reward components and list of values used to calculate reward, (dict of lists, with first idx == reward and 2nd idx == list of values watched)
    #dbgStructs : dictionary holding lists of values used for debugging
    def procRwdCmpRes(self, ttlReward, isGoodDone, isBadDone, rwdType, rwdComp, rwdValWatched, rwdValWtchdStr, succCond, failCond, debug, dct, dbgStructs):
        ttlReward += rwdComp
        #checks for success or failure - should be boolean eqs - set to false if no conditions impact done
        isGoodDone = isGoodDone or succCond
        isBadDone = isBadDone or failCond
        if(len(rwdValWatched) > 1):
            strValTmp = ', '.join(['{:3f}'.format(x) for x in rwdValWatched])
        else :
            strValTmp = rwdValWtchdStr
        strVal = '{0} rwd/pen : {1:3f} | {0} watched vals : {2}'.format(rwdType, rwdComp,strValTmp)


        dct[rwdType] = [rwdComp, rwdValWatched]

        if (debug):
            dbgStructs['rwdComps'].append(rwdComp)
            dbgStructs['rwdTyps'].append(rwdType)
            dbgStructs['successDone'].append(succCond)
            dbgStructs['failDone'].append(failCond)
            dbgStructs['dbgStrList'][0] += '{}\t'.format(strVal)
            dbgStructs['dbgStrList'].append(strVal)  

        return ttlReward, isGoodDone, isBadDone

    #calculate reward based on what rwrd funcs are specified
    def calcRewardAndCheckDone(self, debug):
        numSteps = self.env.steps
        reward = 0
        # if (numSteps % 1== 0):
        #     debug=True
        dbgStructs = defaultdict(list)
        dbgStructs['dbgStrList'].append('\tRwd Comps :\t')
        isGoodDone = False
        isBadDone = False
        dct = defaultdict(list)
        #most reward components use foot location and current com
        avgFootLocAra = self.calcAvgFootBodyLoc()
        com_now = self.comHtBodyDict[self.comHtKey].com()
        #print('All Rewards : {}'.format(self.rwdsToCheck))
        # all implemented rewards :  rwdNames = ['eefDist','action','height','footMovDist','lFootMovDist','rFootMovDist','comcop','comcopvel','UP_COMVEL','X_COMVEL','Z_COMVEL']
        if (self.rwdsToCheck['eefDist']):           
            ##################################
            #distance of end effector from constraint loc -  tolerance is specified in ANA ctor
            eefDistPen, eefDist = self.getRwd_expEffLocDist()
            #this reward component's contribution to reward
            reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='eefDist', rwdComp=-eefDistPen, rwdValWatched=[eefDist], rwdValWtchdStr='{:3f}'.format(eefDist), succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)

        if (self.rwdsToCheck['action']):           
            ###########################################
            #action min
            ##weighting reach elbow and both knees much less in contribution (to allow for higher magnitude torque on those joints) -weigthing knees and elbow has minimal effect with current weights - perhaps need to decrease weigthing values
            #actionPen, actSqMag = self.getRwd_expDofWtActionMin(optVal=self.a)
            actionPen, actSqMag = self.getRwd_expActionMin(optVal=self.a)
            #this reward component's contribution to reward
            reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='action', rwdComp=-actionPen, rwdValWatched=[actSqMag], rwdValWtchdStr='{:3f}'.format(actSqMag),succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)

        if (self.rwdsToCheck['height']) or (self.rwdsToCheck['UP_COMVEL']) or (self.rwdsToCheck['X_COMVEL']) or (self.rwdsToCheck['Z_COMVEL']):       
            #current com checked height or velocity
            #########################
            #com loc and com vel quantities : change self.comHtKey to whichever com body we wish to use to measure height
            bComHeight, bComVel, bComVelScore = self.getRwd_calcVelScores(com_now,self.com_b4,self.minCOMVelVals[self.comHtKey],self.maxCOMVelVals[self.comHtKey], self.dtForAllFrames)

            if (self.rwdsToCheck['height']) :
                #determine reward component and done condition
                heightAboveAvgFoot = bComHeight - avgFootLocAra[0][1] #self.startSeatedCOMHeight #center of COP on ground/feet        
                #reward tracked COM height above starting seated tracked COM height
 
                heightRew, htBelowStandHt = self.getRwd_expHtAboveInitCOM(heightAboveAvgFoot,startHeight=self.stSeatHtOvFtDict[self.comHtKey])
                #height_pen, htBelowStandHt = self.getRwd_expHtBelowStandCOM(heightAboveAvgFoot,.95*self.standHtOvFtDict[self.comHtKey])
                #if height_pen is less than 0 then rwd component is > 0 and it means we're taller than 98% of standHtovFt
                failCond = False # (heightAboveAvgFoot < .25*self.stSeatHtOvFtDict[self.comHtKey])
                reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='height', rwdComp=heightRew, rwdValWatched=[htBelowStandHt,heightAboveAvgFoot], rwdValWtchdStr='{:3f}'.format(htBelowStandHt),succCond=False, failCond=failCond, debug=debug, dct=dct, dbgStructs=dbgStructs)

            # bComVelScore : scores for idx 0 == x vel, idx 1 == y vel, idx 2 == z vel
            if (self.rwdsToCheck['UP_COMVEL']) :
                #failCond = (bComVel[1]<-.2)
                reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='UP_COMVEL', rwdComp=bComVelScore[1], rwdValWatched=[bComVel[1]], rwdValWtchdStr='{:3f}'.format(bComVel[1]),succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)                
            if (self.rwdsToCheck['X_COMVEL']) :
                reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='X_COMVEL', rwdComp=bComVelScore[0], rwdValWatched=[bComVel[0]], rwdValWtchdStr='{:3f}'.format(bComVel[0]),succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)                
            if (self.rwdsToCheck['Z_COMVEL']) :
                reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='Z_COMVEL', rwdComp=bComVelScore[2], rwdValWatched=[bComVel[2]],rwdValWtchdStr='{:3f}'.format(bComVel[2]), succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)
     
        if (self.rwdsToCheck['footMovDist']) or (self.rwdsToCheck['lFootMovDist']) or (self.rwdsToCheck['rFootMovDist']) :       
            #foot location compared to start
            footMovePens, footMovDist = self.getRwd_expFootMoveRew(avgFootLocAra)
            #idx 0 is both, idx 1 is left, idx 2 is right
            if (self.rwdsToCheck['footMovDist']) :
                reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='footMovDist', rwdComp=-footMovePens[0], rwdValWatched=[footMovDist[0]], rwdValWtchdStr='{:3f}'.format(footMovDist[0]),succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)
            if (self.rwdsToCheck['lFootMovDist']) :
                reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='lFootMovDist', rwdComp=-footMovePens[1], rwdValWatched=[footMovDist[1]], rwdValWtchdStr='{:3f}'.format(footMovDist[1]),succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)
            if (self.rwdsToCheck['rFootMovDist']) :
                reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='rFootMovDist', rwdComp=-footMovePens[2], rwdValWatched=[footMovDist[2]], rwdValWtchdStr='{:3f}'.format(footMovDist[2]),succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)
                
        if (self.rwdsToCheck['comcop']) or (self.rwdsToCheck['comcopvel']) or (self.rwdsToCheck['contacts']):
            #com-cop vector length(dist) and spd the vector is changing in magnitude
            contactDict, COPval, _ = self.calcAllContactData()
            if (self.rwdsToCheck['contacts']):
                #TODO calculate contact contributions to reward if being used
                print("No Contact Contributions being calculated!!")
                reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='contacts', rwdComp=0, rwdValWatched=[contactDict], rwdValWtchdStr='{}'.format(contactDict['footGroundContacts']),succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)

            if (self.rwdsToCheck['comcop']) or (self.rwdsToCheck['comcopvel']) :
                COMCOP_VelRwd, COMCOP_DistRew, COMCOPDist_speed  = self.getRwd_expCOMCOPDist(com_now, COPval)                
                if (self.rwdsToCheck['comcop']):
                    comcopDist = np.copy(self.COMCOPDist)
                    reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='comcop', rwdComp=COMCOP_DistRew, rwdValWatched=[comcopDist],rwdValWtchdStr='{:3f}'.format(float(comcopDist)), succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)                    
                if (self.rwdsToCheck['comcopvel']):
                    reward, isGoodDone, isBadDone = self.procRwdCmpRes(reward, isGoodDone, isBadDone, rwdType='comcopvel', rwdComp=COMCOP_VelRwd, rwdValWatched=[COMCOPDist_speed], rwdValWtchdStr='{:3f}'.format(COMCOPDist_speed),succCond=False, failCond=False, debug=debug, dct=dct, dbgStructs=dbgStructs)
        
        #if counting # of frames, check against done here
        done = isGoodDone or isBadDone or (numSteps > 200)
        if (isBadDone):
            reward = 0


        if (debug):
            print('Reward : {} Done : {}  step : {} '.format(reward, done, numSteps))
            print('{}'.format(dbgStructs['dbgStrList'][0]))
            

        #print ('reward : {}\tdone :{}'.format(reward, done))
        return reward, done, dct, dbgStructs


#     #simplest possible reward - COM height above avg foot location and minimize action
#     def calcMinActionMaxCOMHtAboveFtRwd(self, numItersDBG=100):
#         ##################################
#         #contact info
#         avgFootLocAra = self.calcAvgFootBodyLoc()
#         #########################
#         #com loc and com vel quantities : change self.comHtKey to whichever com body we wish to use to measure height
#         com_now = self.comHtBodyDict[self.comHtKey].com()
#         bComHeight, bComVel, bComVelScore = self.getRwd_calcVelScores(com_now,self.com_b4,self.minCOMVelVals[self.comHtKey],self.maxCOMVelVals[self.comHtKey], self.dtForAllFrames)
#         # bComVelScore : scores for idx 0 == x vel, idx 1 == y vel, idx 2 == z vel
#         heightAboveAvgFoot = bComHeight - avgFootLocAra[0][1] #self.startSeatedCOMHeight #center of COP on ground/feet        
#         #reward tracked COM height above starting seated tracked COM height
#         height_rew, htBelowStandHt = self.getRwd_expHtBelowStandCOM(heightAboveAvgFoot,self.standHtOvFtDict[self.comHtKey])
#         ###########################################
#         #action min
#         ##weighting reach elbow and both knees much less in contribution (to allow for higher magnitude torque on those joints) 
#         #weigthing knees and elbow has minimal effect with current weights - perhaps need to decrease weigthing values
#         #actionPen, actSqMag = self.getRwd_expDofWtActionMin(optVal=self.a)
#         actionPen, actSqMag = self.getRwd_expActionMin(optVal=self.a)
        
#         reward = height_rew - actionPen 
#         # if(self.env.steps % 1 == 0):
#         #     print('COM ht - action pen Reward @ step {} : {:.3f}\tActionRwd : -{:.5f}(action sq magnitude : {:.3f})\tno weight action rew :{:.7f}\tCOM ht Reward : {:.5f}(COM ht : {:.3f} |\tCOM ht over avg foot loc : {:.3f}|\tCOM ht below stand ht : {:.3f})'.format(self.env.steps,reward,actionPen,actSqMag,noWtActionRew,height_rew, bComHeight, heightAboveAvgFoot, htBelowStandHt))  
#         #never done from action, just keep going until 

#         done = (self.env.steps >= 100)
#         # success = (heightAboveAvgFoot > .95*self.standCOMHeightOvFt)
#         # failure = (self.env.steps >= 100)
#         # done = success or failure
#         # if done and failure : 
#         #     reward = 0
#         dct = {'broke_sim': False, 'raiseVelScore': 0, 'height_rew':0, 'actionReward':(-actionPen),'done':done}    
#         return reward, done, dct   

#     #this reward will be based only on minimum get up criteria, and may fail
#     #only call if not broke
#     #passed is com to use - either body or maybe head
#     def calcSimpleGetUpRew(self, numItersDBG=100):
#         ##################################
#         #distance of end effector from constraint loc -  tolerance is specified in ANA ctor
#         eefDistPen, eefDist = self.getRwd_expEffLocDist()
#         ##################################
#         #contact info
#         avgFootLocAra = self.calcAvgFootBodyLoc()
#         footMovePens, footMovDist = self.getRwd_expFootMoveRew(avgFootLocAra)
#         #use avg of both for now - idx 0 is both, idx 1 is left, idx 2 is right
#         #########################
#         #com loc and com vel quantities : change self.comHtKey to whichever com body we wish to use to measure height
#         com_now = self.comHtBodyDict[self.comHtKey].com()
#         bComHeight, bComVel, bComVelScore = self.getRwd_calcVelScores(com_now,self.com_b4,self.minCOMVelVals[self.comHtKey],self.maxCOMVelVals[self.comHtKey], self.dtForAllFrames)
#         # bComVelScore : scores for idx 0 == x vel, idx 1 == y vel, idx 2 == z vel
#         heightAboveAvgFoot = bComHeight - avgFootLocAra[0][1] #self.startSeatedCOMHeight #center of COP on ground/feet        
#         #reward tracked COM height above starting seated tracked COM height
#         height_rew, htBelowStandHt = self.getRwd_expHtBelowStandCOM(heightAboveAvgFoot,self.standHtOvFtDict[self.comHtKey])
#         ###########################################
#         #action min
#         ##weighting reach elbow and both knees much less in contribution (to allow for higher magnitude torque on those joints) 
#         #weigthing knees and elbow has minimal effect with current weights - perhaps need to decrease weigthing values
#         #actionPen, actSqMag = self.getRwd_expDofWtActionMin(optVal=self.a)
#         actionPen, actSqMag = self.getRwd_expActionMin(optVal=self.a)
        
#         reward = bComVelScore[1] + height_rew  - footMovePens[0] + actionPen - eefDistPen
        
#         if(self.env.steps % 1 == 0):
#             outStr = list()
#             outStr.append('Smpl TTL Rwd (comVelUpScore + height_rew  - footMovePens[0] + actionPen - eefDistPen) @ step {} : {}'.format(self.env.steps,reward))
#             outStr.append('Rwrds : H_rew :{:.3f}\tvel scrs : upVelScr :{:.3f}\tact : {:.3f}\tfoot move pen : -{:.3f}\tEEF dist pen : -{:.3f}'.format(height_rew, bComVelScore[1], actionPen, footMovePens[0], eefDistPen))
#             outStr.append('Sim Quantities : Assist Frc : {}\tCOM :{}\tCOMVel :{}\tFoot Avg Loc : {}\tfoot dist :{:.3f}'.format(self.desExtFrcVal,com_now, bComVel,avgFootLocAra[0],footMovDist[0]))
#             outStr.append('\t(action sq magnitude : {:.3f})\theightAboveAvgFoot : {:.3f}\t ht Below Stand Ht : {:.3f}\t eefDist from trgt :{:.3f}'.format(actSqMag,heightAboveAvgFoot, htBelowStandHt, eefDist))
#             print('-----')
#             for s in outStr:
#                 print(s)
        

#         done = (self.env.steps >= 200)

#         # #we have succeeded in getting up
#         # successCond = (heightAboveAvgFoot > .95*self.standCOMHeightOvFt)
#         # #following are failure conditions that we wish to prune (?)
#         # failCond = not(
#         #    heightAboveAvgFoot > -0.1         #don't go too low  
#         #    #and bComVelScore[1] > -0.1      #don't fall backwards
#         #    )
    
#         # done = successCond or failCond
#         # #set to 0 only if failed?
#         # if (done and failCond):
#         #     #print('Fail Cond Met : heightAboveAvgFoot : {} and raiseVelScore : {}'.format(heightAboveAvgFoot,bComVelScore[1]))
#         #     reward = 0
            
#         dct = {'broke_sim': False, 'raiseVelScore': bComVelScore[1], 
#            'height_rew':height_rew, 'actionReward':(-actionPen), 'done':done}
#         dbgDict = defaultdict(list)
#         return reward, done, dct, dbgDict




#     #calculate reward for this agent, see if it is done, and return informational dictionary (holding components of reward for example)
#     #called after fwd sim step
#     def calcRewardAndCheckDone_old(self, debug):
#         numSteps = self.env.steps

#         #attempt simplest reward function if not broken
#         if (self.env.useSimpleReward):
#             #return self.calcGAE_MinActionMaxHeadHtAboveFt(self.numStepsDBGDisp)
#             return self.calcMinActionMaxCOMHtAboveFtRwd(self.numStepsDBGDisp)
#             #return self.calcMinActionLeftHandHigh(self.numStepsDBGDisp)
#             #return self.calcSimpleGetUpRew(self.numStepsDBGDisp)
#         #print('Using Full reward function')
#         ###
#         #Scores are added, penalties are subtracted, from reward
#         ###
        
#         ##################################
#         #distance of end effector from constraint loc -  tolerance is specified in ANA ctor
#         eefDistPen, eefDist = self.getRwd_expEffLocDist()

#         ##################################
#         #contact components
#         #contactDict holds counts of contacts, 
#         #COPval holds either calculated center of pressure of hand and foot contacts or avg location of feet projected to the ground
#         #footAvgLoc holds average location of feet
#         print('1276 : calcRewardAndCheckDone : !!!!!!!!!!!!!!!!! This reward function has not been modified to match new exp-based reward components! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#         contactDict, COPval, avgFootLocAra = self.calcAllContactData()
#         #feet shouldn't move that much from starting position
#         #use avg of both for now - idx 0 is both, idx 1 is left, idx 2 is right
#         footMovePens, footMovDist = self.getRwd_expFootMoveRew(avgFootLocAra)    

#         #TODO if going to use this then need to determine proper scale
#         penCntctMult = 100  #contact penalties multiplier - use same value for all penalties
#         #kicking the assistant bot is bad
#         kickBotCntct = contactDict['kickBotContacts']
#         kickBotPen = penCntctMult*kickBotCntct
#         #stumbling on own feet is bad
#         #tripFeetCntacts = contactDict['selfContact']      
#         #tripFeetPen = tripFeetCntacts
#         #foot contact with ground is good - no feet on ground is very bad
#         footGroundCntcts = contactDict['footGroundContacts']
#         footCntctRwd = -penCntctMult if footGroundCntcts == 0 else (10* footGroundCntcts)
#         #hand ground contacts - allow off hand to push on ground
#         handGroundCntcts = contactDict['handGroundContacts']
#         handCntctRwd = handGroundCntcts
#         #non-foot, non-hand ground contacts -penalize these (butt on ground or other bodies on ground)
#         badGroundCntcts = contactDict['badGroundContact']
#         badCntctPen = penCntctMult*badGroundCntcts
        
#         ##################################
#         #distance and velocity scores
#         #body com - values are scalar height, vector velocity, vector score
#         #########################
#         #com loc and com vel quantities : change self.comHtKey to whichever com body we wish to use to measure height
#         com_now = self.comHtBodyDict[self.comHtKey].com()
#         bComHeight, bComVel, bComVelScore = self.getRwd_calcVelScores(com_now,self.com_b4,self.minCOMVelVals[self.comHtKey],self.maxCOMVelVals[self.comHtKey], self.dtForAllFrames)
#         # bComVelScore : scores for idx 0 == x vel, idx 1 == y vel, idx 2 == z vel
#         heightAboveAvgFoot = bComHeight - avgFootLocAra[0][1] #self.startSeatedCOMHeight #center of COP on ground/feet   
#         htAboveStCOMHeight = bComHeight - self.stSeatHtOvFtDict[self.comHtKey] 
#         #reward tracked COM height above starting seated tracked COM height
#         height_rew, htBelowStandHt = self.getRwd_expHtBelowStandCOM(heightAboveAvgFoot,self.standHtOvFtDict[self.comHtKey])

#         #reward foot average height staying close to ground - parabola
#         foot_ht_rew = 5.0 * self.calcRwdRootMethod(avgFootLocAra[0][1], 0, .09) 
#         #distance of COM projected over COP and Vel of COM projected over COP;COMCOP_speed is speed at which distance is shrinking - negative means growing
#         COMCOP_VelScore, COMCOP_DistRew, COMCOP_speed = self.calcCOMCOPRew(com_now, COPval, .1, 3.0)
#         #brokePen = 100 if broke_sim else 0        
#         #minimize requested torques/actions - this returns value that is larger ther smaller the torques/actions are
#         #actionRew, actSqMag = self.getRwd_actionMin(constVal=50,optVal=self.a, mult=1.0) 
#         #actionRew, actSqMag = self.getRwd_expActionMin(optVal=self.a,var=self.sqrtNumActDofs, wt=1.0, mult=1.0)
#         ##weighting reach elbow and both knees much less in contribution (to allow for higher magnitude torque on those joints) 
#         actionPen, actSqMag = self.getRwd_expDofWtActionMin(optVal=self.a)
        
#         ttlReward = height_rew + bComVelScore[1] + bComVelScore[0] + bComVelScore[2]  + COMCOP_VelScore + COMCOP_DistRew + footCntctRwd + foot_ht_rew + handCntctRwd
#         ttlPen = badCntctPen + kickBotPen + eefDistPen + actionPen #+ COMCOP_DistPen
        
#         reward = ttlReward - ttlPen 
#         #for debugging reward
#         if(numSteps % 1 == 0):
#              outStr = list()
#              outStr.append('Total Reward @ step {} : {}'.format(numSteps,reward))
#              L1 = 'Rwrds ({:.2f}): \t H_rew (linear) :{:.2f}\tvel scrs : y :{:.2f}\tx :{:.2f}\tz {:.2f}\tactionRew : {:.3f}(action sq magnitude : {:.3f})'.format(ttlReward,height_rew, bComVelScore[1], bComVelScore[0], bComVelScore[2] ,actionPen,actSqMag )
#              L2 = '\tfoot cnt Score :{:.2f}\tfoot ht reward :{:.2f}\thand cntct :{:.1f}\tCOM->COP vel Score : {:.3}\tCOM->COP dist Score : {:.3}'.format(footCntctRwd, foot_ht_rew,handCntctRwd, COMCOP_VelScore,COMCOP_DistRew)
#              outStr.append(L1 + L2)
#              outStr.append('Penalties  (-{:.2f}):badCntctPen -{:.1f}\tkickBotPen -{:.1f}\teefDistPen -{:.2f}'.format(ttlPen,badCntctPen, kickBotPen, eefDistPen))#, COMCOP_DistPen))
#              outStr.append('Sim Quantities : Assist Frc : {}\tCOM :{}\tCOMVel :{}\tCOP :{}\tFoot Avg Loc : {}'.format(self.desExtFrcVal,com_now, bComVel,COPval,avgFootLocAra[0]))
#              outStr.append('\t\tCOM-COP decrease vel:{:.3}\t heightAboveAvgFoot : {:.3f}\t height above stCOMHt : {:.3f}\tCOM-COP xz dist :{:.3}\teefDist : {:.3f}'.format(COMCOP_speed, heightAboveAvgFoot, htAboveStCOMHeight, self.COMCOPDist,eefDist))
#              print('-----')
#              for s in outStr:
#                  print(s)

#         #we have succeeded in getting up
#         successCond = (heightAboveAvgFoot > .95*self.standHtOvFtDict[self.comHtKey]) #and ()
#         #following are failure conditions that we wish to prune (?)
#         failCond = not(
#             htAboveStCOMHeight > -0.1         #don't go too low below seated position com height          
#             #and eefDistCost < 100 #make sure end effector is not further away than about 5 cm from trajectory target loc
#             #these velocity bounds might be too strict
#             and bComVelScore[1] > -.01 
#             #and xVelScore > 0
#             #and sideVelScore > 0
#             #and foot_ht_rew > -20 #keep foot avg loc close to ground 
#             )
    
#         done = successCond or failCond
# #                not(
# #                #these check failure : 
# #                #make sure we're going up - 0 means at starting seated com height - allow for some down motion
# #                (heightAboveAvgFoot > -0.1)
# #                #make sure end effector is not further away than about 5 cm from trajectory target loc
# #                #and eefDistCost < 100 
# #                #these velocity bounds might be too strict
# #                and raiseVelScore > 0 
# #                and foot_ht_rew > -20   #keep foot avg loc relatively close to the ground
# #                # and xVelScore > -.01  # allowing for slight negative velocities
# #                # and sideVelScore > -.01 # Not allowing more deviation
# #                )
#         #TODO clear reward only on failure?
#         if (done and failCond):
#             #print('Fail Cond Met : heightAboveAvgFoot : {} and raiseVelScore : {}'.format(heightAboveAvgFoot,bComVelScore[1]))
#             reward = 0
#         dct = {'broke_sim': False, 'raiseVelScore': bComVelScore[1], #'COMCOP_VelScore':COMCOP_VelScore,#'velScore': velScore,
#            'height_rew':height_rew, 'actionReward':actionRew, 'done':done}
#         #print ('reward : {}\tdone :{}'.format(reward, done))
#         return reward, done, dct    





    
#Ana using biped skeleton
class humanSkelHolder(ANASkelHolder):
    
    def __init__(self, env, skel, widx, stIdx, fTipOffset):  
        ANASkelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset) 

    #called for each skeleton type, to configure multipliers for skeleton-specific action space
    def _setupSkelSpecificActionSpace(self, action_scale):
        action_scale[[1,2,8,9]] *= .75
        #head 
        action_scale[[17,18]] *= .5
        #2 of each bicep
        action_scale[[20,22,26,28]]*= .5
        # shoulders
        action_scale[[19,25]]*= .75
        #scale ankles actions less
        action_scale[[4,5,11,12]]*= .5
        #scale feet and forearms, hands much less
        action_scale[[6,13, 23,24, 29,30]]*= .25
        return action_scale 
 
    
    #build the configuration of the initial pose of the figure
    def _makeInitPoseIndiv(self):
        initPose = [#sitting with knees bent - attempting to minimize initial collision with ground to minimize NAN actions
            #root orientation 0,1,2
            0,0,1.57,
            #root location 3,4,5
            0,0.06,0,            
            #left thigh 6,7,8 : 6 is bend toward chest, 7 is twist along thigh axis, 8 is spread
            0.87,0,0,
            #left shin, left heel(2), left toe 9,10,11,12
            -1.85,-0.6, 0, 0,
            #right thigh ,13,14,15: 13 is bend toward chest, 14 is twist along thigh axis, 15 is spread
            0.87,0,0,
            #right shin, right heel(2), right toe 16,17,18,19
            -1.85,-0.6,0, 0,
            #abdoment(2), spine 20,21,22
            0,-1.5,0,
            #head 23,24
            0,0,
            #scap left, bicep left(3) 25,26,27,28
            0,0.3,-0.6,0,
            #forearm left, hand left 29,30
            0.6,0,
            #scap right, bicep right (3) 31,32,33,34
            0,.6,1.8,0,
            #forearm right.,hand right 35,36
            0.5,0]
        #set reach hand elbow and both knees' action minimizing wt as close to 0 -> minimially penalize elbow and knee action
        #idx should be actual dof == dof idx - self.stTauIdx
        minActWtIDXs = np.array([3,10,27])
        minWtVals = np.ones(minActWtIDXs.shape)*.1
        self.buildActPenDofWtMat(minActWtIDXs, minWtVals)
        self.rootLocDofs = np.array([3,4,5])
        #set reach hand to be right hand - name of body node, and location of contact with constraint
        self.setReachHand('h_hand_right')
        return initPose

#ana using Kima Skeleton
class kimaHumanSkelHolder(ANASkelHolder):
    #isNewKima is temporary flag until issue with new kima is fixed (updating DART?)
    def __init__(self, env, skel, widx, stIdx, fTipOffset, isNewKima):  
        ANASkelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset)   
        self.isNewKima = isNewKima


    #called for each skeleton type, to configure multipliers for skeleton-specific action space
    def _setupSkelSpecificActionSpace(self, action_scale):
        
        if (self.isNewKima):
            #new kima - ode errors currently

            #these are dof idxs -6 (we never apply actions to first 6 dofs)
            #in action scale ara, it is idx - 6 to account for root dofs
            #left thigh 6,7,8 : 6 is spread, 7 is bend toward chest(positive is back), 8 is twist along thigh axis, 8 is spread
            #right thigh ,12,13,14: 12 is spread, 13 is bend toward chest(positive is back), 14 is twist along thigh axis, 8 is spread

            #spread both legs, twist both legs
            action_scale[[0,2, 6,8]] *=.5
            #thighs pull
            action_scale[[1, 7]] *= .7
            #left shin, left heel(2) 9,10,11 : 9 : bend at knee
            #right shin, right heel(2)  15,16,17 : 15 : bend at knee
            #knee : 
            action_scale[[3,9]]*=.7#.5
            #feet : 
            action_scale[[4,5, 10, 11]]*= .4
            #left and right shoulders/bicep, 2 for each bicep
            action_scale[[15,17, 19, 21]]*= .7#.5
            #scale forearms  less
            action_scale[[18, 22]]*= .3
        else :
            #old kima - different joint limits and configuration, less joint span in waist, used by Akanksha

            #these are dof idxs -6 (we never apply actions to first 6 dofs)
            #irrelevant dofs set to action scale 1/100th to minimize instablity caused by self-collision
            #thigh twist and spread 
            action_scale[[1, 2, 7, 8]] *= .7
            #2 of bicep, and hands
            action_scale[[15,17, 19, 21]]*= .75#.5
            #right shoulders, 2 of bicep, hands
            #action_scale[[25,26]]*= .01
            #shoulders/biceps
            #action_scale[[19,20,21,22,25,26,27,28]] *= .75
            #scale ankles actions less
            action_scale[[4,5, 10, 11]]*= .5
            #scale forearms much less
            action_scale[[18, 22]]*= .75#.5

        return action_scale       

    #build the configuration of the initial pose of the figure
    def _makeInitPoseIndiv(self):
        #pose from sim with feet and butt on ground
        if (self.isNewKima):
            #new kima - ode errors currently - might be fixed by updating to dart 6.4 or higher
            initPose = [
                0,  0.09133,  0,  
                #root orientation 3,4,5 - TODO check this
                0,  0,  0,  
                #left thigh 6,7,8 : 6 is spread, 7 is bend toward chest(positive is back), 8 is twist along thigh axis, 8 is spread
                0.01324, -2.3, 0.03374,   
                #left shin, left heel(2) 9,10,11 : 9 : bend at knee
                1.66,-.7,0,
                #right thigh ,12,13,14: 12 is spread, 13 is bend toward chest(positive is back), 14 is twist along thigh axis, 8 is spread
                0.01324, -2.3, -0.03374, 
                #right shin, right heel(2)  15,16,17 : 15 : bend at knee
                1.66,-.7,0,
                #abdoment(2), spine 18,19, 20 : 18:side to side, 19:front to back : 20 : twist
                0.0,-0.2,0.0, 
                #bicep left(3) 21,22,23                
                -.1,-.5,0,
                #forearm left,24
                .5,  
                # bicep right (3) 25, 26, 27 : 25: outward from sides, 26 : front/back (negative forward)
                0.1,-1.4,0,  
                #forearm right. 28
                1.5
            ]
        else :
            #old kima - different joint limits and configuration, less joint span in waist, used by Akanksha
            initPose = [
                    #root location 0,1,2 :[ 0.06891,  0.27575,  0.01631]
                    0,  0.09133,  0,  
                    #root orientation 3,4,5 Z Y X
                    # Z is clocksise dir on the paper top to bottom
                    #Y is clockwise direction coming out of paper. Guy looks in direction of arrow
                    1.53011,  0,  0,  
                    #left thigh 6,7,8 : 6 is bend toward chest, 7 is twist along thigh axis, 8 is spread
                    0.90392,  0.03374,  0.01324,  
                    #left shin, left heel(2) 9,10,11
                    1.92636, -0.58696,  0.03805,  
                    #right thigh ,12,13,14: 12 is bend toward chest, 13 is twist along thigh axis, 14 is spread
                    0.82051,  0.00663,  0.00911,  
                    #right shin, right heel(2)  15,16,17
                    1.91208, -0.43955,  0.003,
                    #abdoment(2), spine 18,19, 20
                    0.05288, -1.49464,  0.05056,  
                    #bicep left(3) 21,22,23                
                    0.24801, -0.49269, -0.01829,  
                    #forearm left,24
                    0.59213,  
                    # bicep right (3) 25, 26, 27
                    0.15881, -1.56151, 0.43283,  
                    #forearm right. 28
                    0.72403
                ]
        #set reach hand elbow and both knees' action minimizing wt as close to 0 -> minimially penalize elbow and knee action
        #idx should be actual dof == dof idx - self.stTauIdx
        minActWtIDXs = np.array([3,9,22])
        minWtVals = np.ones(minActWtIDXs.shape)*.1
        self.buildActPenDofWtMat(minActWtIDXs, minWtVals)
        
        self.rootLocDofs = np.array([0,1,2])
        #set reach hand to be right hand - name of body node
        self.setReachHand('r-lowerarm')
        return initPose

#abstract base class for helper robot (full body or fixed arm)
class helperBotSkelHolder(skelHolder, ABC):
                 
    def __init__(self, env, skel, widx, stIdx, fTipOffset):          
        skelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset)
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
        
        #max iterations of IK - keep low, doesn't use that many anyway
        self.IKMaxIters = 3
        
        ########################
        ##  runtime flags
        #do not use randomized initial state - use init state that has been evolved to be in contact with constraint
        self.randomizeInitState = False
        #Monitor torques seen - managed in parent class
        #self.monitorTorques = True
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
        #whether or not to debug IK calc
        self.debug_IK = False
        
        #the force being used
        numDim = 3 if (self.useLinJacob) else 6
        self.useForce = np.zeros(numDim)

        #sets dimension of optimization vector : function lives in child class
        self.nOptDims = self.getNumOptDims()
      
    #helper bot would only use this if it was solving its action using RL
    def setInitRWDValues(self):
        pass
        
    #need to apply tau every step  of sim since dart clears all forces afterward - this is any class-specific actions necessary for every single sim step/frame
    def applyTau_priv(self):
        pass    
    
    #helper bot will IK to appropriate world position here, and reset init pose    
    def _setToInitPose_Priv(self):
        #IK's to initial eff position in world, to set up initial pose
        if(self.initEffPosInWorld is not None):
            self.skel.set_positions(self.initPose)
            #IK eff to constraint location ( only done initially to set up robot!)
            self.IKtoPassedPos(self.initEffPosInWorld)
            self.initPose = np.copy(self.skel.q)        
    
    #called after initial pose is set 
    def postPoseInit(self):
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
            self.getSkelqDot(),
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
        dofOnes = np.ones(self.ndofs)
        self.oneAOvTS = dofOnes/self.timestep
        
        #negative because subttracted Tau in MA equality constraint
        self.tauDot = dofOnes * -1
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
            #self._reset_minMaxGuessVals()
            
#        if(self.monitorGenForce):
#            self.minMaxFrcDict = {}
#            self.totGenFrc = list()
#            self._resetMinMaxMonitors(self.minMaxFrcDict, self.useForce.shape)
#            #self._reset_minMaxGenForceVals()
            
        #create optimizer - uses default LN_COBYLA unless alg is passed as arg
        self.initOptimizer()           
        
        #for operational space controller : gains values from dart example
        Kp = 5.0
        Kd = 0.01
        self.osc_Kp = np.diag(np.ones(6)*Kp)
        self.osc_Kd = np.diag(np.ones(self.ndofs)*Kd)
        
    # def _reset_minMaxGuessVals(self):
    #     #for monitoring min and max guess values
    #     self.minGuess = np.ones(self.nOptDims)*1000000000
    #     self.maxGuess = np.ones(self.nOptDims)*-1000000000
        
        
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
        self.trackBodyVel = (self.trackBodyCurPos - self.trackBodyLastPos)/self.timestep
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


#    #return body torques to provide self.desExtFrcVal at toWorld(self.constraintLoc)
#    #provides JtransFpull component of equation
#    def getPullTau(self, useLinJacob, debug=False):     
#        if (useLinJacob) :             
#            self.useForce = self.desExtFrcVal
#            #using only linear : 3 rows x ndofs cols
#            Jpull = self.reachBody.linear_jacobian(offset=self.reachBodyOffset)            
#        else : 
#            #wrench
#            #TODO verify target orientation should be 0,0,0
#            self.useForce = np.zeros(6)
#            self.useForce[3:]=self.desExtFrcVal
#             #using linear and rotational == world
#            Jpull = self.reachBody.world_jacobian(offset=self.reachBodyOffset) 
#        if(debug):
#            print('getPullTau : pull force being used : {} '.format(self.useForce))
#
#        JTrans = np.transpose(Jpull)
#        res = JTrans.dot(self.useForce)
#        JTransInv = np.linalg.pinv(JTrans)
#        #last 3 rows as lin component
#        return res, JTransInv, Jpull, Jpull[-3:,:]
            
#        JtransInvC = self.calcPInv(Jlin_trans, 3)
#        if np.allclose(JtransInv,JtransInvC,1e-9):
#            print('numpy pinv calc on jtransinv yield equivalent result to manual calc')
#        else :
#            print('numpy pinv calc on jtrasninv different from manual calc : \n{}\n and {}'.format(JtransInv,JtransInvC))  
#        if np.allclose(res,resWd,1e-9) :
#            print('using world and lin jacobians yield equivalent result')
#        else :
#            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!using lin and world jacobians not equivalent in calculating JtransFrc_pull : \n{}\n and \n{}\n'.format(res,resWd))
        #Jw_pinv = JwTrans.dot(np.linalg.inv(Jw.dot(JwTrans) + tmpIdent6))        
        #print('\nNumpy pinv calc on jtrasninv and on world jwtransinv : \n{}\n and {}'.format(JtransInv,JwTransInv))
        
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
        vecToCnstrnt = self.cnstrntBody.to_world(x=self.cnstrntOnBallLoc) - self.reachBody.to_world(x=self.reachBodyOffset)
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
        #per timestep constraint and eef locations and various essential jacobians
        self.setPerStepStateVals(True)
        #Mass Matrix == self.skel.M
        #M is dof x dof 
        self.M = self.skel.M
        #M * qd / dt - precalc for MAconst
        #M / dt = precalc for MAconst - multiply by qdotPrime
        self.M_ovDt =self.M/self.timestep
        #derivative of MA eq w/respect to qdotprime
        self.M_dqdotPrime = self.M.dot(self.oneAOvTS)
        #precalc -Mqdot/dt
        self.Mqd = -self.M_ovDt.dot(self.skel.dq)

        #CfG == C(q,dq) + G; it is dof x 1 vector
        self.CfG = self.skel.coriolis_and_gravity_forces() 
        #TODO constraint forces - investigate
        self.CntrntFrc = self.skel.constraint_forces()        

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
    
    #dt set to timestep * frameskips
    #functionality before sim step is executed on this skel
    #here is where we calculate the robot's control torques
    def preStep(self, actions):   
        #if not mobile skeleton, perform IK on position
        if (not self.isFrwrdSim):
            if (self.debug):
                print('helperBotSkelHolder::preStep : {} robot set to not mobile, so no optimization being executed, but IK to constraint position performed'.format(self.skel.name))
            self.setPerStepStateVals(False)
            self.IKtoCnstrntLoc()
            #self.tau = np.zeros(self.ndofs)
            return
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
            #mma only handles inequality constraints
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
                
            #restore environment for actual frwrd step
            #self.env.restoreSimState()            
            self.doStepTestsAndSetTau_OptCntrl()
        
        
    #IK end effector to constraint position       
    def delSqDist(self, p1, p2):
        diff = p1 - p2
        return diff, diff.dot(diff) 
    
    def _IK_setSkelAndCompare(self, q, pos):
        self.skel.set_positions(q)
        effWorldPos = self.reachBody.to_world(x=self.reachBodyOffset)
        diff = pos - effWorldPos
        return diff, diff.dot(diff), effWorldPos 
    
    #IK eef to world location of constraint
    def IKtoCnstrntLoc(self):
        self.IKtoPassedPos(self.trackBodyCurPos)
    #IK end effector to passed position (world coords)
    def IKtoPassedPos(self, pos):
        #minimize .5 *(pos - effPos_w)^2 
        #== .5 * (pos - T.effLcl)^2
        #
        if(self.debug_IK):
            print('\nIK to pos : {}'.format(pos))
        delPt, distSqP, effWorldPos= self._IK_setSkelAndCompare(self.skel.q, pos)
#        effWorldPos = self.reachBody.to_world(x=self.reachBodyOffset)
#        delPt, distP = self.delSqDist(pos,effWorldPos)
        iters = 0
        oldQ = self.skel.q
        oldDistSq = distSqP
        iters = 0
        while ((distSqP > 0.000001) and (iters < self.IKMaxIters)):
            JPullPInv = np.linalg.pinv(self.reachBody.linear_jacobian(offset=self.reachBodyOffset))
            delQ = JPullPInv.dot(delPt)
            alpha = 5.0
            #reuse same jacobian
            while True :#apply current jacobian repeatedly until doesn't improve
                newQ = self.skel.q + alpha*delQ
                delPt, distSqP, effWorldPos = self._IK_setSkelAndCompare(newQ, pos)
                #print('iters : {} | alpha : {} | distP (sqDist) : {} | old dist : {}'.format(iters, alpha, distP,oldDist))
                if (distSqP > oldDistSq):#got worse, try again with alpha *= .5
                    #return to previous state
                    delPt, oldDistSq, effWorldPos = self._IK_setSkelAndCompare(oldQ, pos)                        
                    if(alpha > .00001): #alpha not too small
                        alpha *= .5                    
                    else :#alpha too small and getting too
                        distSqP = oldDistSq
                        break
                        
                else :#got better, continue
                    oldDistSq = distSqP
                    oldQ = newQ
            #input()
            
            self.trackBodyCurPos = pos
            if(self.debug_IK):
                print('iter:{} delPt : {} | delQ {} |  eff world pos {}'.format(iters,delPt, delQ, effWorldPos))
            iters +=1
        if(self.debug_IK):
            print('IK Done : final world position {}\n'.format(effWorldPos))

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
        poseAccel[(self.optPoseUseIDXs)] = (qdotPrime[(self.optPoseUseIDXs)] - self.curMatchPoseDot)/self.timestep
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
        newLoc = curLoc + self.timestep * curWrldVel
        #trackBody needs to be set for this objective
        locDiff = newLoc - self.trackBodyCurPos
        locPart = locWt * (.5 * (locDiff.dot(locDiff)))
        #gradient of locPrt = locDiff * d(locDiff) 
        #d(locDiff) = timestep * self.JpullLin
        locGrad = locWt * locDiff.dot(self.timestep*self.JpullLin)
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
        else:
            print('MAcnstVec satisfied by result')
        if(debug):
            for x in result:
                print('\t{}'.format(x))  
        return passed
            
    #build force dictionary to be used to verify pull force - this is called after sim step
    def dbgBuildFrcDict(self):
        ma = self.skel.M.dot(self.skel.ddq )    
        cg = self.skel.coriolis_and_gravity_forces() 
        cnstrntFrc = self.skel.constraint_forces()
        #torque cntrol desired to provide pulling force at contact location on reaching hand    
        Tau_JtFpull_new, JtPullPInv_new, Jpull_new, JpullLin_new = self.getPullTau(self.useLinJacob,debug=True)
        
        self.frcD = {}
        self.frcD['tau']=np.copy(self.tau)
        self.frcD['ma']=ma
        self.frcD['cg']=cg
        self.frcD['cnstf']=cnstrntFrc
        #jacobian pemrose inverse to pull contact point
        self.frcD['JtPullPInv_new'] = JtPullPInv_new
        
        self.frcD['jtDotCGrav'] = JtPullPInv_new.dot(cg)
        self.frcD['jtDotCnstFrc'] = JtPullPInv_new.dot(cnstrntFrc)
        self.frcD['jtDotMA'] = JtPullPInv_new.dot(ma)
        self.frcD['jtDotTau'] = JtPullPInv_new.dot(self.frcD['tau'])
                
        
        #total pull force of listed forces (doesn't include contact force calc if present)
        self.frcD['totPullFrc'] = self.frcD['jtDotTau'] - self.frcD['jtDotCGrav'] - self.frcD['jtDotMA']            
        self.frcD['totPullFrcCnst'] = self.frcD['totPullFrc'] + self.frcD['jtDotCnstFrc']        
        
        if(self.monitorGenForce):
            self._checkiMinMaxVals(self.frcD['totPullFrcCnst'], self.minMaxFrcDict)
            #self._checkiMinMaxVals(self.frcD['totPullFrcCnst'], self.minMaxFrcDict, self.maxGenFrc)
            self.totGenFrc.append(self.frcD['totPullFrcCnst'])
        
        #self.frcD['totPullFrcOld'] = self.frcD['jtDotTau'] - self.frcD['jtDotCGrvOld'] #- self.frcD['jtDotMAOld']      
        
    def dbg_dispMinMaxGuesses(self):
        self.dbgDispGuess_priv(self.minMaxGuessDict['min'], 'Min Guess Vals Seen ')
        print('\n')
        self.dbgDispGuess_priv(self.minMaxGuessDict['max'], 'Max Guess Vals Seen ')
        print('\n')
    
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
        
    
    #display instance classes partition of guess values for debugging
    @abstractmethod
    def dbgDispGuess_priv(self, guess, name=' '):
        pass
    #test results, display calculated vs simulated force results
    @abstractmethod
    def testFrcRes(self):      
        pass     
    
    
    #perform post-step calculations for robot - no reward for Inv Dyn
    def calcRewardAndCheckDone(self, debug):        
        if (not self.isFrwrdSim):
            if (self.debug):
                print('helperBotSkelHolder::calcRewardAndCheckDone : No Optimization since {} set to not mobile'.format(self.skel.name))
        else:
            self.testFrcRes()
        done=False
        rwd=0
        #preserve this format even though rllab uses different format for result dictionary
        dct = {'broke_sim': False, 'raiseVelScore': 0, 'height_rew':0,
                    'actionPenalty':0, 'is_done': done}   
        dbgDict = defaultdict(list)
        return rwd, done, dct, dbgDict           

#class to hold assisting robot
class robotSkelHolder(helperBotSkelHolder):
                 
    def __init__(self, env, skel, widx, stIdx, fTipOffset):          
        helperBotSkelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset)
        self.name = 'Humanoid Helper Bot'
 
        self.numOptIters = 1000
        #const bound magnitude
        self.bndCnst = 200
        #robot biped uses contacts in optimization process
        self.numCntctDims = 12
        #robot optimization attempts to match pose
        self.doMatchPose = True
        #self.nOptDims = self.getNumOptDims()

    #called for each skeleton type, to configure multipliers for skeleton-specific action space
    def _setupSkelSpecificActionSpace(self, action_scale):
        action_scale[[1,2,8,9]] *= .75
        #head 
        action_scale[[17,18]] *= .5
        #2 of each bicep
        action_scale[[20,22,26,28]]*= .5
        # shoulders
        action_scale[[19,25]]*= .75
        #scale ankles actions less
        action_scale[[4,5,11,12]]*= .5
        #scale feet and forearms, hands much less
        action_scale[[6,13, 23,24, 29,30]]*= .25
        return action_scale 
        
    #build the configuration of the initial pose of the figure
    def _makeInitPoseIndiv(self):
        initPose = np.zeros(self.skel.ndofs)
        #move to initial body position
        initPose[1] = 3.14
        initPose[3] = 0.98
        initPose[4] = .85
        #bend at waist
        initPose[21] = -.4
        #stretch out left arm at shoulder to align with hand
        initPose[26] = .25
        #stretch out left hand
        initPose[27] = -1.2        
        #set reach hand to be left hand - name of body node
        self.setReachHand('h_hand_left')
        return initPose
        
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
                    grad[x,gradColSt] = (fcntct[stIdx] - oldFcntct[stIdx])/self.timestep
                    grad[x,(gradColSt+2)] = (fcntct[stIdx+2] - oldFcntct[stIdx+2])/self.timestep
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
    
    #display results of fwd simulation
    def testFrcRes(self):        
        #build fundamental force dictionary, in self.frcD
        self.dbgBuildFrcDict()
        #pull force should be == to jtpullpinv.dot(-MA - CG - JtcntctTrans.(cntctFrc) + Tau)
        #this is only internal force in opposite direction, need all contributing factors to get force in opposite direction
        #i.e. get all external forces, subtract gravity contribution of skeleton
        #calc body torques from all contacts -> Jtrans@cntctpt * fcntct
        print('Test results after {} simulation steps :'.format(self.numSimSteps))
        cntctBdyTrqTtl = np.zeros(self.ndofs)
        contactRes = self.getMyContactInfo()
        self._cntctFrcTtl = np.zeros(self.useForce.shape)
        #find total body torque contribution from contacts
        for k,v in contactRes.items():
            self._cntctFrcTtl[3:] += v.ttlfrc
            cntctBdyTrqTtl += v.calcTtlBdyTrqs()
            
        #contact-related
        self.frcD['cntctBdTrq']=np.copy(cntctBdyTrqTtl)
        self.frcD['jtDotCntct'] = self.frcD['JtPullPInv_new'].dot(cntctBdyTrqTtl)
        
        totPullFrcNoCntct = self.frcD['totPullFrc']
        totPullFrc = totPullFrcNoCntct + self.frcD['jtDotCntct']
        totPullFrcCntct = totPullFrcNoCntct - self._cntctFrcTtl
        
        sumTtl = np.zeros([3])
        for k,v in self.frcD.items():
            if 'jt' in k:
                print('\t{} : {}'.format(k,v))
                sumTtl += v
        
        print('\tContact Force : {}'.format(self._cntctFrcTtl))
        print('\tDesired Force : {}'.format(self.useForce))

        frcDiffNow = self.frcD['totPullFrc'] - self.useForce
        #frcDiffOld =self.frcD['totPullFrcOld']- self.useForce    
        
        #print('\nDifference between proposed force and actual : {} \n\tdesFrc : {} \n\ttotPullFrcNoCntct:{}\n\ttotPullFrc  : {}\n\tcntctFrc : {}\n\ttotPullFrcCntct : {}'.format(frcDiff,self.useForce,totPullFrcNoCntct,totPullFrc, self._cntctFrcTtl,totPullFrcCntct))
        print('\nDifference (new) between proposed force and actual : {}'.format(frcDiffNow))
        #print('\nDifference (old) between proposed force and actual : {}'.format(frcDiffOld))
        print('\tdesFrc : {}'.format(self.useForce))
        print('\ttotPullFrcNoCntct:{}\n\ttotPullFrc  : {}\n\tcntctFrc : {}\n\ttotPullFrcCntct : {}\n\tSum of all components :{}'.format(totPullFrcNoCntct,totPullFrc, self._cntctFrcTtl,totPullFrcCntct, sumTtl))

    #display this classes guess configuration
    def dbgDispGuess_priv(self, guess, name=' '):
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
                 
    def __init__(self, env, skel, widx, stIdx, fTipOffset):          
        helperBotSkelHolder.__init__(self,env, skel,widx,stIdx, fTipOffset)
        self.name = 'KR5 Helper Arm'
        #set to true to use box constraints
        self.numOptIters = 100       #bounds magnitude
        self.useBoxConstraints = False
        #bound for torques
        self.bndCnst = 500.0
        #limit for qdot proposals only if box constraints are set to true
        self.qdotLim = 5.0   
        self.doMatchPose = True
        #self.nOptDims = self.getNumOptDims()

    #called for each skeleton type, to configure multipliers for skeleton-specific action space
    def _setupSkelSpecificActionSpace(self, action_scale):
        #no action scales for this robot
        return action_scale 
        
    #build the configuration of the initial pose of the figure
    def _makeInitPoseIndiv(self):
        
        #IK so that end effector position == tracking 
        #move to initial body positions
#        initPose[1]=.15
#        initPose[2]=.25
        #print('bot arm :\n{}'.format(self.skel.q))
        #possibly not necessary but set root's connection to box
        #set reach hand  - arm only has 1 hand
        self.setReachHand('palm')
        #IK constraint hand to location in world - dist of .1 from tip of human finger
        #self.IKtoPassedPos(self.helpedSkelH.reachBody.to_world(x=np.array([0,-.2319,0])))
        initPose = np.copy(self.skel.q) 
        #actual init pose set by IK to target constraint/grab location        
        return initPose
    
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
#        print('bot head height : {} '.format(self.skel.body(self.headBodyName).com()[1] ))
#        print('bot foot height : {} '.format(self.skel.body('h_heel_left').com()[1] ))
#        print('bot com  : {} '.format(self.skel.com() ))
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
        velWt = 0.0001
        locWt = 100.0
        
        #matching location is assisted by matching velocity
        #but this causes huge torques and weird force changes
        pVelWt = 0.0001
        #frcWt = 0.0
        #match pose, if performed
        #posePart, poseGrad = self.pose_objFunc(poseWt, qdotPrime)

        #minimize velocity
        velPart, velGrad = self.genericObj(velWt, qdotPrime, self.curQdot ,self.qdotWts)        
        #minimize distance of end effector from constraint body
        locPart, locGrad = self.effPos_objFunc(locWt, qdotPrime)
        
        #minimize difference in ball and end effector velocities
        pVelPart, pVelGrad = self.effVel_objFunc(pVelWt, qdotPrime)        

        #frcPart = 0
#        if(frcWt > 0):
#            frcPart, frcGrad = self.force_objFunc(frcWt, tau)   
        
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
    
    #NEED TO VERIFY THESE VALUES
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
        maRes = self.M_ovDt.dot(qdotPrime) + self.Mqd + self.CfG - self.CntrntFrc
        
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

    #display results of fwd simulation
    def testFrcRes(self):      

        #build fundamental force dictionary, in self.frcD
        self.dbgBuildFrcDict()
        print('\nTest results after {} simulation steps :'.format(self.numSimSteps))
        sumTtl = np.zeros(self.useForce.shape)
        if self.debug: 
            for k,v in self.frcD.items():
                if ('jt' in k) :#or ('tot' in k):
                    print('\t{} : \t{}'.format(k,v))
                    sumTtl += v
        print('\t\tForce - CfG - ma : \t{}'.format(self.frcD['totPullFrc']))
        print('\t\tAbove W/Cnst Frc : \t{}'.format(self.frcD['totPullFrcCnst']))
        #print('\n\tPull Force : \t{}'.format(self.frcD['jtDotTau']))
        print('\t\tTarget Force : \t\t{}\n'.format(self.useForce))
        #print('\tDifference : \t{}\n'.format(frcDiffNew))        
         
        #frcDiffOld =self.frcD['totPullFrcOld'] - self.useForce
        #print('Difference between proposed force and actual : {} \n\tdesFrc : {} \n\tjtDotTau  : {}\n\tcntctFrc : {} \n\tskelGrav : {}'.format(frcDiff,self.useForce,jtDotTau, cntctFrcTtl,skelGrav))
        #print('\nDifference between proposed force and actual : \n{} \n\tdesFrc : \t{}\n\ttotPullFrc  : \t{}\n\tSum of all components :\t{}'.format(frcDiff,self.wrenchFrc, totPullFrc, sumTtl))
        #print('\nDifference between proposed force and actual : \n\t\t\t{} \n\tdesFrc : \t{}\n\ttotPullFrc  : \t{}\n\n'.format(frcDiffNew,self.useForce, self.frcD['totPullFrc']))
        
    #display this classes guess configuration
    def dbgDispGuess_priv(self, guess, name=' '):
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
