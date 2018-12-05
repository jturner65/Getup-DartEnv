#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

import numpy as np
#import pydart2 as pydart
from gym import utils
from gym.envs.dart import assist2bot_env
from os import path

from collections import defaultdict

#environment where robot/robot arm helps RL policy-driven human get up using force propossal from value function approximation provided by human policy baseline
#uses constraint delta instead of assist force for observation and robot optimization
#class DartStandUp3dAssistEnvCnstrnt(dart_env_2bot.DartEnv2Bot, utils.EzPickle):
class DartStandUp3dAssistEnvCnstrnt(assist2bot_env.DartAssist2Bot_Env, utils.EzPickle):
    def __init__(self): #,args): #set args in registration listing in __init__.py
        """
        This class will manage external/interacting desired force!! (not DartEnv2Bot)
        put all relevant functionality in here, to keep it easily externally accessible
        """
        #move robot arm out of the way (behind ANA)
        moveArm = False

        if moveArm : 
            kr5Loc = path.join('KR5','KR5 sixx R650_moveArm.urdf')      #for kr5 arm as assistant  
            modelLocs = [kr5Loc,'kima/getUpWithHelperBot3D_armKima_old_moveArm.skel']      
        else :
            kr5Loc = path.join('KR5','KR5 sixx R650.urdf')      #for kr5 arm as assistant  
            modelLocs = [kr5Loc,'kima/getUpWithHelperBot3D_armKima_old.skel']     
        
        kr5HelperBotLoc = path.join('KR5','KR5 sixx R650_helper.urdf') 
        ########################
        ## loading world/skels
        #modelLocs = ['getUpWithHelperBot3D_damp.skel']     #for two biped skels    
        #kr5Loc = path.join('KR5','KR5 sixx R650.urdf')      #for kr5 arm as assistant  
        #for debugging - moves arm out of way
        #kr5Loc = path.join('KR5','KR5 sixx R650_moveArm.urdf')      #for kr5 arm as assistant  

        #modelLocs = [kr5Loc,'getUpWithHelperBot3D_arm.skel']   #regular biped as ana
        #kima biped below with new kima skel - experienced ode issues which killed rllab, might be fixable with updating dart to 6.4.0+ using wenhao's modified lcp.cpp seems to address the issue.
        #modelLocs =  [kr5Loc,'kima/getUpWithHelperBot3D_armKima.skel']
        #kima biped below with kima skel from 2/18 - different joint limits, mapping of dofs and euler angle layout of root - MUST KEEP OLD in file name - used for both new experiments and ak's policy with constraint
        #modelLocs = [kr5Loc,'kima/getUpWithHelperBot3D_armKima_old.skel']     
        #for debugging - moves arm Platform out of the way
        #modelLocs = [kr5Loc,'kima/getUpWithHelperBot3D_armKima_old_moveArm.skel']      

        #set pose to not be prone - needs to be set before call to parent ctor - do this for testing
        self.setInitPoseAsGoalState = False
        fs = 1.0
        #multiple of frame skip
        ts = .01/fs
        assist2bot_env.DartAssist2Bot_Env.__init__(self, modelLocs, fs, dt=ts, helperBotLoc=kr5HelperBotLoc, disableViewer=False)
        
        #load eef traj final location fleishman distribution object based on sampled locations of standing ana moving reach hand
        _ = self.sampleGoalEefRelLoc()        
        
        #connect human to constraint 
        self.connectHuman = True
        #connect bot to constraint - if bot is connected then constraint should be passive, but dy
        self.connectBot = True

        #initialize all force and trajectory values: extAssistSize=6 will add force application location target to observation
        #NOTE :if bot is connected, trajectory _MUST_ be passive or explodes
        forcePassive = False
        trajTyp = 'gauss'#'servo'       #servo joint cannot be moved by external force - can train with servo, but to use bot must use freejoint
        self.initAssistTrajVals(extAssistSize=3, useANAHeightTraj=False,  trajTyp=trajTyp, setTrajDynamic=False, setTrajPassive=(self.connectBot or forcePassive))
        #self.initAssistTrajVals(extAssistSize=3, useANAHeightTraj=False,  trajTyp='gauss', setTrajDynamic=True, setTrajPassive=False)
        
        #whether or not to stop when trajectory is finished
        self.stopWhenTrajDone = False
        #display debug information regarding force application - turn off if training
        self.dbgAssistData = False             
        #display ANA reward dbg data - slow, make false if training
        self.dbgANAReward = True
        #calc and display post-step ANA eef force - make false if training
        self.dbgANAEefFrc = False
        #display full ANA force results if dbgANAEefFrc is true
        self.dbgANAFrcDet = False
        ############################
        # set human/robot conection and motion trajectory object and params  - if train==True then overrides dynamicbot to be false (OPT to solve for location is expensive)
        #self.trainPolicy is defined as static variable in dart_env_2bot TODO make parent class method that enables this to be trained -after- object is instanced
        #_solvingBot : whether or not helper bot's motion is solved
        #_botSolving is type of solving Bot should engage in : 0 is IK, 1 is constraint optimization dyn, 2 is IK-SPD 
        #removed : #_helpingBot : whether or not helper bot is actually coupled to human(i.e. helping by directly applying force)  
        #build gaussian trajectory
        SPDGain = 1000   #only used for _botSolving==2,_solvingBot==True
        self.setTrainAndInitBotState(self.trainPolicy, _solvingBot=True, _botSolving=2, _SPDGain=SPDGain)   
        #whether this environment uses force as an assistive component or not - default is no
        self.assistIsFrcBased = False

        #add mimic bot used to IK
        if (self.solveBotIK_SPD):
            self.initMimicBot()

        utils.EzPickle.__init__(self)
        
    #set the alt bot body node clrs to be different
    def setRGBAofAltBot(self, skel):
        #from dart_env_2bot.py : list of tuples of skels to render (idx0) with color list (idx1)
        #self.skelsToRndrWClr= []
        skelTup = (skel,[1.0,0.0,0.0,0.5])
        self.skelsToRndrWClr.append(skelTup)
    

    #set assist objects to be able to collide
    def setAssistObjsCollidable(self):
        self._setSkelCollidable(self.boxSkel,True)
        self._setSkelCollidable(self.skelHldrs[self.botIdx].skel, True)

    #individual environment-related components of assist initialization
    def initAssistTraj_indiv(self): 
        #this env uses displacements.  Here we should initialize the bounds of those displacements
        #displacement bounds for this trajectory
        self.dispBnds = self.trackTraj.getMinMaxBnds()#np.array([[0.0,0.0,-0.001],[0.2, 0.5, 0.001]]) 
        print("initAssistTraj_indiv : disp bnds : {} ".format(self.dispBnds))
        
    #set up mimic bot - duplicate of helper bot to be used to IK to positions without breaking constraints
    def initMimicBot(self):
        if (self.helperBotFullPath is None): return
        self.dart_world.add_skeleton(self.helperBotFullPath)
        #mimic skel is last skel added
        mimicSkel = self.dart_world.skeletons[-1]
        mimicSkel.setName("mimicBot")
        #change colors
        self.setRGBAofAltBot(mimicSkel)
        #turn off collisions 
        self._setSkelNoCollide(mimicSkel, mimicSkel.name)
        #turn off mobile so no frwd sim executed
        mimicSkel.set_mobile(False)
        self.skelHldrs[self.botIdx].setMimicBot(_mimicSkel=mimicSkel)

    def _buildIndivAssistObj(self, assistFileName):
        print ('Constraint Assistance object not found at {} - Building new object'.format(assistFileName))
        flags=defaultdict(int, {'dispCnstrnt':True, 'usePresetAssist':False , 'dbgAssistData':True }) 
        assistDict = self._buildAssistObjCnstrntDefault(flags=flags)
        self.assistObj = assistDict['assistObj']
        print ('Constraint Assistance object built and saved at {}!!'.format(assistDict['objFileName']))
   
    def _buildAssistObjCnstrntDefault(self, flags):
        return self._buildAssistObj(dim=3, initVals=np.array([0.01, 0.01, 0.01]), cmpFuncs = [[[0,1,2],'gauss']],cmpBnds = [[],[],[],[]],cmpNames = ['delCnstrnt x', 'delCnstrnt y', 'delCnstrnt z'],frcIdxs=[],useMultNotForce=False, flags=flags)

    #return assistive component of ANA's observation - put here so can be easily modified
    def getSkelAssistObs(self, skelHldr):
        #constraint displacement component
        cnstrntDispObs = self.trackTraj.getTrajObs('disp')
        return cnstrntDispObs

    #return the names of each dof of the assistive component for this environment
    def getSkelAssistObsNames(self):
        trajTarStr = 'traj disp x, traj disp y, traj disp z'
        return trajTarStr

    #initialize assist component at beginning of reset by querying end loc from fleish polynomial dist
    def _resetAssist_Indiv(self):
        self.endEefTarLoc = self.sampleGoalEefRelLoc()
        self.trajResetEndLoc = self.endEefTarLoc

    #call this for bot prestep - For IK (kinematic solution) only so far.  Need to modify to accept dynamic solution solver
    def botPreStep(self):
        #print("Bot prestep")
        #calc robot optimization tau/IK Pos per frame
        self.skelHldrs[self.botIdx].preStep(np.array([0]))                  
        #set bot torque to 0 to debug robot behavior or ignore opt result
        #self.skelHldrs[self.botIdx].dbgResetTau()  

    #return a reward threshold below which a rollout is considered bad
    def getMinGoodRO_Thresh(self):
        #TODO should be based on ANA's reward formulation, ideally derived by ANA's performance
        #conversely, should check if ANA is done successfully or if ANA failed
        return 0

    #query value function, find ideal assist given ANA's state, find necessary bot control to provide this assist
    #will set bot's tau, and will return new control for ANA
    def findBotDispCntrol_IKSPD(self, ANA, bot, ANAObs, policy, useDet):
        #query VF with current ana state
        _,initTardisp,_ = self.getObsComponents(ANAObs)
        tarDisp, _ = self.getTargetAssist(ANAObs)
        #print("Init Tar Disp : {} VF Tar Disp : {}".format(initTardisp, tarDisp))
        if(not np.allclose(initTardisp, tarDisp)):
            #tarDisp = initTardisp
            #set target displacement for tajectory
            self.trackTraj.setVFTrajObs("disp",tarDisp)
            
        #derive the control torque by determining the new pose for the assistant robot given the desired displacement
        bot.frwrdSimBot_DispIKSPD(tarDisp, dbgFrwrdStep=False)

        #build an observation, query policy for optimal 
        ANAObs[-(len(tarDisp)):len(ANAObs)] = tarDisp
        
        #use ANA observation in policy to get appropriate action
        if (policy is None):
            return None
        else :
            action, actionStats = policy.get_action(ANAObs)
            if(useDet):#deterministic policy - use mean
                action = actionStats['mean']
        return action            

    #TODO need to rebuild all these things to handle constraint displacement and not force            
    #use this to perform per-step bot opt control of force and feed it to ANA - use only when consuming policy
    #ANA and bot are not connected explicitly - bot is optimized to follow traj while generating force, 
    #ANA is connected to traj constraint.
    def stepBotForAssist(self, a, ANA):
        bot = self.skelHldrs[self.botIdx]
        #values set externally via self.setCurrPolExpDict()
        policy = self.ANAPolicy
        useDet = not self.useRndPol
        #vfOptObj = self.vfOptObj
        print('Using Bot Assist for step {}'.format(self.sim_steps))
        actionsUsed = []
        #forward simulate  - dummy res dict
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhldr':'None', 'reason':'OK', 'skelState':self.skelHldrs[self.humanIdx].state_vector()}
        done = False
        fr = 0
        #iterate every frame step, until done
        while (not done) and (fr < self.frame_skip):  
            fr +=1  
            #if bot not connected allow traj to evolve
            if (not self.connectBot):
                self.doneTracking = self.stepTraj(fr)               

            ANAObs = ANA.getObs()
            action = self.findBotDispCntrol_IKSPD(ANA, bot, ANAObs,policy,useDet)
            if action is None:
                action = a
                print("stepBotForAssist : No Policy set : action used :  {}".format(action))

            actionsUsed.append(action)
            #do not call prestep with new action in ANA, this will corrupt reward function, since initial COM used for height and height vel calcs is set in prestep before frameskip loops
            #set clamped tau in ana to action
            ANA.tau=ANA.setClampedTau(action) 
            
            #step all skels
            #apply all torques to skeletons
            for _,v in self.skelHldrs.items():
                #tau is set in calling prestep function - needs to be sent to skeleton every sim step
                v.applyTau()              
            self.dart_world.step()    

            #after every sim step, run this - frame is expected to have been incrmented by here
            done, self.assistResDictList = self.perFrameStepCheck(ANA,resDict, fr, self.solvingBot, self.dbgANAEefFrc,self.dbgANAFrcDet, self.assistResDictList, stopNow=(self.doneTracking and self.stopWhenTrajDone), botDict={})   
            #print("----------------------- End frame skip {} of {} -----------------------------".format(fr, self.frame_skip))   

        #print("----------------------- End Step -----------------------------")   
        #build avg action
        if self.frame_skip == 1:
            avAction=actionsUsed[0]
        else :
            avAction = np.copy(actionsUsed[0])
            numActions = len(actionsUsed)
            for i in range(1,numActions):
                avAction += actionsUsed[i]
            avAction /= numActions
        #self.pauseForInput("stepBotForAssist")

        return self.endStepChkANA(avAction,  done, resDict, self.dbgANAReward)


    #individual code for prestep setup, before frames loop
    def preStepSetUp_indiv(self, ANA):
        pass

    #code for assist robot per frame per step in vanilla step function
    def botPerFramePerStep_indiv(self):
        if(self.solvingBot): 
            self.botPreStep()  
        

   #will return list of most recent step's per-bot force result dictionaries - list of dicts (2-key) each key pointing to either an empty dictionary or a dict holding the force results generated by that bot
    def getPerStepEefFrcDicts(self):
        #per-frame ara of tuples of frcDictionaries for bot and ana, keyed by frame
        return self.assistResDictList 
        
    #take a raw assistance proposal from value function optimization, and find the actual assist displacement to use
    def getTarAssist_indiv(self, vfExists, val, rawAssist, origAssist):
        if vfExists : 
            rawDispAssist = rawAssist[0:]
            #TODO need to clip this?  should be some max value for assistance
            print('standUp3d_2Bot_Cnstrnt::getTarAssist_indiv : vf pred score : {}| pred rawDispAssist : {}'.format(val, rawDispAssist))
            return rawDispAssist, rawDispAssist
        else :
            print('standUp3d_2Bot_Cnstrnt::getTarAssist_indiv : vf does not exist, returning original observation assist component')
            #without opt object, just send defaults
            return origAssist, origAssist      

    #return whether or not end effector force is calculated for either ANA or helper bot
    # used when consuming a policy, so the policy consumer knows whether or not the dictionary entry for EefFrcResDicts exists or not        
        
    #return whether the passed values are valid for assistance within the limits proscribed by the sim env
    #also return clipped version of chkAssist kept within bounds, if they exist
    def isValidAssist(self, chkAssist):
        inBnds = not ((self.dispBnds[0] > chkAssist).any() or (chkAssist > self.dispBnds[1]).any())
 
        return inBnds#, bndedFrc
    
    #return a 2d array of lower/higher bounds of assist component    
    def getAssistBnds(self):
        #TODO Determine valid bounds of displacement assist vector
        #TODO need to build this based on trajectory displacement vector bounds - these will be much smaller than these bounds
        #tmpBnds = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]]) 
        return self.dispBnds 

    #return a random assist value to initialize vf Opt
    def getRandomAssist(self, assistBnds):
        #return random assist value within bounds - this is seed value for vfOpt - use current value 
        res = self.trackTraj.getTrajObs('disp')#self.trackTraj.getRandDispVal()
        return res

    #set environment-specific values from external call for integrating bot and ana
    #TODO
    def setCurrPolExpDict_indiv(self, initAssistVal):
        pass
    
    #these should only be called during rollouts consuming value function predictions or other optimization results, either force or force mults
    #value function provides actual force given a state - NOTE! does not change state of useSetFrc flag unless chgUseSetFrc is set to true
    #setting passed useSetFrc so that if chgUseSetFrc is specified as true then useSetFrc val must be provided, but otherwise it is ignored
    #set desired assist externally during rollout
    def setAssistDuringRollout(self, val, valDefDict): 
        pass

    ###########################
    #externally called functions
    


    #verifies passed value is force multiplier within specified bounds
    #returns whether legal and reason if not legal - consumed by fitness functions in trpoTests
    def isLegalAssistVal(self, assistVal, isFrcMult):
        #TODO
        return True, 'OK'

    
    #specify this in instanced environment - handle initialization of ANA's skelhandler - environment-specific settings
    def _initAnaKimaSkel(self, skel, skelType, actSclBase,  basePose): 
        
        #used to modify kima's ks values when we used non-zero values - can be removed for ks==0
        ksVal = 0
        kdVal = 5.0
        #use full action scale
        actSclIdxMult = []     
        if hasattr(self, 'setInitPoseAsGoalState') and self.setInitPoseAsGoalState:
            #goal pose - override standard pose of "seated"
            basePose = self.getPoseVals('goal', skel, skelType)
        #not using below currently - no joint torque scaling
#        else :
#            isKimaNew = ('new' in skelType.lower())
#            #standard kima skel - pose is seated with arm raised - already set to this before call from dart_env_2bot
#            #multipliers derived via heuristic
#            if (isKimaNew):
#                print('Using Expanded Action Scales on New Kima')
#                actSclIdxMult = [([0, 2, 6, 8], .5),  #spread both legs, twist both legs
#                            ([1, 7],.7), #thighs pull
#                            ([3,9], .7), #knee 
#                            ([4,5, 10, 11], .4), #feet 
#                            ([15,17, 19, 21], .7), #left and right shoulders/bicep, 2 for each bicep
#                            ([18, 22], .3)] #scale forearms 
#            else:
#                print('Using Expanded Action Scales on Old Kima')
#                actSclIdxMult = [ ([1, 2, 7, 8], .7),   #thigh twist and spread 
#                        ([15,17, 19, 21], .7),          #2 of bicep, and hands
#                        ([4,5, 10, 11], .5),            #ankles
#                        ([18, 22], .75 )]               #forearms
        self._fixJointVals(skel, kd=kdVal, ks=ksVal)  
        return actSclBase, actSclIdxMult, basePose