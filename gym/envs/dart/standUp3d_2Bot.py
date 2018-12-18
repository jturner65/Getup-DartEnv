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
#class DartStandUp3dAssistEnv(dart_env_2bot.DartEnv2Bot, utils.EzPickle):
class DartStandUp3dAssistEnv(assist2bot_env.DartAssist2Bot_Env, utils.EzPickle):
    def __init__(self): #,args): #set args in registration listing in __init__.py
        """
        This class will manage external/interacting desired force 
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
        fs = 5.0
        ts = .01/fs
        assist2bot_env.DartAssist2Bot_Env.__init__(self, modelLocs, fs, dt=ts, helperBotLoc=kr5HelperBotLoc, disableViewer=False)        
        #initialize all force and trajectory values: extAssistSize=6 will add force application location target to observation
        self.initAssistTrajVals(extAssistSize=3, useANAHeightTraj=False, trajTyp='linear', setTrajDynamic=False, setTrajPassive=False)
        
        #connect human to constraint 
        self.connectHuman = False
        #connect bot to constraint
        self.connectBot = False
        
        #whether or not to stop when trajectory is finished
        self.stopWhenTrajDone = False
        #display debug information regarding force application - turn off if training
        self.dbgAssistData = True             
        #display ANA reward dbg data - slow, make false if training
        self.dbgANAReward = True
        #calc and display post-step ANA eef force - make false if training
        self.dbgANAEefFrc = False
        #display full ANA force results if dbgANAEefFrc is true
        self.dbgANAFrcDet = False
        ############################
        # set human/robot conection and motion trajectory object and params  - if train==True then overrides dynamicbot to be false (OPT to solve for location is expensive)
        #self.trainPolicy is defined as static variable in dart_env_2bot TODO make parent class method that enables this to be trained -after- object is instanced
        #setBotSolve : whether or not helper bot's motion is solved
        #setBotDynamic : whether bot is set to be dynamically simulated or not (if not mobile is set to false)
        #botSolvingMethod is type of solving Bot should engage in : 0 is IK, 1 is constraint optimization dyn, 2 is IK-SPD 
        #spd gain is only used for IK_SPD solve 
        botDict = defaultdict(int,{'setBotSolve':0, 'setBotDynamic':1, 'botSolvingMethod':0, 'SPDGain':10000000,'frcBasedAssist':True})
        self.setTrainAndInitBotState(self.trainPolicy, botDict=botDict)                                
        utils.EzPickle.__init__(self)
 
    #individual environment-related components of assist initialization
    def initAssistTraj_indiv(self): 
        #whether to train on force multiplier (if false train on actual force, which has been status quo so far - this may lead to instabilities since input magnitude is much higher than other observation components?
        self.useMultNotForce = True
        #whether to always use specific assist value set in init or to randomly regenerate assist at each reset 
        self.usePresetAssist = False         
        #list of x,y,z initial assist force multipliers
        self.frcMult = np.array([0.0, 0.0, 0.0])
        print('INIT ASSIST FORCE MULTIPLIER BEING SET TO : {}'.format(self.frcMult))
        if(self.usePresetAssist) :
            print('!!!!!!!!!!!!!!!!!!!!!!!! Warning : DartStandUp3dAssistEnv:ctor using hard coded force multiplier {} for all rollouts !!!!!!!!!!!!!!'.format(self.frcMult))        
        #bounds of force mult to be used in random force generation
        self.frcMultBnds = np.array([[0.0,0.0,-0.001],[0.2, 0.5, 0.001]])
        self.frcBnds = self.getForceFromMult(self.frcMultBnds)  
        
    #handl 
    def _buildIndivAssistObj(self, assistFileName):
        print ('Force Assistance object not found at {} - Building new object'.format(assistFileName))
        flags=defaultdict(int, {'hasForce':True, 'useMultNotForce':self.useMultNotForce, 'usePresetAssist':self.usePresetAssist , 'dbgAssistData':True })    
        assistDict = self._buildAssistObjFrcDefault(flags)
        self.assistObj = assistDict['assistObj']
        print ('Force Assistance object built and saved at {}!!'.format(assistDict['objFileName']))


    #return assistive component of ANA's observation - put here so can be easily modified
    def getSkelAssistObs(self, skelHldr):
        #frc component
        if self.useMultNotForce : 
            frcObs = self.frcMult
        else :
            frcObs = self.assistForce
        
        #frc application target loc on traj element, if used
        if self.extAssistSize == 6:
           tarLoc = skelHldr.cnstrntBody.to_world(x=skelHldr.cnstrntOnBallLoc)
           return np.concatenate([frcObs, tarLoc])        
        return frcObs

    #return the names of each dof of the assistive component for this environment
    def getSkelAssistObsNames(self):
        if self.useMultNotForce:
            resStr = 'frcMult x, frcMult y, frcMult z'
        else :
            resStr = 'frc x, frc y, frc z'
        if self.extAssistSize == 6:
            trajTarStr = 'traj tar x, traj tar y, traj tar z'
            resStr = '{},{}'.format(resStr, trajTarStr)
        return resStr
        
    #call this for bot prestep
    def botPreStep(self, frc, frcNorm, recip=True):
        #set sphere forces to counteract gravity, to hold still in space if bot is applying appropriate force
        #self.grabLink.bodynodes[0].set_ext_force(self.sphereForce)        
        #set the desired force the robot wants to generate 
        self._setTargetAssist(self.skelHldrs[self.botIdx],frc, frcNorm, reciprocal=recip)
        #calc robot optimization tau/IK Pos per frame
        self.skelHldrs[self.botIdx].preStep(np.array([0]))                  
        #set bot torque to 0 to debug robot behavior or ignore opt result
        #self.skelHldrs[self.botIdx].dbgResetTau()  

    #return a reward threshold below which a rollout is considered bad
    def getMinGoodRO_Thresh(self):
        #TODO should be based on ANA's reward formulation, ideally derived by ANA's performance
        #conversely, should check if ANA is done successfully or if ANA failed
        return 0

       
    #sets assist force from pre-set force multipliers, and broadcasts force to all skeleton handlers  
    def initAssistForce(self, frc, frcMult):
        self._buildFrcCnstrcts(frc,frcMult)
        for k,hndlr in self.skelHldrs.items(): 
            self._setTargetAssist(hndlr, self.assistForce, self.frcNormalized, reciprocal=False)
          
    #perform forward step of trajectory, forward step bot to find bot's actual force generated
    #then use this force to modify ANA's observation, which will then be used to query policy for 
    #new action.
    #BOT'S STATE IS NOT RESTORED if  restoreBotSt=False
    def frwrdStepTrajBotGetAnaAction(self, fr, useDet, policy, tarFrc, recip, restoreBotSt=True):
        ANA = self.skelHldrs[self.humanIdx]
        #derive target trajectory location using current ANA's height or by evolving trajectory
        #let human height determine location of tracking ball along trajectory    
        #TODO As of 11/1/18, trajectory evolution has been modified so this will need rebuilding  
        self.doneTracking = self.stepTraj(fr) 
        #solve for bot's actual force based on target force
        #returns bot force, bot force multiplier of ana's mg, and a dictionary of various dynamic quantities about the bot
        f_hat, botResFrcDict = self.skelHldrs[self.botIdx].frwrdStepBotForFrcVal(tarFrc, recip, restoreBotSt=restoreBotSt, dbgFrwrdStep=False)                  
        #update ANA with force (for observation)
        f_norm = np.linalg.norm(f_hat)
        self._setTargetAssist(ANA, f_hat, f_norm, reciprocal=False)
        #get ANA observation
        anaObs = ANA.getObs()
        #use ANA observation in policy to get appropriate action
        action, actionStats = policy.get_action(anaObs)
        if(useDet):#deterministic policy - use mean
            action = actionStats['mean']             
        return action, botResFrcDict

    #individual code for prestep setup, before frames loop
    #send target force to ANA - this force must get applied whenever tau is applied unless ANA is coupled directly to helper bot
    def preStepSetUp_indiv(self, ANA):
        self._setTargetAssist(ANA,self.assistForce, self.frcNormalized, reciprocal=False)

    #use this to perform per-step bot opt control of force and feed it to ANA - use only when consuming policy
    #ANA and bot are not connected explicitly - bot is optimized to follow traj while generating force, 
    #ANA is connected to traj constraint.
    def stepBotForAssist(self, a, ANA):
        #values set externally via self.setCurrPolExpDict()
        policy = self.ANAPolicy
        useDet = not self.useRndPol
        print('Using FHat for step {}'.format(self.sim_steps))
        actionUsed = np.zeros(a.shape)
        botRecipFrc = True
        
        #forward simulate  - dummy res dict
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhldr':'None', 'reason':'OK', 'skelState':self.skelHldrs[self.humanIdx].state_vector()}
        done = False
        fr = 0
        #get target force, either from currently assigned assist force, or from querying value function
        tarFrc, tarFrcMult = self.getTargetAssist(ANA.getObs())
        #iterate every frame step, until done
        while (not done) and (fr < self.frame_skip):  
            fr +=1  
#            #get target force, either from currently assigned assist force, or from querying value function
#            tarFrc, tarFrcMult = self.getTargetAssist(ANA.getObs())
            #find new action for ANA based on helper bot's effort to generate target force
            action, botResFrcDict = self.frwrdStepTrajBotGetAnaAction(fr, useDet, policy, tarFrc, botRecipFrc, restoreBotSt=False)
            #save for dbg
            actionUsed += action
            #do not call prestep with new action in ANA, this will corrupt reward function, since initial COM used for height and height vel calcs is set in prestep before frameskip loops
            #set clamped tau in ana to action
            ANA.tau=ANA.setClampedTau(action) 
            #force this to be true here, since human and bot are not connected explicitly
            ANA.setAssistFrcEveryTauApply = True 
            #send torques to human skel (and reapply assit force)
            ANA.applyTau()
            
            #step all skels but helper bot
            self.skelHldrs[self.botIdx]._stepAllButMe()

             #after every sim step, run this
            done, self.assistResDictList = self.perFrameStepCheck(ANA,resDict, fr, self.solvingBot, self.dbgANAEefFrc,self.dbgANAFrcDet, self.assistResDictList, stopNow=(self.doneTracking and self.stopWhenTrajDone), botDict=botResFrcDict)   
            #done = self.perFrameStepCheck(ANA,resDict, fr, self.solvingBot, self.dbgANAEefFrc, botDict=botResFrcDict, stopWhenTrajDone=True)            
    
        #calc avg action used over per frame over frameskip, get reward and finish step
        if(fr > 0):#if only 1 action before sim broke, don't avg action
            actionUsed /= (1.0 * fr)
            
        return self.endStepChkANA(actionUsed, done, resDict, self.dbgANAReward)

    #code for assist robot per frame per step in vanilla step function
    def botPerFramePerStep_indiv(self):
        if(self.solvingBot): 
            self.botPreStep(self.assistForce, self.frcNormalized, recip=True)  
 
    #will return list of most recent step's per-bot force result dictionaries - list of dicts (2-key) each key pointing to either an empty dictionary or a dict holding the force results generated by that bot
    def getPerStepEefFrcDicts(self):
        #per-frame ara of tuples of frcDictionaries for bot and ana, keyed by frame
        return self.assistResDictList 
    
    #take a raw assistance proposal from value function optimization, and find the actual assist force used
    def getTarAssist_indiv(self, vfExists, val, rawAssist, origAssist):
        if vfExists : 
            rawFrcAssist = rawAssist[0:]
            #clip force mult to be within bounds - check if force or frc mult
            if self.useMultNotForce : #if vf used multiplier not force itself
                tarFrcMultRaw = rawFrcAssist
            else :#uses actual force in vfOpt
                tarFrcMultRaw = self.getMultFromFrc(rawFrcAssist) 
            tarFrcMult = np.clip(tarFrcMultRaw, self.frcMultBnds[0], self.frcMultBnds[1])
            tarFrc = self.getForceFromMult(tarFrcMult)
            print('standUp3d_2Bot::getTarAssist_indiv : vf pred score : {}| pred tarFrcMult : {}| clipped tar frc : {}| clipped tar frc mult : {}'.format(val, tarFrcMultRaw, tarFrc, tarFrcMult))
            return tarFrc, tarFrcMult
        else :
            #without opt object, just send original
            #TODO
            return self.assistForce, self.frcMult        
        
    #return whether or not end effector force is calculated for either ANA or helper bot
    # used when consuming a policy, so the policy consumer knows whether or not the dictionary entry for EefFrcResDicts exists or not        
        
    #return whether the passed values are valid for assistance within the limits proscribed by the sim
    #also return clipped version of chkAssist kept within bounds
    def isValidAssist(self, chkAssist):
        if self.useMultNotForce :              
            inBnds = not ((self.frcMultBnds[0] > chkAssist).any() or (chkAssist > self.frcMultBnds[1]).any())
        else :            
            inBnds = not ((self.frcBnds[0] > chkAssist).any() or (chkAssist > self.frcBnds[1]).any())
        return inBnds#, bndedFrc
    
    #return a 2d array of lower/higher bounds of assist component    
    def getAssistBnds(self):
        return np.copy(self.frcMultBnds) 

    #initialize assist force, either randomly or with force described by env consumer
    def _resetAssist_Indiv(self):
        if(not self.usePresetAssist):  #using random force 
            frc, fMult, _ = self._getRndFrcAndMults(self.frcMultBnds)
        #set assistive force vector for this rollout based on current self.frcMult, and send to all skel holders
        self.initAssistForce(frc, fMult)
        #what to use for traj reset end loc
        self.trajResetEndLoc = self.frcMult
        if(self.dbgAssistData):
            print('_resetAssist_Indiv setting : multX : {:.3f} |multY : {:.3f}|multZ : {:.3f}\nforce vec : {}'.format(self.frcMult[0],self.frcMult[1],self.frcMult[2],['{:.3f}'.format(i) for i in self.assistForce[:3]]))


    #returns a vector of mults and a force vector randomly generated - DOES NOT SET ANY FORCE VALUES
    #also returns value actually used in observation
    def _getRndFrcAndMults(self, frcMultBnds):
        frcMult = self.np_random.uniform(low=frcMultBnds[0], high=frcMultBnds[1])
        frc = self.getForceFromMult(frcMult)
        obsVal = frcMult if self.useMultNotForce else frc
        return frc, frcMult, obsVal

    #return a random assist value to initialize vf Opt
    def getRandomAssist(self, assistBnds):
        #return random assist value within bounds
        _,_,obsVal = self._getRndFrcAndMults(assistBnds)
        return obsVal

    def setCurrPolExpDict_indiv(self, initAssistVal):
        #Set to be constant and set to frcMult - is overridden if vfOptObj is included
        self.usePresetAssist = True         
        #set assistive force vector for this rollout based on current self.frcMult, and send to all skel holders
        if self.useMultNotForce :
            frc = self.getForceFromMult(initAssistVal)
            frcMult = initAssistVal
        else : 
            frc = initAssistVal
            frcMult = self.getMultFromFrc(initAssistVal)           
        
        self.initAssistForce(frc, frcMult)
        print('standUp3d_2Bot::setCurrPolExpDict : Forcing assist force to be constant value : {}'.format(self.assistForce))
       
    ###########################
    #externally called functions
    
    #find bot force to be used for current step and restore bot state - step trajectory forward as well, and then restore trajectory
    #called externally just to get a single force value given current test frc value
    def frwrdStepTrajAndBotForFrcVal(self, tarFrc=None, recip=True, dbgFrwrdStep=True):
        if(tarFrc is None):
            tarFrc=self.assistForce
        print('Using Target force {} with bot'.format(self.assistForce))
        #save current traj state 
        self.trackTraj.saveCurVals()
        #either use ana's height for trajectory location or move trajectory kinematically to evolve traj
        self.doneTracking = self.stepTraj(1)  
        #solve for bot's actual force
        f_hat, botResDict = self.skelHldrs[self.botIdx].frwrdStepBotForFrcVal(tarFrc, recip, restoreBotSt=True, dbgFrwrdStep=dbgFrwrdStep)   
        #get appropriate f_hat multiplier based on ANA's mass
        fMult_hat = self.getMultFromFrc(f_hat)
        if(dbgFrwrdStep):
            print('frwrdStepTrajAndBotForFrcVal : actual bot assist force : {} : actual assist multiplier : {}'.format(f_hat, fMult_hat))        

        #restore trajectory position
        self.trackTraj.restoreSavedVals()        
        #return bot force, force mult, and bot resultant dyanmic quantities in dictionary
        return f_hat, fMult_hat, botResDict

    ####################################
    ## Externally called / possibly deprecated or outdated methods

  
    #build assist components - force vec, force multiplier vector and frc dir vector
    #call internally - TODO replace this with code in assistClass
    def _buildFrcCnstrcts(self, frc, frcMult):
        self.frcMult = frcMult
        self.assistForce = frc
        norm=np.linalg.norm(frcMult)
        if (norm == 0) :
            self.frcNormalized = np.array([0,0,0]) 
        else : 
            self.frcNormalized = (frcMult / norm)

    #this should only be called during rollouts consuming value function predictions or other optimization results, either force or force mults
    #value function provides actual force given a state - NOTE! does not change state of useSetFrc flag unless chgUseSetFrc is set to true
    #setting passed useSetFrc so that if chgUseSetFrc is specified as true then useSetFrc val must be provided, but otherwise it is ignored
    #set desired assist externally during rollout
    def setAssistDuringRollout(self, val, valDefDict=defaultdict(int)): 
        if(valDefDict['chgUseSetFrc']==1):
            self.usePresetAssist = (valDefDict['useSetFrc']==1)
        if valDefDict['passedMultNotFrc']==1: #val is mult or force- need to use this to set 
            self._buildFrcCnstrcts(self.getForceFromMult(val),val)                
        else :            #val is frc
            self._buildFrcCnstrcts(val, self.getMultFromFrc(val))
        for k,hndlr in self.skelHldrs.items():
            self._setTargetAssist(hndlr, self.assistForce, self.frcNormalized, reciprocal=False)
            
    def setAssistFrcMultDuringRollout(self, frcMult, chgUseSetFrc, useSetFrc=None):
        if(chgUseSetFrc):
            self.usePresetAssist = useSetFrc
        self._buildFrcCnstrcts(self.getForceFromMult(frcMult),frcMult)
        for k,hndlr in self.skelHldrs.items():
            self._setTargetAssist(hndlr, self.assistForce, self.frcNormalized,  reciprocal=False)
     
    #set passed skelHandler's desired force - either target force or force generated by bot arm.  
    #bot will always used target assist force, while human will use either target or bot-generated
    #only call internally ?
    def _setTargetAssist(self, skelhldr, desFrc, desFMultNorm, reciprocal):
        #reciprocal force is only set for assist robot, and only for debugging dynamics optimization
        skelhldr.setDesiredExtAssist(desFrc, desFMultNorm, setReciprocal=reciprocal)    
           
    ################
    ##### old; to be removed
    # #called externally to set force multiplier - sets reset to use this force and not random force
    # def setForceMult(self, frcMult):
    #     self.usePresetAssist = True
    #     self.frcMult = np.copy(frcMult)    
            
    # #given a specific force, set the force multiplier, which is used to set force
    # #while this seems redundant, it is intended to preserve the single entry point 
    # #to force multiplier modification during scene reset
    # def setFrcMultFromFrc(self, frc):
    #     frcMult = self.getMultFromFrc(frc) 
    #     #print('frcMultx = {} | frcMulty = {}'.format(frcMultX,frcMultY))
    #     self.setForceMult(frcMult)
    
    #verifies passed value is force multiplier within specified bounds
    #returns whether legal and reason if not legal - consumed by fitness functions in trpoTests
    def isLegalAssistVal(self, assistVal, isFrcMult):
        if len(assistVal) != self.extAssistSize:#verify size
            return False, 'TOTAL SIZE'
        #check force multiplier component
        numFrcVals = len(self.frcMult)
        frcComp = assistVal[0:numFrcVals]
        locComp = assistVal[numFrcVals:len(assistVal)]
        if isFrcMult :             
            frcMult = frcComp
        else :
            frcMult = self.getMultFromFrc(frcComp)
        #make sure frcMult is within bounds
        if not self.isValidAssist(frcMult) :
            return False, 'VAL OOB'
        #check location component if exists, to be within reasonable, reachable distance of ANA
        if (self.extAssistSize) > 3 :
            #locComp
            pass
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