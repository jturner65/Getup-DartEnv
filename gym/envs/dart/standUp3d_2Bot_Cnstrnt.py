#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 

import numpy as np
#import pydart2 as pydart
from gym import utils
from gym.envs.dart import assist2bot_env
from os import path
from skelHolders import robotArmSkelHolder, robotArmDispSkelHolder

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
        #NOTE :if bot is connected and actively solving, trajectory _MUST_ be passive or explodes
        #set forcePassive to true, setTrajDyn to true to use optimization process
        #set forcePassive and setTrajDyn to false, use robotArmSkelHolder, to demo behavior
        #set to false to use IK
        forcePassive = True
        #trajTyp = 'servo'       #servo joint cannot be moved by external force - can train with servo, but to use bot must use freejoint
        trajTyp = 'gauss'       #gauss joint cannot be solved dynamically, must be displaced kinematically, or else set to passive
        self.initAssistTrajVals(extAssistSize=3, useANAHeightTraj=False,  trajTyp=trajTyp, setTrajDynamic=True, setTrajPassive=forcePassive)#(self.connectBot or forcePassive))
        #self.initAssistTrajVals(extAssistSize=3, useANAHeightTraj=False,  trajTyp='gauss', setTrajDynamic=True, setTrajPassive=False)
        
        #whether or not to stop when trajectory is finished
        self.stopWhenTrajDone = False
        #allow pause directives in code, wait on user input for debugging
        self.allowPauseForInput = True
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
        #self.trainPolicy is defined as static variable in dart_env_2bot TODO make parent class method that enables this to be trained -after- object is instanced (??)
        #setBotSolve : whether or not helper bot's motion is solved
        #setBotDynamic : whether bot is set to be dynamically simulated or not (if not mobile is set to false)
        #botSolvingMethod is type of solving Bot should engage in : 0 is IK, 1 is constraint optimization dyn, 2 is IK-SPD, 3 is mimic generates desired target force to be applied to bot
        #spd gain is only used for IK_SPD solve 
        #frcBasedAssist is whether this environment uses a force as the assistive component of observation, or something else, such as displacement
        botDict = defaultdict(int,{'setBotSolve':1, 'setBotDynamic':1, 'botSolvingMethod':2, 'SPDGain':10000000, 'frcBasedAssist':False})
        self.setTrainAndInitBotState(self.trainPolicy, botDict=botDict)    

        utils.EzPickle.__init__(self)

    #return appropriate robot arm skel holder for this environment
    def getBotSkelHldr(self, skel, widx, stIDX, eefOffset):
        #return robotArmSkelHolder(self, skel, widx, stIDX, fTipOffset=eefOffset)
        return robotArmDispSkelHolder(self, skel, widx, stIDX, fTipOffset=eefOffset) 

    #individual environment-related components of assist initialization
    def initAssistTraj_indiv(self): 
        #this env uses displacements.  Here we should initialize the bounds of those displacements
        #displacement bounds for this trajectory
        self.dispBnds = self.trackTraj.getMinMaxBnds()#np.array([[0.0,0.0,-0.001],[0.2, 0.5, 0.001]]) 
        print("initAssistTraj_indiv : disp bnds : {} ".format(self.dispBnds))    

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

    #returns optimal displacement and action from policy query with opt displacement for current ANA state
    def getOptDispAndANAAction(self, ANAObs, lastAction, policy, useDet):
        #get displacement component of current observation
        _,initTardisp,_ = self.getObsComponents(ANAObs)
        #query VF with current ana state to get optimal displacement
        tarDisp, _ = self.getVFTargetAssist(ANAObs) 
        print("standUp3d_2Bot_Cnstrnt::getOptDispAndANAAction : Init Tar Disp : {} VF Tar Disp : {}".format(initTardisp, tarDisp))

        #TODO forcing target displacement to be initial displacement (i.e. ignoring vf pred) - only do this until vf process is operational        
        useDisp = initTardisp # tarDisp
        #set target displacement for tajectory
        #self.trackTraj.setVFTrajObs("disp",useDisp)
        self.trackTraj.setOptDeriveMovVec(useDisp)

        #if not changing traj (opt tar is very close to expected tar), then policy won't change either
        if (policy is None) or (np.allclose(initTardisp,tarDisp)):
            return lastAction, tarDisp, initTardisp, useDisp

        #get new observation vector with this optimal action, use to query new policy
        newANAObs = np.copy(ANAObs)
        newANAObs[-(len(useDisp)):] = useDisp
        action, actionStats = policy.get_action(newANAObs)
        
        if(useDet):#deterministic policy - use mean
            action = actionStats['mean']
        return action, tarDisp, initTardisp, useDisp

    # #given ANA's current state, we find the appropriate control for the assitant bot
    # def findBotControlViaOptimization(self, ANA, ANAObs, bot, lastAction, policy, useDet):
    #     #first determine optimal displacement given ANA's current state
    #     action, tarDisp, initTardisp, useDisp = self.getOptDispAndANAAction(ANAObs, lastAction, policy, useDet)

    #     return lastAction

    # #query value function, find ideal assist displacement given ANA's state, 
    # #frwrd sim ANA to find ext seen at ANA's eef to generate this displacement. restore ANA
    # #find bot control to synthesize this force at eef
    # def findBotDispCntrl_DispFrc(self, ANA, ANAObs, bot, lastAction, policy, useDet):
    #     ANAObs = ANA.getObs()
    #     action, tarDisp, origDisp, useDisp = self.getOptDispAndANAAction(ANAObs, lastAction, policy, useDet)
        
    #     #Step sim forward with passive bot to get ANA's eef force dictionary for given action and target displacement, to see how much force the bot should generate 
    #     eefFrc = self.stepFrwrdThenRestore(action)
                  
    #     #derive the control torque via optimization necessary to generate force seen at EEF by ANA
    #     #bot.frwrdSimBot_DispToFrc(-1*eefFrc, dbgFrwrdStep=True)
    #     #bot.frwrdSimBot_DispToFrc(eefFrc, dbgFrwrdStep=True)
    #     #
        
    #     #use ANA observation in policy to get appropriate action
    #     if (policy is None):
    #         return lastAction
    #     else :
    #         action, actionStats = policy.get_action(ANAObs)
    #         if(useDet):#deterministic policy - use mean
    #             action = actionStats['mean']
    #     return action      
    
    #query value function, find ideal assist given ANA's state, find necessary bot control to provide this assist
    #will set bot's tau, and will return new control for ANA
    #Not used
    # def findBotDispCntrl_IKSPD(self, ANA, ANAObs, bot, lastAction, policy, useDet):
    #     ANAObs = ANA.getObs()
    #     action, tarDisp, origDisp, useDisp = self.getOptDispAndANAAction(ANAObs, lastAction, policy, useDet)
    #     #query VF with current ana state           
    #     #derive the control torque by determining the new pose for the assistant robot given the desired displacement
    #     bot.frwrdSimBot_DispIKSPD(tarDisp, dbgFrwrdStep=False)

    #     #build an observation, query policy for optimal 
    #     ANAObs[-(len(tarDisp)):] = tarDisp
        
    #     #use ANA observation in policy to get appropriate action
    #     if (policy is None):
    #         return None
    #     else :
    #         action, actionStats = policy.get_action(ANAObs)
    #         if(useDet):#deterministic policy - use mean
    #             action = actionStats['mean']
    #     return action            

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
        print('Using Bot Assist for step {} | Policy exists : {} | using policy mean : {} '.format(self.sim_steps, (policy!=None), useDet))
        actionsUsed = []
        #forward simulate  - dummy res dict
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhldr':'None', 'reason':'OK', 'skelState':self.skelHldrs[self.humanIdx].state_vector()}
        done = False
        fr = 0
        lastAction = a
        action = a

        #iterate every frame step, until done
        while (not done) and (fr < self.frame_skip):  
            fr +=1  
            #get ANA's current observed state
            ANAObs = ANA.getObs()            
            #find best action for ANA based on optimal displacement for current state
            #tarDisp is vf pred proposal for best displacement, initTarDisp is displacement from observation (displacement set from trajectory evolution)
            # useDisp is displacement being used to generate new observation/control (either tarDisp orr initTarDisp - this is here for debugging optimization)
            action, tarDisp, initTarDisp, useDisp = self.getOptDispAndANAAction(ANAObs, lastAction, policy, useDet)

            actionsUsed.append(action)
            #do not call prestep with new action in ANA, this will corrupt reward function, since initial COM used for height and height vel calcs is set in prestep before frameskip loops
            #set clamped tau in ana to action
            ANA.tau=ANA.setClampedTau(action) 

            #set displacement for bot to use in optimization - only defined for robotArmDispSkelHolder class 
            #solves for bot torque and applies to bot
            bot.solveBotOptCntrlForDisp(useDisp)
            #allow traj component to evolve - if is passive, just use to synthesize approximation for next displacement 
            self.doneTracking = self.stepTraj(fr)        

            input()       
           
            #step all skels
            #apply all torques to skeletons
            for _,v in self.skelHldrs.items():
                #tau is set in calling prestep function - needs to be sent to skeleton every sim step
                v.applyTau()              
            self.dart_world.step()    
            lastAction = action
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

    def _dbgOneSkelCnstrntFrcVals(self, valsToDisp, useLin, hldr, hldrName, showCntct):
        resVals, resDict = hldr.dbgCalcAllEefFrcComps(hldrName)
        valsToDisp.extend(resVals)
        
        resDict['cntctFrc'] = hldr.getTtlCntctEefFrcs(useLin)
        if (showCntct):
            valsToDisp.append("{} cntctfrc : \t{}".format(hldrName, resDict['cntctFrc']))
            
        resDict['eefCnstrntFrc'] = hldr.getEefCnstrntFrc(useLin)  #jPullTransInv dot cnstrntfrces
        valsToDisp.append("{}EefCnsrtFrc : \t{}".format(hldrName,resDict['eefCnstrntFrc']))
        resDict['aggrEefFrc'] = resDict['jtDotTau'] - resDict['jtDotMA'] - resDict['jtDotCGrav'] + resDict['eefCnstrntFrc']
        valsToDisp.append("{}AggrEefFrc : \t{}".format(hldrName,resDict['aggrEefFrc']))
        valsToDisp.append("")
        return resDict


    #display and pause for input the cnstrnt force seen at balljoint constraint locations for ANA, bot and traj ball
    def dbgCnstrntFrcsSeen(self, eefFrc, ANA, bot, trajObj):
        useLin = ANA.useLinJacob
        valsToDisp =[]
        valsToDisp.append("All frcs in world space at handler's end effector.")
        valsToDisp.append("ANA eefFrc : \t{}".format(eefFrc)) #constraint forces and contact forces taken into account
        valsToDisp.append("")
        anaResDict = self._dbgOneSkelCnstrntFrcVals(valsToDisp, useLin, ANA, "ANA", True) 
        botResDict = self._dbgOneSkelCnstrntFrcVals(valsToDisp, useLin, bot, "bot", False)
        
        #traj obj calc
        trajObj = self.trackTraj.trackObj
        #_skel, _name,  _useLinJacob, _body, _offset):
        trajResVals, trajResDict, JTransInvTraj = ANA.dbgCalcAllEefFrcCompsForPassedSkel(_skel=trajObj, _name="traj", _useLinJacob=useLin, _body= trajObj.body(0), _offset=ANA.cnstrntOnBallLoc)
        valsToDisp.extend(trajResVals)
        trajResDict['eefCnstrntFrc'] = JTransInvTraj.dot(trajObj.constraint_forces())
        valsToDisp.append("{}EefCnsrtFrc : \t{}".format('traj',trajResDict['eefCnstrntFrc']))
        trajResDict['aggrEefFrc'] =  - trajResDict['jtDotMA'] - trajResDict['jtDotCGrav'] + trajResDict['eefCnstrntFrc']
        valsToDisp.append("{}AggrEefFrc : \t{}".format('traj',trajResDict['aggrEefFrc']))

        valsToDisp.append("")
        # frcDiff = eefFrc + botResDict['eefCnstrntFrc'] + trajEefFrc 
        # valsToDisp.append("Diff between Ball cnstrnt frc from Ball POV and eef frc of ANA and BOT : {}".format(frcDiff))
        frcDiffCnstrnt = anaResDict['eefCnstrntFrc'] + anaResDict['cntctFrc'] + botResDict['eefCnstrntFrc'] + trajResDict['eefCnstrntFrc']
        valsToDisp.append("Sum of constraint frcs for traj,bot, ana : {}".format(frcDiffCnstrnt))
        aggrFrcDiff = anaResDict['aggrEefFrc'] + botResDict['aggrEefFrc'] + trajResDict['aggrEefFrc']
        valsToDisp.append("Sum of constraint frc of ball and aggr frc of ANA and BOT : {}".format(aggrFrcDiff))
        self.pauseForInput("DartStandUp3dAssistEnvCnstrnt:stepFrwrdThenRestore", waitOnInput=True, valsToDisp=valsToDisp)  

    #this will step forward all skels but will then restore all states, and will not have any lasting effects on skel state
    #this is done so that frwrd step results can be aggregated and used non-destructively
    #this is to be called from helper bot
    def stepFrwrdThenRestore(self, a):
        ANA = self.skelHldrs[self.humanIdx]
        bot = self.skelHldrs[self.botIdx]
        #preserve all skel states (q,qdot,qdotdot, tau) and whether bot is active or not;
        for _,v in self.skelHldrs.items():
            v._saveSimState()

        #make sure bot is passive exerts no control
        bot.dbgResetTau()
        #make constraint active and save current state
        self.trackTraj.saveAndSetActive()

        #apply ANA's tau (only tau that was set) based on passed action proposal
        ANA.tau = ANA.setClampedTau(a)

        #forward simulate
        self.dart_world.step()   

        #check ANA skel for frc profile
        eefFrc = ANA._calFrcAtEef()
        #display debug info
        self.dbgCnstrntFrcsSeen(eefFrc, ANA, bot, self.trackTraj.trackObj)

        #restore constraint state and make constraint passive
        self.trackTraj.restoreAndSetPassive()
        #restore all states to pre-step state
        for _,v in self.skelHldrs.items():
            v._restoreSimState()
        
        #return the force the bot should generate
        return eefFrc

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
        #TODO this is currently not random on every call
        res = self.trackTraj.getTrajObs('disp')
        #uncomment below to get random assist displacement
        #res = self.trackTraj.getRandDispVal()
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