import numpy as np
#import pydart2 as pydart
from gym import utils
from gym.envs.dart import dart_env_2bot
from os import path

from followTraj import followTraj

#environment where robot/robot arm helps RL policy-driven human get up using force propossal from value function approximation provided by human policy baseline
class DartStandUp3dAssistEnv(dart_env_2bot.DartEnv2Bot, utils.EzPickle):
    def __init__(self):
        """
        This class will manage external/interacting desired force!! (not DartEnv2Bot)
        put all relevant functionality in here, to keep it easily externally accessible
        """
        #set numpy printout precision options
        np.set_printoptions(precision=5)
        np.set_printoptions(suppress=True)
        np.set_printoptions(linewidth=220)
        ########################
        ## loading world/skels
        #modelLocs = ['getUpWithHelperBot3D_damp.skel']     #for two biped skels    
        kr5Loc = path.join('KR5','KR5 sixx R650.urdf')      #for kr5 arm as assistant  
        #modelLocs = [kr5Loc,'getUpWithHelperBot3D_arm.skel']   #regular biped as ana
        #kima biped below with new kima skel - experienced ode issues which killed rllab, might be fixable with updating dart to 6.4.0+ using wenhao's modified lcp.cpp seems to address the issue.
        #modelLocs =  [kr5Loc,'kima/getUpWithHelperBot3D_armKima.skel']
        #kima biped below with kima skel from 2/18 - different joint limits, mapping of dofs and euler angle layout of root - MUST KEEP OLD in file name - used for both new experiments and ak's policy with constraint
        modelLocs = [kr5Loc,'kima/getUpWithHelperBot3D_armKima_old.skel']        
        #set to use old policy (trained with constraint) configurations, or not - eventually get rid of this when reasonable policy is built
        self.useAK_Kima = False
        
        #set false if wanting to consume policy, true to train - this disables viewer
        #if True then using policy will yield error : AttributeError: 'NoneType' object has no attribute 'runSingleStep'
        trainPolicy = False
        print('\n!!!!!!!!!!!!!!!!! Viewer is disabled : {}\n'.format(trainPolicy))

        self.setTrainGAE(modelLocs, trainPolicy, True)
        #dart_env_2bot.DartEnv2Bot.__init__(self, modelLocs, 8, dt=.002, disableViewer=trainPolicy)

        ####################################
        # external/interactive force initialization
        #init and clear all policy, experiment, and vfrd-related quantities so that they exist but are false/clear
        self.clearCurrPolExpDict()
        
        #initialize all force and trajectory values
        if (not self.noAssist):
            self.initAssistFrcTrajVals()
        
        #connect human to constraint 
        self.connectHuman = False
        #connect bot to constraint
        self.connectBot = False

        #display debug information regarding force application - turn off if training
        self.dbgAssistFrcData = True             
        #display ANA reward dbg data - slow, make false if training
        self.dbgANAReward = True
        #calc and display post-step ANA eef force - make false if training
        self.dbgANAEefFrc = False
        #display full ANA force results if dbgANAEefFrc is true
        self.dbgANAFrcDet = False
        ############################
        # set human/robot conection and motion trajectory object and params  - if train==True then overrides dynamicbot to be false (OPT to solve for location is expensive)
        #_solvingBot : whether or not helper bot's motion is solved
        #_dynamicBot : true solves dynamics, false solves IK
        #removed : #_helpingBot : whether or not helper bot is actually coupled to human(i.e. helping by directly applying force) 
        self.setTrainAndInitBotState(trainPolicy, _solvingBot=False, _dynamicBot=False, trajTyp='linear')
                                
        utils.EzPickle.__init__(self)


    #setup to either train getup ala GAE paper or getup normally
    def setTrainGAE(self, modelLocs, trainPolicy, trainGAE):
        if trainGAE : 
            #!! move bot and box out of skel's way - back 2 m in skel file (in z)
            #set pose to be prone
            self.setProne = True
            #if no assistance then set this to true - only use for GAE getup test - get rid of this once test complete
            self.noAssist = True
            #changed frameskip to 1 from 8 to match GAE
            dart_env_2bot.DartEnv2Bot.__init__(self, modelLocs, 1, dt=.01, disableViewer=trainPolicy)
        else :
            #set pose to not be prone
            self.setProne = False
            #if no assistance then set this to true - only use for GAE getup test - get rid of this once test complete
            self.noAssist = False
            dart_env_2bot.DartEnv2Bot.__init__(self, modelLocs, 8, dt=.002, disableViewer=trainPolicy)



    #initialize all assist force and trajectory values
    def initAssistFrcTrajVals(self):
        
        #currently using 3 dof force and 3 dof frc loc : assisting component size for observation
        self.extAssistSize = 6
        self.updateFrcType()
        
        #whether or not the trajectory should be followed
        self.followTraj = True
        #True : ANA height to determine location along trajectory; False : kinematically move traj forward based on frame in rollout 
        self.useANAHeightForTrajLoc = True
        
        #whether to train on force multiplier (if false train on actual force, which has been status quo so far)
        self.useMultNotForce = False
        #whether to always use specific force set here in ctor or to randomly regenerate force
        self.usePresetFrc = False         
        #list of x,y,z initial assist force multipliers
        self.frcMult = np.array([0.05, 0.5, 0.0])
        print('INIT ASSIST FORCE MULTIPLIER BEING SET TO : {}'.format(self.frcMult))
        #set this to nonzero value modify random force gen result for training to test learning - ignored if self.usePresetFrc is true
        self.cheatGetUpMult = np.array([0,0,0])
        if((self.usePresetFrc) or ( np.any(self.cheatGetUpMult > 0))):
            print('!!!!!!!!!!!!!!!!!!!!!!!! Warning : DartStandUp3dAssistEnv:ctor using hard coded force multiplier {} or augmenting the force multiplier by +{}'.format(self.frcMult,self.cheatGetUpMult))        
        #bounds of force mult to be used in random force generation
        self.frcMultBnds = np.array([[0.0,0.3,-0.001],[0.2, 0.8, 0.001]])
        self.frcBnds = self.getForceFromMult(self.frcMultBnds)        


    #return assistive component of ANA's observation - put here so can be easily modified
    def getSkelAssistObs(self, skelHldr):
        if self.noAssist :
            return np.array([])
        #frc component
        frcObs = skelHldr.getObsForce()
        #frc application target on traj element
        tarLoc = skelHldr.cnstrntBody.to_world(x=skelHldr.cnstrntOnBallLoc)
        return np.concatenate([frcObs, tarLoc])
   
    #set essential state flags if training human policy
    #If training human (or purely consuming trained policy) : 
    #   robot should be set to not mobile, not connected to constraint ball, and not simulated/solved (although IK to ball might be good for collision info with human)
    #   human needs to get assist force applied -every timestep-
    #_dynamicBot is simulated - if training then forced to false
    #sovlingBot uses IK to traj loc if not Simulated, solves ID if is simulated, if false and dynamic bot arm is ragdoll
    #helpingBot is connected to human
    def setTrainAndInitBotState(self, _train, _solvingBot=False, _dynamicBot=False, trajTyp='linear'):
        #human is always mobile
        self.skelHldrs[self.humanIdx].setSkelMobile(True)
        #whether we are training or not
        self.trainHuman = _train
        #_dynamicBot bot is fwd simulated - set mobile if true, if immobile (false) either does nothing or solves IK
        self.skelHldrs[self.botIdx].setSkelMobile(_dynamicBot)       
        #bot will solve either IK of eef if not active or ID of force gen if active
        self.solvingBot = _solvingBot

        #set to false to enable robot to help, otherwise, set to true if applying specific force to robot - 
        #external force must be applied every time step
        #must be set before apply_tau unless robot is actively helping
        self.skelHldrs[self.humanIdx].setAssistFrcEveryTauApply = True

        if (_train):
            # #set to false to enable robot to help, otherwise, set to true if training and applying specific force to robot
            # #must be set before apply_tau unless robot is actively helping
            # self.skelHldrs[self.humanIdx].setAssistFrcEveryTauApply = True
            #set mobility - turn off mobility of bot during training
            self.skelHldrs[self.botIdx].setSkelMobile(False) 
            #display ANA reward dbg data - slow, make false if training
            self.dbgANAReward = False
            #calc and display post-step ANA eef force - make false if training
            self.dbgANAEefFrc = False
            #display full ANA force results if dbgANAEefFrc is true
            self.dbgANAFrcDet = False
            #display debug information regarding force application - turn off if training
            self.dbgAssistFrcData = False      
            #turn off all debug displays during training          
            for _,hndlr in self.skelHldrs.items():
                hndlr.debug=False            
        #else :
            # #if not training : 
            # if (_helpingBot):
            #     #do not apply force to human if robot is actively helping - force should be sent to robot via simulation
            #     self.skelHldrs[self.humanIdx].setAssistFrcEveryTauApply = False
            #     pass
            # else :
            #     #force to apply assist every step - demonstrating force without policy
            #     #not training, bot is solving but not helping - if helping then needs to be constrained to human to exert force
            #     if(not _train):# and _solvingBot):
            #         self.skelHldrs[self.humanIdx].setAssistFrcEveryTauApply = True    

        #build trajectory object being used to evolve motion of constraint
        self.trackTraj = self.trackTrajFactory(trajTyp)     
        #self.trackTraj.debug = True
        self.constraintsBuilt = False
        #set human and helper bot starting poses - these are just rough estimates of initial poses - poses will be changed every reset
        self.skelHldrs[self.humanIdx].setStartingPose()
        #set states and constraints - true means set robot init pose, so that it can IK to eef pos
        if (not self.noAssist):
            self._resetEefLocsAndCnstrnts(True)

        
    #call this for bot prestep
    def botPreStep(self, frc, frcMult, recip=True):
        #set sphere forces to counteract gravity, to hold still in space if bot is applying appropriate force
        #self.grabLink.bodynodes[0].set_ext_force(self.sphereForce)
        
        #set the desired force the robot wants to generate 
        self._setTargetForce(self.skelHldrs[self.botIdx],frc,frcMult, reciprocal=recip)
        #calc robot optimization tau/IK Pos per frame
        self.skelHldrs[self.botIdx].preStep(np.array([0]))                  
        #set bot torque to 0 to debug robot behavior or ignore opt result
        #self.skelHldrs[self.botIdx].dbgResetTau()  

    #return a reward threshold below which a rollout is considered bad
    def getMinGoodRO_Thresh(self):
        #TODO should be based on ANA's reward formulation, ideally derived by ANA's performance
        #conversely, should check if ANA is done successfully or if ANA failed
        return 0

    #update observation dimensions of human and robot when force type/size changes
    #extAssistSize is number of components in assistive vector
    def updateFrcType(self):
        #adding 3 to correspond to the location of the target in world space
        self.skelHldrs[self.humanIdx].setObsDim(self.skelHldrs[self.humanIdx].obs_dim + self.extAssistSize)
        #TODO robot observation dim needs to change with pose estimator - not going to have oracle into human's state
        self.skelHldrs[self.botIdx].setObsDim(self.skelHldrs[self.botIdx].obs_dim + self.skelHldrs[self.humanIdx].obs_dim) 
        #env needs to set observation and action space variables based on currently targetted skeleton
        self.updateObsActSpaces()
        
    #sets assist force from pre-set force multipliers, and broadcasts force to all skeleton handlers  
    def initAssistForce(self):
        self.assistForce = self.getForceFromMult(self.frcMult) 
        for k,hndlr in self.skelHldrs.items():
            #reciprocal force is only set for assist robot, and only for debugging dynamics optimization
            hndlr.setDesiredExtForce(self.assistForce, self.frcMult, setReciprocal=False, obsUseMultNotFrc=self.useMultNotForce)

    #step traj forward based on either ana's height or incrementing trajectory         
    #derive target trajectory location using current ANA's height - let human height determine location of tracking ball along trajectory       
    #steps = # of steps to step trajectory if using preset trajectory step instead of ana's height 
    def stepTraj(self, steps, curFrame, frcMult, dbgTraj=False):
        done = self.doneTracking
        #either use ana's height for trajectory location or move trajectory kinematically
        if(self.useANAHeightForTrajLoc) :
            #try use next step's predicted com location (using com vel) to get next trajLoc
            trajLoc = self.skelHldrs[self.humanIdx].getRaiseProgress()
            done = self.trackTraj.setTrackedPosition(trajLoc,movVec=frcMult)
        else :
            #advance tracked location until end of trajectory, otherwise don't move
            i = 0
            while not done and i<steps:
                i+=1
                done = self.trackTraj.advTrackedPosition(movVec=frcMult)
        if dbgTraj : print('Traj is at end : {} : trajProgress : {} step : {} frame : {} '.format(done, self.trackTraj.trajStep, self.sim_steps, curFrame))
        return done
          
    #perform forward step of trajectory, forward step bot to find bot's actual force generated
    #then use this force to modify ANA's observation, which will then be used to query policy for 
    #new action.
    #BOT'S STATE IS NOT RESTORED if  restoreBotSt=False
    def frwrdStepTrajBotGetAnaAction(self, curFrame, useDet, policy, tarFrc, tarFrcMult, recip, restoreBotSt=True):
        #derive target trajectory location using current ANA's height or by evolving trajectory
        #let human height determine location of tracking ball along trajectory            
        self.doneTracking = self.stepTraj(1,curFrame, tarFrcMult, dbgTraj=False)   
        #solve for bot's actual force based on target force
        #returns bot force, bot force multiplier of ana's mg, and a dictionary of various dynamic quantities about the bot
        f_hat, fMult_hat, botResFrcDict = self.skelHldrs[self.botIdx].frwrdStepBotForFrcVal(tarFrc, tarFrcMult, recip, obsUseMultNotFrc=self.useMultNotForce, restoreBotSt=restoreBotSt, dbgFrwrdStep=False)                  
        #update ANA with force (for observation)
        self._setTargetForce(self.skelHldrs[self.humanIdx],f_hat, fMult_hat, reciprocal=False)
        #get ANA observation
        anaObs = self.skelHldrs[self.humanIdx].getObs()
        #use ANA observation in policy to get appropriate action
        action, actionStats = policy.get_action(anaObs)
        if(useDet):#deterministic policy - use mean
            action = actionStats['mean']             
        return action, botResFrcDict

    #code to run before every frame loop
    def preStepSetUp(self, a):
        #per-frame ara of tuples of frcDictionaries for bot and ana, keyed by frame
        self.frcResDictList = [] 
        #this force must get applied whenever tau is applied
        #reciprocal force is only set for assist robot, and only for debugging dynamics optimization
        self._setTargetForce(self.skelHldrs[self.humanIdx],self.assistForce, self.frcMult, reciprocal=False)
        #must call prestep on human only before frameskip loop, to record initial COM before 
        self.skelHldrs[self.humanIdx].preStep(a)  

            
    #use this to perform per-step bot opt control of force and feed it to ANA - use only when consuming policy
    #ANA and bot are not connected explicitly - bot is optimized to follow traj while generating force, 
    #ANA is connected to traj constraint.
    def stepBotForF_Hat(self, a):
        #values set externally via self.setCurrPolExpDict()
        policy = self.ANAPolicy
        useDet = not self.useRndPol
        print('Using FHat for step {}'.format(self.sim_steps))
        actionUsed = np.zeros(a.shape)
        botRecipFrc = True
        #below done in preStepSetUp
        # #per-frame ara of tuples of frcDictionaries for bot and ana, keyed by frame
        # frcResDictList = []
        
        # #must call prestep on human only before frameskip loop, to record initial COM before forward steppiong
        # #initial setting of tau is ignored since using setting it with bot's generated force every step
        # self.skelHldrs[self.humanIdx].preStep(a)    
        
        #forward simulate  - dummy res dict
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhndlr':'None', 'reason':'FINE', 'skelState':self.skelHldrs[self.humanIdx].state_vector()}
        done = False
        fr = 0
        #get target force, either from currently assigned assist force, or from querying value function
        tarFrc, tarFrcMult = self.getTargetForce(self.skelHldrs[self.humanIdx].getObs())
        #iterate every frame step, until done
        while (not done) and (fr < self.frame_skip):  
            fr +=1  
#            #get target force, either from currently assigned assist force, or from querying value function
#            tarFrc, tarFrcMult = self.getTargetForce(self.skelHldrs[self.humanIdx].getObs())
            #find new action for ANA based on helper bot's effort to generate target force
            action, botResFrcDict = self.frwrdStepTrajBotGetAnaAction(fr, useDet, policy, tarFrc, tarFrcMult, botRecipFrc, restoreBotSt=False)
            #save for dbg
            actionUsed += action
            #do not call prestep with new action in ANA, this will corrupt reward function, since initial COM used for height and height vel calcs is set in prestep before frameskip loops
            #set clamped tau in ana to action
            self.skelHldrs[self.humanIdx].tau=self.skelHldrs[self.humanIdx].setClampedTau(action) 
            #force this to be true here, since human and bot are not connected explicitly
            self.skelHldrs[self.humanIdx].setAssistFrcEveryTauApply = True 
            #send torques to human skel (and reapply assit force)
            self.skelHldrs[self.humanIdx].applyTau()

            #save current state, then set false so helper bot doesn't get re-solved/re-integrated
            oldBotMobileState = self.skelHldrs[self.botIdx].skel.is_mobile()
            #set bot imobile - already sim'ed
            self.skelHldrs[self.botIdx].setSkelMobile(False)                 
            #forward sim world
            self.dart_world.step()            
            #restore bot skel's mobility state
            self.skelHldrs[self.botIdx].setSkelMobile(oldBotMobileState)
         
            #check to see if ana broke sim each frame, if so, return with broken flag, frame # when broken, and skel causing break
            #helper bot checked in frwrd step function
            brk, chkSt, reason = self.skelHldrs[self.humanIdx].checkSimIsBroken()
            if(brk):
                done = True #don't break out until 
                resDict={'broken':True, 'frame':fr, 'skelhndlr':self.skelHldrs[self.humanIdx].name, 'reason':reason, 'skelState':chkSt}
            
            #get ANA's resultant frc dictionary if not broken
            elif self.dbgANAEefFrc :
                ANAFrcDict = self.skelHldrs[self.humanIdx].monFrcTrqPostStep(dispFrcEefRes=True, dispEefDet=self.dbgANAFrcDet)
            else :
                ANAFrcDict = {}

            self.frcResDictList.append({'ana':ANAFrcDict,'bot':botResFrcDict})
            #stop process if trajectory at end?
            if self.doneTracking :
                done=True
                    
            #evolve sim time with timestep, save how much time has elapsed since reset
            self.timeElapsed += self.dart_world.dt
            
    
        #calc avg action used over per frame over frameskip, get reward and finish step
        if(fr > 0):#if only 1 action before sim broke, don't scale action
            actionUsed /= (1.0 * fr)
            
        return self.endStepChkHuman(actionUsed, done, resDict, self.dbgANAReward)
  
    #needs to have different signature to support robot policy
    #a is list of two action sets - actions of robot, actions of human
    #a has been clipped before it reaches here to be +/- 1.0 - is this from normalized environment?
    def step(self, a): 
        #for reproducing GAE results - no assist force used
        if (self.noAssist):
            return self.stepNoAssist(a)

        #set up initial step stuff - send force, initialize frcResDictList, if used, execute prestep for ana
        self.preStepSetUp(a)
        #if using policy to update action/observation, call this instead - do not use if training human policy
        if (self.updateActionFromBotFrc==2) and not self.trainHuman:
            return self.stepBotForF_Hat(a)

        # #below done in preStepSetUp
        # #per-frame ara of tuples of frcDictionaries for bot and ana, keyed by frame
        # frcResDictList = []    
        # #this function takes a single vector, either force + torque or just force
        # #this force must get applied whenever tau is applied
        # #reciprocal force is only set for assist robot, and only for debugging dynamics optimization
        # self._setTargetForce(self.skelHldrs[self.humanIdx],self.assistForce, self.frcMult, reciprocal=False)
        # #must call prestep on human only before frameskip loop, to record initial COM before 
        # self.skelHldrs[self.humanIdx].preStep(a)  
        
        #forward simulate 
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhndlr':'None', 'reason':'FINE', 'skelState':self.skelHldrs[self.humanIdx].state_vector()}
        done = False
        fr = 0
        #iterate every frame step, until done
        while (not done) and (fr < self.frame_skip):   
            fr += 1
            #derive target trajectory location using current ANA's height or by evolving trajectory
            #let human height determine location of tracking ball along trajectory            
            self.doneTracking = self.stepTraj(1,fr, self.frcMult,False)#dbgTraj=(fr==1))
                
            #init bot and solve for IK/ID
            if(self.solvingBot): 
                self.botPreStep(self.assistForce, self.frcMult, recip=True)
            
            #apply all torques to skeletons
            for _,v in self.skelHldrs.items():
                #tau is set in calling prestep function - needs to be sent to skeleton every sim step
                v.applyTau()              

            self.dart_world.step()
         
            #check to see if sim is broken each frame, for any skel handler, if so, return with broken flag, frame # when broken, and skel causing break
            chk,resDictTmp = self.checkWorldStep(fr)
            if(chk):
                done = True #don't break out until bot is processed
                resDict=resDictTmp

            self.perFrameStepCheck(resDict, self.solvingBot, self.dbgANAEefFrc)
            #stop process if trajectory at end? might want to continue
#            if self.doneTracking :
#                done=True
            
            #evolve sim time with timestep, save how much time has elapsed since reset
            self.timeElapsed += self.dart_world.dt  
        #get reward and finish step
        return self.endStepChkHuman(a,  done, resDict, self.dbgANAReward)

    #simple step for getup without assistance
    #just take a and apply to skeleton
    def stepNoAssist(self, a):
        #print('StepNoAssist a on entry : {}'.format(a))
        self.skelHldrs[self.humanIdx].preStep(a)  
        #no assist force
        self.skelHldrs[self.humanIdx].setAssistFrcEveryTauApply = False
        
        #forward simulate 
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhndlr':'None', 'reason':'FINE', 'skelState':self.skelHldrs[self.humanIdx].state_vector()}
        done = False
        fr = 0
        #iterate every frame step, until done
        while (not done) and (fr < self.frame_skip):   
            fr += 1
            
            #apply torques to ana skel
            self.skelHldrs[self.humanIdx].applyTau()              

            self.dart_world.step()
         
            #check to see if sim is broken each frame, for any skel handler, if so, return with broken flag, frame # when broken, and skel causing break
            brk, chkSt, reason = self.skelHldrs[self.humanIdx].checkSimIsBroken()
            if(brk):
                done = True #don't break out until 
                resDict={'broken':True, 'frame':fr, 'skelhndlr':self.skelHldrs[self.humanIdx].name, 'reason':reason, 'skelState':chkSt}

            #evolve sim time with timestep, save how much time has elapsed since reset
            self.timeElapsed += self.dart_world.dt  

        return self.endStepChkHuman(a,  done, resDict, self.dbgANAReward)
    
    #handle post-step robot assistant performance - return values only relevant if calculating a reward
    #debugRwd : whether to save debug reward data in dictionary dbgDict
    #dbgEefFrc : whether to query dynamic data to determine force at end effector and display results to console
    #build ana's frc result dictionary; build bot's dictionary before calling bot's post step
    def perFrameStepCheck(self, resDict, dbgBot, dbgAna):
        fDict={}
        if(dbgBot): 
            fDict['bot'] = self.skelHldrs[self.botIdx].monFrcTrqPostStep(dispFrcEefRes=True,dispEefDet=True)
            #poststep for bot handles if broken, 
            #returns obBot,rwdBot,doneBot, bot_DbgDict
            _,_,_,_= self.skelHldrs[self.botIdx].postStep(resDict) 
        else:
            fDict['bot'] = {}

        if(dbgAna):
            fDict['ana'] = self.skelHldrs[self.humanIdx].monFrcTrqPostStep(dispFrcEefRes=True, dispEefDet=self.dbgANAFrcDet) 
        else:
            fDict['ana'] = {}                
        self.frcResDictList.append(fDict)   

    #after all frames executed, call ANA's reward function and return performance, observation, whether or not done,and informational and dbg dictionaries
    #also preserve ana's state if 
    def endStepChkHuman(self, a, done, resDict, dbgAnaRwrd):
        #finished another step of frwrd sim
        self.sim_steps +=1
        #calculate reward
        #debugRwd : whether to save debug reward data in dictionary dbgDict and display reward data - slow, don't do if training
        #dbgEefFrc : whether to query dynamic data to determine force at end effector and display results to console
        ob, reward, ana_done, ana_dbgDict = self.skelHldrs[self.humanIdx].postStep(resDict, debugRwd=dbgAnaRwrd)   
        #dict actually defined as something in rllab code, so need to know what can be sent to it - use this to denote broken simulation so that traj is thrown out
        #TODO - fix this - breaks on some of the components in this dictionary, if every rollout does not have the same configuration
        #tensor_utils pukes if all the data in retDict is not the same, with all the same keys.  So dummy values need to be made for broken sim
        retDict = {'brokenSim':resDict['broken'], 'ana_actionUsed':a, 'simInfoDict':resDict}
        
        #need to gate these and not access them if training
        #if wanting the debug info for ANA's reward
        if dbgAnaRwrd : 
            retDict['ANA_cust_dbgDict']=ana_dbgDict

        return ob, reward, (done or ana_done), retDict  

    #will return list of most recent step's per-bot force result dictionaries - list of dicts (2-key) each key pointing to either an empty dictionary or a dict holding the force results generated by that bot
    def getPerStepEefFrcDicts(self):
        #per-frame ara of tuples of frcDictionaries for bot and ana, keyed by frame
        return self.frcResDictList 
        
    #set passed skelHandler's desired force - either target force or force generated by bot arm.  
    #bot will always used target assist force, while human will use either target or bot-generated
    #only call internall ?
    def _setTargetForce(self, skelHndlr, desFrc, desFMult, reciprocal):
        skelHndlr.setDesiredExtForce(desFrc, desFMult, setReciprocal=reciprocal, obsUseMultNotFrc=self.useMultNotForce)    
        
    #will return an optimal force prediction given an observation of Ana. 
    def getTargetForce(self, obs):
        if(self.vfOptObj is None): #no vfOptObj so just use pre-set forces
            return self.assistForce, self.frcMult
        else:
            #get target force and multiplier from vfOPT
            resDict = self.vfOptObj.findVFOptFrcForObs(obs)   
            val,rawFrc = list(resDict.items())[0]
            #clip force to be within bounds
            tarFrc = np.clip(rawFrc, self.frcBnds[0], self.frcBnds[1])
            print('standUp3d_2Bot::getTargetForce : vf pred score : {}| pred tarFrc:{} clipped tar frc : {}'.format(val, rawFrc, tarFrc))
            tarFrcMult = self.getMultFromFrc(tarFrc)
            return tarFrc, tarFrcMult  

    #return whether or not end effector force is calculated for either ANA or helper bot
    # used when consuming a policy, so the policy consumer knows whether or not the dictionary entry for EefFrcResDicts exists or not        
        
    #return whether the passed force is valid within the limits proscribed by the sim
    #also return clipped version of chkFrc kept within bounds
    def isValidForce(self, chkFrc):
        inBnds = ((self.frcBnds[0] < chkFrc).all() and (chkFrc < self.frcBnds[1]).all())
        #bndedFrc = np.clip(chkFrc, self.frcBnds[0], self.frcBnds[1])
        return inBnds#, bndedFrc
    
    #returns dictionary for human-to-ball, ball center, and robot-to-ball constraint locations
    #CALLED AFTER HUMAN POSITION RESET
    #use these to determine where to set up constraints - returns world coordinates derived from current human pose
    def getNewGrabLocs(self):
        grabLocs = {}
        #where human is touching constraint, in terms of human hand pose - x is offset from com of human reach hand, y axis is along finger length
        grabLocs['h_toCnstrnt'] = np.copy(self.skelHldrs[self.humanIdx].getHumanCnstrntLocOffsetWorld())
        #where robot is touching constraint in world coords in terms of human hand
        grabLocs['r_toCnstrnt'] = np.copy(self.skelHldrs[self.humanIdx].getHumanCnstrntLocOffsetWorld())#.reachBody.to_world(x=np.array([0,-.2319,0]))    
        #constraint center - center of spherical constraint representation is 1/2 between robot and human constrnt locs
        grabLocs['cnstrntCtr'] = (grabLocs['h_toCnstrnt'] + grabLocs['r_toCnstrnt']) * .5
        #print('\ngetNewGrabLocs : sphere ctr in world : {}|\tto local : {}\n'.format(grabLocs['cnstrntCtr'],self.skelHldrs[self.humanIdx].reachBody.to_local(grabLocs['cnstrntCtr'])))
        return grabLocs
    
    
    #called at beginning of each rollout
    def reset_model(self):
        self.dart_world.reset()
        if (not self.noAssist):
            #initialize assist force, either randomly or with force described by env consumer
            self._resetForce()
        #total sim time for rollout
        self.timeElapsed = 0
        ## of steps
        self.sim_steps = 0
        #human pose must dominate on reset - remove constraints if exist and find new human pose
        #TODO verify this works -remove constraints to rebuild them after poses have been remade!   
        #if constraining, NEED TO REMOVE CONSTRAINTS HERE SO HUMAN CAN RE-POSE WITHOUT CONSTRAINT INTERFERENCE
        
        #self.dart_world.remove_all_constraints()
        #self.constraintsBuilt = False
        
        #print('reset_model called in standUp3d_2Bot.py')
        obsList = {}
        #first reset human pose with variation
        obsList[self.humanIdx] = self.skelHldrs[self.humanIdx].reset_model() 
        if (not self.noAssist):  
            #reset models 
            self._resetEefLocsAndCnstrnts(False)
            obsList[self.botIdx] = self.skelHldrs[self.botIdx].reset_model()

        #return active model's observation
        return obsList[self.actRL_SKHndlrIDX]

    #reset trajectory location, 
    #called initially and when reset_model is called - initial poses must be set
    #   clear constraints if exist
    #   set pose of human
    #   set location of constraints in world coordinates relative to human pose
    #   set pose of robot relative to expected constraint location
    #   add constraints at locations, if constraints are expected to be made
    #   return human reset_model result as initial observation
    #set initial grab locs for both skel handlers
    #build, if necessary, constraints for each agent
    #check if constraint needs to be built and if so build it
    #setBotStart : whether or not bot needs to have starting pose set - only on initial entry into program
    def _resetEefLocsAndCnstrnts(self, setBotStart):
        #once human pose is set, including random components if appropriate, reconnect constraints
        #find positions for constraints in world coordinates for human, tracked ball and robot.  This is based on human pose. 
        grabLocs = self.getNewGrabLocs()
        #not done tracking
        self.doneTracking = False
        #initialize tracked trajectory, passing constraint initial location in world, which will also move ball - pass frc mult as relative direction to evolve trajectory
        self.trackTraj.initTraj(grabLocs['cnstrntCtr'], self.frcMult)
        #set world location for constraints for each skel holder  cPosInWorld, cBody, addBallConstraint, setConnect):           
        self.skelHldrs[self.humanIdx].initConstraintLocs(grabLocs['h_toCnstrnt'],self.grabLink.bodynodes[0])
        self.skelHldrs[self.botIdx].initConstraintLocs(grabLocs['r_toCnstrnt'],self.grabLink.bodynodes[0])
        #helper bot relies on human skel set to starting pose, so that it can IK to appropriate location - bot's eff target loc needs to be set appropriately
        #next rebuild initial pose for assistant robot, with IK to location of constraint
        if(setBotStart):
            self.skelHldrs[self.botIdx].setStartingPose() 
        #rebuild constraints if not built - if not building constraints, this does nothing
        #TODO : need to just constrain human directly to bot no intermediary object ?
        if(not self.constraintsBuilt):
            if(self.connectHuman):
                self.skelHldrs[self.humanIdx].addBallConstraint()
            if(self.connectBot):#TODO this may fail if bot is actually connected to constraint - shouldn't ever be necessary, since bot's force is being transferred manually to ana currently
                self.skelHldrs[self.botIdx].addBallConstraint()
            self.constraintsBuilt = True
        
        if(self.dbgAssistFrcData):
            print('initPoseAndResetCnstrnts after IK : sphere pos : {}|\tTarget Robot Eff Pos: {}|\tActual Robot Eff Pos: {}'.format(self.grabLink.com(),grabLocs['r_toCnstrnt'], self.skelHldrs[self.botIdx].dbg_getEffLocWorld()))

    #initialize assist force, either randomly or with force described by env consumer
    def _resetForce(self):
        if(not self.usePresetFrc):  #using random force 
            _, self.frcMult, _ = self.getRndFrcAndMults(self.frcMultBnds)

        #set assistive force vector for this rollout based on current self.frcMult, and send to all skel holders
        self.initAssistForce()
        if(self.dbgAssistFrcData):
            print('_resetForce setting : multX : {:.3f} |multY : {:.3f}|multZ : {:.3f}\nforce vec : {}'.format(self.frcMult[0],self.frcMult[1],self.frcMult[2],['{:.3f}'.format(i) for i in self.assistForce[:3]]))

        
    #returns a vector of mults and a force vector randomly generated - DOES NOT SET ANY FORCE VALUES
    #also returns value actually used in observation
    def getRndFrcAndMults(self, frcMultBnds):
        frcMult = self.np_random.uniform(low=frcMultBnds[0], high=frcMultBnds[1])
        frc = self.getForceFromMult(frcMult)
        obsVal = frcMult if self.useMultNotForce else frc
        return frc, frcMult, obsVal
    
    #set policy dictionary, holding refs to trained policy and baseline
    #and set whether to updateAction or not every step with results of bot force calc
    def setCurrPolExpDict(self, polDict, expDict, frcMult, updateAction=0, vfOptObj=None):
        self.currPolDict = polDict
        self.expDict = expDict
        self.useRndPol = expDict['useRndPol']
        self.ANAPolicy = polDict['policy']
        self.vfOptObj = vfOptObj
        #update every step - use bot's actual generated force to update ANA's assist force
        self.updateActionFromBotFrc = updateAction 
        #Set to be constant and set to frcMult - is overridden if vfOptObj is included
        self.usePresetFrc = True         
        #list of x,y,z initial assist force multipliers
        self.frcMult = frcMult
        #set assistive force vector for this rollout based on current self.frcMult, and send to all skel holders
        self.initAssistForce()
        print('standUp3d_2Bot::setCurrPolExpDict : Forcing assist force to be constant value : {}'.format(self.assistForce))
        
        
    #clear out all references to policy and experiment dictionaries and set updateActionFromBotFrc to 0, so it is ignored
    def clearCurrPolExpDict(self):
        #per-frame list of dictionaries (1 per bot) of dictionaries of frc gen results
        self.frcResDictList = [] 

        self.updateActionFromBotFrc =0
        #dictionary holding policy and baseline being used for this environment - if not none can use this to query actions
        self.currPolDict = {}
        self.expDict = {}
        self.useRndPol = None
        self.ANAPolicy = None
        self.vfOptObj = None

    ###########################
    #externally called functions
    
    #find bot force to be used for current step and restore bot state - step trajectory forward as well, and then restore trajectory
    #called externally just to get a single force value given current test frc value
    def frwrdStepTrajAndBotForFrcVal(self, tarFrc=None, tarFrcMult=None, recip=True):
        if(tarFrc is None) and (tarFrcMult is None):
            tarFrc=self.assistForce
            self.frcMult
        elif(tarFrc is None):
            tarFrc = self.getForceFromMult(tarFrcMult)
        elif (tarFrcMult is None):            
            tarFrcMult = self.getMultFromFrc(tarFrc)
        print('Using Target force {} and frc mult {} with bot'.format(self.assistForce, self.frcMult))
        #save current traj state 
        self.trackTraj.saveCurVals()
        #either use ana's height for trajectory location or move trajectory kinematically to evolve traj
        self.doneTracking = self.stepTraj(1,1, tarFrcMult, dbgTraj=True)  
        #solve for bot's actual force
        f_hat, fMult_hat, botResDict = self.skelHldrs[self.botIdx].frwrdStepBotForFrcVal(tarFrc, tarFrcMult, recip, obsUseMultNotFrc=self.useMultNotForce, restoreBotSt=True, dbgFrwrdStep=True)                  
        #restore trajectory position
        self.trackTraj.restoreSavedVals()        
        #return bot force, force mult, and bot resultant dyanmic quantities in dictionary
        return f_hat, fMult_hat, botResDict

    ####################################
    ## Externally called / possibly deprecated or outdated methods

    #also called internally
    def getForceFromMult(self, frcMult):
        return self.skelHldrs[self.humanIdx].mg * frcMult

    #also called internally
    def getMultFromFrc(self, frc):
        return frc/self.skelHldrs[self.humanIdx].mg  

    def setDebugMode(self, dbgOn):
        self.dbgAssistFrcData = dbgOn

     #configure class to use random forces on rollout reset instead of preset force value
    def unsetForceVal(self):
        self.usePresetFrc = False 

    #these should only be called during rollouts consuming value function predictions or other optimization results, either force or force mults
    #value function provides actual force given a state - NOTE! does not change state of useSetFrc flag unless chgUseSetFrc is set to true
    #setting passed useSetFrc so that if chgUseSetFrc is specified as true then useSetFrc val must be provided, but otherwise it is ignored
    def setAssistForceDuringRollout(self, frc, chgUseSetFrc, useSetFrc=None):
        #if we wish to change the state of self.usePresetFrc, check it here
        if(chgUseSetFrc):
            self.usePresetFrc = useSetFrc
        self.frcMult = self.getMultFromFrc(frc)
        self.assistForce = np.copy(frc)
        for k,hndlr in self.skelHldrs.items():
            #reciprocal force is only set for assist robot, and only for debugging dynamics optimization
            hndlr.setDesiredExtForce(self.assistForce, self.frcMult, setReciprocal=False, obsUseMultNotFrc=self.useMultNotForce)  

    def setAssistFrcMultDuringRollout(self, frcMult, chgUseSetFrc, useSetFrc=None):
        if(chgUseSetFrc):
            self.usePresetFrc = useSetFrc
        self.frcMult = np.copy(frcMult)
        self.assistForce = self.getForceFromMult(self.frcMult)
        for k,hndlr in self.skelHldrs.items():
            #reciprocal force is only set for assist robot, and only for debugging dynamics optimization
            hndlr.setDesiredExtForce(self.assistForce, self.frcMult, setReciprocal=False, obsUseMultNotFrc=self.useMultNotForce)  
            
    def getCurAssistForce(self):
        return self.assistForce, self.frcMult


    ################
    ##### old; to be removed
    # #called externally to set force multiplier - sets reset to use this force and not random force
    # def setForceMult(self, frcMult):
    #     self.usePresetFrc = True
    #     self.frcMult = np.copy(frcMult)    
            
    # #given a specific force, set the force multiplier, which is used to set force
    # #while this seems redundant, it is intended to preserve the single entry point 
    # #to force multiplier modification during scene reset
    # def setFrcMultFromFrc(self, frc):
    #     frcMult = self.getMultFromFrc(frc) 
    #     #print('frcMultx = {} | frcMulty = {}'.format(frcMultX,frcMultY))
    #     self.setForceMult(frcMult)
    
    #for external use only - return observation variable given passed state and state dots
    #obs is slightly different than pure q/qdot (includes height in world frame), requiring skel to be modified
    #restores skel pose when finished - make sure q is correctly configured
    #call to force using random force values - ignores initial foce value setting
    def getObsFromState(self, q, qdot):
        return self.activeRL_SkelHndlr.getObsFromState(q,qdot)
    
    #given passed observation vector, return :
    #  prefix : obs not including any forc components
    #  negative length of force, which is idx in observation where force begins
    #  frcCmponent : force component of observation
    def getObsComponents(self, obs):
        # number of force components
        lenFrc = len(self.frcMult)
        return obs[:-lenFrc], obs[-lenFrc:], -lenFrc

    #verifies passed value is force multiplier within specified bounds
    #returns whether legal and reason if not legal
    def isLegalFrcOrMultVal(self, frcVal, isFrcMult):
        if len(frcVal) != len(self.frcMult):#verify size
            return False, 'SIZE'
        if isFrcMult : 
            frcMult = frcVal
        else :
            frcMult = self.getMultFromFrc(frcVal)
        #make sure frcMult is within bounds
        if np.any(frcMult < self.frcBnds[0]) or np.any(frcMult > self.frcBnds[1]):
            return False, 'VAL OOB'
        return True, 'OK'

    #also called externally by testing mechanisms (? TODO: verify)
    def setToInitPose(self):
        self.activeRL_SkelHndlr.setToInitPose()

    #only called externally now, in vf optimization
    def _get_obs(self):
        return self.activeRL_SkelHndlr.getObs()

    #only called externally, in VF optimization - return the force part of the observation, either the multiplier or the force itself
    def _get_obsFrc(self):
        return self.activeRL_SkelHndlr.getObsForce()

    #build random state   
    def getRandomInitState(self, poseDel=None):       
        return self.activeRL_SkelHndlr.getRandomInitState(poseDel)
    
    #set initial state externally - call before reset_model is called by training process
    def setNewInitState(self, _qpos, _qvel):       
        self.activeRL_SkelHndlr.setNewInitState(_qpos, _qvel)
        
    """
    This method is called when the viewer is initialized and after every reset
    Optionally implement this method, if you need to tinker with camera position
    and so forth.
    """    
    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5   
    
    # #returns dictionary for human-to-ball, ball center, and robot-to-ball constraint locations
    # #CALLED AFTER HUMAN POSITION RESET
    # #use these to determine where to set up constraints - returns world coordinates derived from current human pose
    # def getNewGrabLocs(self):
    #     grabLocs = {}
    #     if (self.useOldBallLocs): #from akanksha's policy - far from constraint, not ideal, only for samsung demo
    #         #assist bot constraint world location
    #         grabLocs['r_toCnstrnt'] = np.copy(self.skelHldrs[self.humanIdx].reachBody.to_world(x=self.ballInitCnstrntLoc))#self.ballInitCnstrntLoc= np.array([ 0.12366121, -0.40942853,  0.02042496])
    #         #put bot and constraint ball at same location
    #         grabLocs['cnstrntCtr'] = np.copy(self.skelHldrs[self.humanIdx].reachBody.to_world(x=self.ballInitCnstrntLoc))#self.ballInitCnstrntLoc= np.array([ 0.12366121, -0.40942853,  0.02042496])
    #         #ANA lcl constraint loc
    #         grabLocs['h_toCnstrnt'] = np.copy(self.skelHldrs[self.humanIdx].reachBody.to_world(x=self.anaInitCnstrntLoc))
        
    #     else : 
    #         #where human is touching constraint, in terms of human hand pose - x is offset from com of human reach hand, y axis is along finger length
    #         grabLocs['h_toCnstrnt'] = np.copy(self.skelHldrs[self.humanIdx].getHumanCnstrntLocOffsetWorld())
    #         #where robot is touching constraint in world coords in terms of human hand
    #         grabLocs['r_toCnstrnt'] = np.copy(self.skelHldrs[self.humanIdx].getHumanCnstrntLocOffsetWorld())#.reachBody.to_world(x=np.array([0,-.2319,0]))    
    #         #constraint center - center of spherical constraint representation is 1/2 between robot and human constrnt locs
    #         grabLocs['cnstrntCtr'] = (grabLocs['h_toCnstrnt'] + grabLocs['r_toCnstrnt']) * .5
    #         #print('\ngetNewGrabLocs : sphere ctr in world : {}|\tto local : {}\n'.format(grabLocs['cnstrntCtr'],self.skelHldrs[self.humanIdx].reachBody.to_local(grabLocs['cnstrntCtr'])))
    #     return grabLocs

    #call this to use 128/128 policy trained with old environment and constraints with this environment - only for samsung demo
    # def setupUsingOldEnvPol(self, use=False):
    #     #[ 0.5198388  -0.14554007  0.17514266]
    #     if(use):
    #         print('\n!!!!!!!!!! Forcing env and skel handler to use old very low action scale and high skeleton KS from old environment (from 06/18) !!!!!!!!!!!!!!!!!!!!')
    #         #SETTING THIS TO CONSUME Akanksha's old 128/128 policy in new environment - scale is very low compared to current scale.
    #         self.useAK_actionScale = True
    #         #not using the locations below
    #         self.useOldBallLocs = False
    #         #target ball and assist bot constraint location, relative to ana's reach hand (local to ana's reach hand)
    #         self.ballInitCnstrntLoc= np.array([ 0, -0.3875, 0])#([ 0.12366121, -0.40942853,  0.02042496])
    #         #self.botInitCnstrntLoc = np.array([ 0.18549182, -0.53182405,  0.03063744])
    #         #ANA lcl constraint loc, relative to ana's reach hand
    #         #self.anaInitCnstrntLoc = np.array([ 0.12366121, -0.40942853,  0.02042496])#([0,-0.35375,0])#([ 0.06183061, -0.28703302,  0.01021248])
    #         self.anaInitCnstrntLoc = np.array([0, -0.35375, 0])
    #         #constraint body locations during a successful rollout TODO attempt to build a lagrange polynomial for this for a trajectory
    #         cnstrntTraj = [
    #         [0.520974,0.773155,0.173659],[0.520974,0.773155,0.173659],[0.521196,0.773519,0.176006],[0.520524,0.775421,0.180469],[0.519009,0.778986,0.183347],[0.516928,0.783199,0.184038],[0.514786,0.787458,0.182785],[0.512134,0.792097,0.179557],
    #         [0.508631,0.797866,0.175107],[0.504594,0.805015,0.169981],[0.500754,0.812884,0.164572],[0.497699,0.820554,0.158224],[0.495487,0.828134,0.150781],[0.493820,0.835995,0.143398],[0.492314,0.844283,0.137103],[0.490653,0.852936,0.132384],
    #         [0.488729,0.861700,0.129322],[0.487267,0.870897,0.126062],[0.487237,0.879035,0.122199],[0.487931,0.886328,0.118849],[0.488838,0.893502,0.116152],[0.489673,0.900900,0.114096],[0.490307,0.908627,0.112621],[0.490680,0.916747,0.111647],
    #         [0.490768,0.925315,0.111086],[0.490607,0.934307,0.110892],[0.490262,0.943701,0.111027],[0.489820,0.953469,0.111450],[0.489393,0.963583,0.112107],[0.489106,0.973988,0.112911],[0.489081,0.984684,0.113736],[0.489437,0.995670,0.114455],
    #         [0.490263,1.006951,0.114943],[0.491609,1.018536,0.115115],[0.493478,1.030433,0.114944],[0.495842,1.042653,0.114451],[0.498655,1.055187,0.113705],[0.501866,1.068004,0.112802],[0.505433,1.081058,0.111841],[0.509318,1.094279,0.110917],
    #         [0.513493,1.107601,0.110101],[0.517937,1.120955,0.109443],[0.522618,1.134298,0.108994],[0.527523,1.147552,0.108738],[0.532654,1.160681,0.108668],[0.538009,1.173669,0.108771],[0.543593,1.186465,0.109054],[0.549406,1.199042,0.109502],
    #         [0.555429,1.211381,0.110120],[0.561664,1.223463,0.110861],[0.568134,1.235273,0.111687],[0.574829,1.246785,0.112584],[0.581753,1.257980,0.113539],[0.588902,1.268843,0.114543],[0.596271,1.279386,0.115564],[0.603847,1.289600,0.116594],
    #         [0.611613,1.299481,0.117620],[0.619546,1.309022,0.118644],[0.627626,1.318252,0.119645],[0.635819,1.327145,0.120626],[0.644101,1.335686,0.121595],[0.652448,1.343861,0.122566],[0.660835,1.351674,0.123531],[0.669241,1.359112,0.124519],
    #         [0.677643,1.366180,0.125516],[0.686038,1.372908,0.126515],[0.694378,1.379282,0.127495],[0.702639,1.385301,0.128465],[0.710794,1.390957,0.129450],[0.718834,1.396250,0.130436],[0.726727,1.401193,0.131392],[0.734495,1.405798,0.132315],
    #         [0.742098,1.410054,0.133233],[0.749273,1.413913,0.134091],[0.756220,1.417466,0.135005],[0.763174,1.420826,0.135924],[0.770100,1.423988,0.136738],[0.776903,1.426896,0.137563],[0.783550,1.429550,0.138410],[0.790062,1.431970,0.139262],
    #         [0.796446,1.434175,0.140117],[0.802684,1.436179,0.140972],[0.808770,1.437997,0.141823],[0.814701,1.439639,0.142669],[0.820474,1.441119,0.143510],[0.826280,1.442596,0.144482],[0.831923,1.443990,0.145569],[0.837344,1.445276,0.146569],
    #         [0.842562,1.446471,0.147426],[0.847600,1.447610,0.148051],[0.852478,1.448719,0.148359],[0.857202,1.449801,0.148369],[0.861777,1.450853,0.148152],[0.866203,1.451868,0.147787],[0.870482,1.452838,0.147350],[0.874615,1.453755,0.146909],
    #         [0.878608,1.454605,0.146520],[0.882465,1.455381,0.146226],[0.886188,1.456075,0.146053],[0.889790,1.456687,0.146016],[0.893324,1.457251,0.146058],[0.896711,1.457725,0.146159],[0.899959,1.458109,0.146339],[0.903073,1.458410,0.146605],
    #         [0.906057,1.458636,0.146956],[0.908913,1.458795,0.147389],[0.911605,1.458886,0.147898],[0.914214,1.458935,0.148466],[0.916697,1.458941,0.149093],[0.919056,1.458912,0.149768],[0.921291,1.458856,0.150480],[0.923401,1.458780,0.151220],
    #         [0.925388,1.458690,0.151978],[0.927252,1.458590,0.152747],[0.928994,1.458486,0.153520],[0.930615,1.458380,0.154291],[0.932116,1.458276,0.155057],[0.933498,1.458176,0.155814],[0.934762,1.458082,0.156560],[0.935910,1.457994,0.157294],
    #         [0.936943,1.457915,0.158016],[0.937862,1.457843,0.158726]
    #         ]      
    #         #use followTraj.solvePolyFromCoeffs(trajCoeffs, t) to get location on trajectory, where t = [0,1]
    #         trajCoeffs = followTraj.buildEqFromPts(cnstrntTraj, deg=10)
    #         #
    #     else:
    #         self.useAK_actionScale = False
    #         self.useOldBallLocs = False