import numpy as np
#import pydart2 as pydart
from gym import utils
from gym.envs.dart import dart_env_2bot
from os import path

from followTraj import followTraj
from assistClass import Assist
import utils_Getup as util

#environment where robot/robot arm helps RL policy-driven human get up using force propossal from value function approximation provided by human policy baseline
class DartStandUp3dAssistEnv(dart_env_2bot.DartEnv2Bot, utils.EzPickle):
    def __init__(self): #,args): #set args in registration listing in __init__.py
        """
        This class will manage external/interacting desired force 
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

        #set pose to not be prone - needs to be set before call to parent ctor - do this for testing
        self.setInitPoseAsGoalState = False

        #dart_env_2bot.DartEnv2Bot.__init__(self, modelLocs, 8, dt=.002, disableViewer=False)
        dart_env_2bot.DartEnv2Bot.__init__(self, modelLocs, 1, dt=.01, disableViewer=False)

        ####################################
        # external/interactive force initialization
        #init and clear all policy, experiment, and vfOptObj-related quantities so that they exist but are false/clear
        self.clearCurrPolExpDict()
        
        #initialize all force and trajectory values: extAssistSize=6 will add force application location target to observation
        self.initAssistTrajVals(extAssistSize=3)
        
        #connect human to constraint 
        self.connectHuman = True
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
        #self.trainPolicy is defined as static variable in dart_env_2bot TODO make parent class method that enables this to be trained -after- object is instanced
        #_solvingBot : whether or not helper bot's motion is solved
        #_dynamicBot : true solves dynamics, false solves IK
        #removed : #_helpingBot : whether or not helper bot is actually coupled to human(i.e. helping by directly applying force) 
        self.setTrainAndInitBotState(self.trainPolicy, _solvingBot=False, _dynamicBot=False, trajTyp='linear')                                
        utils.EzPickle.__init__(self)

    #initialize all assist force and trajectory values
    def initAssistTrajVals(self, extAssistSize):        
        #currently using 3 dof force and 3 dof frc loc : assisting component size for observation
        self.extAssistSize = extAssistSize      #set to 6 to use force location too
        self.updateFrcType()
        
        #whether or not the trajectory should be followed
        self.followTraj = True
        #True : ANA height to determine location along trajectory; False : kinematically move traj forward based on frame in rollout 
        self.useANAHeightForTrajLoc = True

        ###################################
        # assistive component used in this environment dim, _initVals, _cmpFuncs, _cmpBnds, _cmpNames, _frcIdxs, _env, _rndGen)
        #self.assist = Assist(dim, assistVals, assistFuncs, assistBnds, cmpNames, frcIdxs, env, useMultNotForce=True, flags=None)
        #flagIds = ['hasForce', 'useMultNotForce', 'usePresetAssist' , 'dbgAssistData']
        self.loadAssistObj(util.assistFrcFileName)
        # try : 
        #     self.assist = Assist.loadAssist(util.assistFrcFileName, self)
        #     print ('Assistance object loaded from {}!!'.format(util.assistFrcFileName))
        # except :             
        #     self._loadIndivAssistObj(util.assistFrcFileName)
            
        #whether to train on force multiplier (if false train on actual force, which has been status quo so far - this may lead to instabilities since input magnitude is much higher than other observation components?
        self.useMultNotForce = True
        #whether to always use specific force set here in ctor or to randomly regenerate force 
        self.usePresetFrc = True         
        #list of x,y,z initial assist force multipliers
        self.frcMult = np.array([0.01, 0.01, 0.0])
        print('INIT ASSIST FORCE MULTIPLIER BEING SET TO : {}'.format(self.frcMult))
        if(self.usePresetFrc) :
            print('!!!!!!!!!!!!!!!!!!!!!!!! Warning : DartStandUp3dAssistEnv:ctor using hard coded force multiplier {} for all rollouts !!!!!!!!!!!!!!'.format(self.frcMult))        
        #bounds of force mult to be used in random force generation
        self.frcMultBnds = np.array([[0.0,0.3,-0.001],[0.2, 0.8, 0.001]])
        self.frcBnds = self.getForceFromMult(self.frcMultBnds)   

    #handl 
    def _buildIndivAssistObj(self, assistFileName):
        print ('Force Assistance object not found at {} - Building new object'.format(assistFileName))
        flags={'hasForce':True, 'useMultNotForce':True, 'usePresetAssist':False , 'dbgAssistData':True }        
        assistDict = self._buildAssistObjFrcDefault(flags)
        self.assistObj = assistDict['assistObj']
        print ('Force Assistance object built and saved at {}!!'.format(assistDict['objFileName']))


    #return assistive component of ANA's observation - put here so can be easily modified
    def getSkelAssistObs(self, skelHldr):
        #frc component
        frcObs = skelHldr.getObsAssist()
        #frc application target on traj element, if used
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
        self.skelHldrs[self.humanIdx].setToInitPose()
        #set states and constraints - true means set robot init pose, so that it can IK to eef pos
        self._resetEefLocsAndCnstrnts(True)

        
    #call this for bot prestep
    def botPreStep(self, frc, frcMult, frcNorm, recip=True):
        #set sphere forces to counteract gravity, to hold still in space if bot is applying appropriate force
        #self.grabLink.bodynodes[0].set_ext_force(self.sphereForce)        
        #set the desired force the robot wants to generate 
        self._setTargetAssist(self.skelHldrs[self.botIdx],frc,frcMult, frcNorm, reciprocal=recip)
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
    def initAssistForce(self, frcMult):
        self._buildFrcCnstrcts(self.getForceFromMult(frcMult),frcMult)
        for k,hndlr in self.skelHldrs.items(): 
            self._setTargetAssist(hndlr, self.assistForce, self.frcMult, self.frcNormalized, reciprocal=False)
            #reciprocal force is only set for assist robot, and only for debugging dynamics optimization
            #hndlr.setDesiredExtAssist(self.assistForce, self.frcMult, setReciprocal=False, obsUseMultNotFrc=self.useMultNotForce)

    #step traj forward based on either ana's height or incrementing trajectory         
    #derive target trajectory location using current ANA's height - let human height determine location of tracking ball along trajectory       
    #steps = # of steps to step trajectory if using preset trajectory step instead of ana's height 
    def stepTraj(self, steps, curFrame, frcMult, dbgTraj=False):
        done = self.doneTracking
        #self.frcNormalized
        moveVec = self.frcNormalized * .5
        #either use ana's height for trajectory location or move trajectory kinematically
        if(self.useANAHeightForTrajLoc) :
            #try use next step's predicted com location (using com vel) to get next trajLoc
            trajLoc = self.skelHldrs[self.humanIdx].getRaiseProgress()
            done = self.trackTraj.setTrackedPosition(trajLoc,movVec=moveVec)
        else :
            #advance tracked location until end of trajectory, otherwise don't move
            i = 0
            while not done and i<steps:
                i+=1
                done = self.trackTraj.advTrackedPosition(movVec=moveVec)
        if dbgTraj : print('Traj is at end : {} : trajProgress : {} step : {} frame : {} '.format(done, self.trackTraj.trajStep, self.sim_steps, curFrame))
        return done
          
    #perform forward step of trajectory, forward step bot to find bot's actual force generated
    #then use this force to modify ANA's observation, which will then be used to query policy for 
    #new action.
    #BOT'S STATE IS NOT RESTORED if  restoreBotSt=False
    def frwrdStepTrajBotGetAnaAction(self, curFrame, useDet, policy, tarFrc, tarFrcMult, recip, restoreBotSt=True):
        ANA = self.skelHldrs[self.humanIdx]
        #derive target trajectory location using current ANA's height or by evolving trajectory
        #let human height determine location of tracking ball along trajectory    
        #TODO need to use location derived from VFopt, if exists        
        self.doneTracking = self.stepTraj(1,curFrame, tarFrcMult, dbgTraj=False)   
        #solve for bot's actual force based on target force
        #returns bot force, bot force multiplier of ana's mg, and a dictionary of various dynamic quantities about the bot
        f_hat, fMult_hat, botResFrcDict = self.skelHldrs[self.botIdx].frwrdStepBotForFrcVal(tarFrc, tarFrcMult, recip, obsUseMultNotFrc=self.useMultNotForce, restoreBotSt=restoreBotSt, dbgFrwrdStep=False)                  
        #update ANA with force (for observation)
        f_norm = np.linalg.norm(fMult_hat)
        self._setTargetAssist(ANA,f_hat, fMult_hat,f_norm, reciprocal=False)
        #get ANA observation
        anaObs = ANA.getObs()
        #use ANA observation in policy to get appropriate action
        action, actionStats = policy.get_action(anaObs)
        if(useDet):#deterministic policy - use mean
            action = actionStats['mean']             
        return action, botResFrcDict

    #code to run before every frame loop
    def preStepSetUp(self, a, ANA):
        #per-frame ara of tuples of frcDictionaries for bot and ana, keyed by frame - diagnostic
        self.frcResDictList = [] 
        #send target force to ANA - this force must get applied whenever tau is applied unless ANA is coupled directly to helper bot
        self._setTargetAssist(ANA,self.assistForce, self.frcMult, self.frcNormalized, reciprocal=False)
        #must call prestep on human only before frameskip loop, to record initial COM quantities before sim cycle
        ANA.preStep(a)  
            
    #use this to perform per-step bot opt control of force and feed it to ANA - use only when consuming policy
    #ANA and bot are not connected explicitly - bot is optimized to follow traj while generating force, 
    #ANA is connected to traj constraint.
    def stepBotForF_Hat(self, a, ANA):
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
        # ANA.preStep(a)    
        
        #forward simulate  - dummy res dict
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhldr':'None', 'reason':'FINE', 'skelState':self.skelHldrs[self.humanIdx].state_vector()}
        done = False
        fr = 0
        #get target force, either from currently assigned assist force, or from querying value function
        tarFrc, tarFrcMult, trajTar = self.getTargetAssist(ANA.getObs())
        #iterate every frame step, until done
        while (not done) and (fr < self.frame_skip):  
            fr +=1  
#            #get target force, either from currently assigned assist force, or from querying value function
#            tarFrc, tarFrcMult, trajTar = self.getTargetAssist(ANA.getObs())
            #find new action for ANA based on helper bot's effort to generate target force
            action, botResFrcDict = self.frwrdStepTrajBotGetAnaAction(fr, useDet, policy, tarFrc, tarFrcMult, botRecipFrc, restoreBotSt=False)
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
            
#            #save current state, then set false so helper bot doesn't get re-solved/re-integrated
#            oldBotMobileState = self.skelHldrs[self.botIdx].skel.is_mobile()
#            #set bot imobile - already sim'ed
#            self.skelHldrs[self.botIdx].setSkelMobile(False)                 
#            #forward sim world
#            self.dart_world.step()            
#            #restore bot skel's mobility state
#            self.skelHldrs[self.botIdx].setSkelMobile(oldBotMobileState)
         
            #check to see if ana broke sim each frame, if so, return with broken flag, frame # when broken, and skel causing break
            #helper bot checked in frwrd step function
            brk, chkSt, reason = ANA.checkSimIsBroken()
            if(brk):
                done = True #don't break out until 
                resDict={'broken':True, 'frame':fr, 'skelhldr':ANA.name, 'reason':reason, 'skelState':chkSt}
            
            #get ANA's resultant frc dictionary if not broken
            elif self.dbgANAEefFrc :
                ANAFrcDict = ANA.monFrcTrqPostStep(dispFrcEefRes=True, dispEefDet=self.dbgANAFrcDet)
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
        ANA = self.skelHldrs[self.humanIdx]
        #set up initial step stuff - send force, initialize frcResDictList, if used, execute prestep for ana
        self.preStepSetUp(a, ANA)
        #if using policy to update action/observation, call this instead - do not use if training human policy
        if (self.updateActionFromBotFrc==2) and not self.trainHuman:
            return self.stepBotForF_Hat(a, ANA)

        #forward simulate 
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhldr':'None', 'reason':'FINE', 'skelState':ANA.state_vector()}
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
                self.botPreStep(self.assistForce, self.frcMult, self.frcNormalized, recip=True)            
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
        print('Ball Cnstrnt Frc : {}'.format(self.grabLink.constraint_forces()))
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
    #only call internally ?
    def _setTargetAssist(self, skelhldr, desFrc, desFMult, desFMultNorm, reciprocal):
        skelhldr.setDesiredExtAssist(desFrc, desFMult, desFMultNorm, setReciprocal=reciprocal, obsUseMultNotFrc=self.useMultNotForce)    
        
    #will return an optimal assistance prediction given an observation of Ana. 
    def getTargetAssist(self, obs):
        #TODO need to get this if using target location
        trajTar = np.zeros(3)
        if(self.vfOptObj is None): #no vfOptObj so just use pre-set forces at locattion of eef, if used
            #TODO : target for trajectory is location of traj ball
             
            return self.assistForce, self.frcMult,trajTar
        else:
            #get target force and multiplier from vfOPT
            resDict = self.vfOptObj.findVFOptAssistForObs(obs)   
            #rawAssist is all assistive components
            val,rawAssist = list(resDict.items())[0]
            #lenFrcComp = len(self.frcMult)
            rawFrcAssist = rawAssist[0:]
            #clip force mult to be within bounds - check if force or frc mult
            if self.useMultNotForce : 
                tarFrcMultRaw = rawFrcAssist
            else :#uses actual force in vfOpt
                tarFrcMultRaw = self.getMultFromFrc(rawFrcAssist) 

            tarFrcMult = np.clip(tarFrcMultRaw, self.frcMultBnds[0], self.frcMultBnds[1])
            tarFrc = self.getForceFromMult(tarFrcMult)
            print('standUp3d_2Bot::getTargetAssist : vf pred score : {}| pred tarFrcMult : {}| clipped tar frc : {}| clipped tar frc mult : {}'.format(val, tarFrcMultRaw, tarFrc, tarFrcMult))
            return tarFrc, tarFrcMult, trajTar

    #return whether or not end effector force is calculated for either ANA or helper bot
    # used when consuming a policy, so the policy consumer knows whether or not the dictionary entry for EefFrcResDicts exists or not        
        
    #return whether the passed values are valid for assistance within the limits proscribed by the sim
    #also return clipped version of chkAssist kept within bounds
    def isValidAssist(self, chkAssist):
        inBnds = ((self.frcBnds[0] < chkAssist).all() and (chkAssist < self.frcBnds[1]).all())
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
            self.skelHldrs[self.botIdx].setToInitPose() 
        #rebuild constraints if not built - if not building constraints, this does nothing
        #TODO : need to just constrain human directly to bot no intermediary object ?
        if(not self.constraintsBuilt):
            if(self.connectHuman):
                self.skelHldrs[self.humanIdx].addBallConstraint()
            if(self.connectBot):#TODO this may fail if bot is actually connected to constraint - shouldn't ever be necessary, since bot's force is being transferred manually to ana currently
                self.skelHldrs[self.botIdx].addBallConstraint()
            self.constraintsBuilt = True
        
        if(self.dbgAssistFrcData):
            print('initPoseAndResetCnstrnts after IK : sphere pos : {}|\tTarget Robot Eff Pos: {}|\tActual Robot Eff Pos: {}\n'.format(self.grabLink.com(),grabLocs['r_toCnstrnt'], self.skelHldrs[self.botIdx].dbg_getEffLocWorld()))

    #initialize assist force, either randomly or with force described by env consumer
    def _resetForce(self):
        if(not self.usePresetFrc):  #using random force 
            _, self.frcMult, _ = self.getRndFrcAndMults(self.frcMultBnds)

        #set assistive force vector for this rollout based on current self.frcMult, and send to all skel holders
        self.initAssistForce(self.frcMult)
        if(self.dbgAssistFrcData):
            print('_resetForce setting : multX : {:.3f} |multY : {:.3f}|multZ : {:.3f}\nforce vec : {}'.format(self.frcMult[0],self.frcMult[1],self.frcMult[2],['{:.3f}'.format(i) for i in self.assistForce[:3]]))

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
        #set assistive force vector for this rollout based on current self.frcMult, and send to all skel holders
        self.initAssistForce(frcMult)
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
            tarFrcMult = self.getMultFromFrc(tarFrc)
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

    #these should only be called during rollouts consuming value function predictions or other optimization results, either force or force mults
    #value function provides actual force given a state - NOTE! does not change state of useSetFrc flag unless chgUseSetFrc is set to true
    #setting passed useSetFrc so that if chgUseSetFrc is specified as true then useSetFrc val must be provided, but otherwise it is ignored
    def setAssistForceDuringRollout(self, frc, chgUseSetFrc, useSetFrc=None):
        #if we wish to change the state of self.usePresetFrc, check it here
        if(chgUseSetFrc):
            self.usePresetFrc = useSetFrc
        self._buildFrcCnstrcts(frc, self.getMultFromFrc(frc))
        for k,hndlr in self.skelHldrs.items():
            self._setTargetAssist(hndlr, self.assistForce, self.frcMult,self.frcNormalized, reciprocal=False)
            #reciprocal force is only set for assist robot, and only for debugging dynamics optimization
            #hndlr.setDesiredExtAssist(self.assistForce, self.frcMult, setReciprocal=False, obsUseMultNotFrc=self.useMultNotForce)  

    def setAssistFrcMultDuringRollout(self, frcMult, chgUseSetFrc, useSetFrc=None):
        if(chgUseSetFrc):
            self.usePresetFrc = useSetFrc
        self._buildFrcCnstrcts(self.getForceFromMult(frcMult),frcMult)
        for k,hndlr in self.skelHldrs.items():
            self._setTargetAssist(hndlr, self.assistForce, self.frcMult, self.frcNormalized,  reciprocal=False)
            #reciprocal force is only set for assist robot, and only for debugging dynamics optimization
            #hndlr.setDesiredExtAssist(self.assistForce, self.frcMult, setReciprocal=False, obsUseMultNotFrc=self.useMultNotForce)  
            
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
        return self.activeRL_skelhldr.getObsFromState(q,qdot)
    
    #given passed observation vector, return :
    #  prefix : obs not including any forc components
    #  negative length of force, which is idx in observation where force begins
    #  frcCmponent : force component of observation
    def getObsComponents(self, obs):
        # number of force components and other assistance components, such as location of application
        lenAssist = self.extAssistSize
        return obs[:-lenAssist], obs[-lenAssist:], -lenAssist

    #verifies passed value is force multiplier within specified bounds
    #returns whether legal and reason if not legal 
    def isLegalAssistVal(self, assistVal, isFrcMult):
        if len(assistVal) != self.extAssistSize:#verify size
            return False, 'TOTAL SIZE'
        #check force multiplier component
        numFrcVals = len(self.frcMult)
        frcComp = assistVal[0:numFrcVals]
        locComp = assistVal[numFrcVals:-1]
        if isFrcMult :             
            frcMult = frcComp
        else :
            frcMult = self.getMultFromFrc(frcComp)
        #make sure frcMult is within bounds
        if np.any(frcMult < self.frcBnds[0]) or np.any(frcMult > self.frcBnds[1]):
            return False, 'VAL OOB'
        #check location component if exists, to be within reasonable, reachable distance of ANA
        if (self.extAssistSize) > 3 :
            #locComp
            pass
        return True, 'OK'

    #also called externally by testing mechanisms (? TODO: verify)
    def setToInitPose(self):
        self.activeRL_skelhldr.setToInitPose()

    #only called externally now, in vf optimization
    def _get_obs(self):
        return self.activeRL_skelhldr.getObs()

    #only called externally, in VF optimization - return the force part of the observation, either the multiplier or the force itself
    def _get_obsFrc(self):
        return self.activeRL_skelhldr.getObsAssist()

    #build random state   
    def getRandomInitState(self, poseDel=None):       
        return self.activeRL_skelhldr.getRandomInitState(poseDel)
    
    #set initial state externally - call before reset_model is called by training process
    def setNewInitState(self, _qpos, _qvel):       
        self.activeRL_skelhldr.setNewInitState(_qpos, _qvel)
        
    #call if wanting to change existing reward components after skeleton holders are built.
    def setDesRwdComps(self, rwdList):
        #set static variable.  UGH.
        dart_env_2bot.DartEnv2Bot.getRwdsToUseBin(rwdList)
        #reset ANA's reward functions used
        self.activeRL_skelhldr.setRwdsToUse(dart_env_2bot.DartEnv2Bot.rwdCompsUsed)
        
    """
    This method is called when the viewer is initialized and after every reset
    Optionally implement this method, if you need to tinker with camera position
    and so forth.
    """    
    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5   
    
    #build a new assist object from a json file
    def getAssistObj(self, fileName, env):
        return Assist.loadAssist(fileName, env)
    
    