#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from gym.envs.dart import dart_env_2bot
#import gym.envs.dart.self.getDartEnvBaseDir() as glblDartEnvLoc
from followTraj import * #circleTraj,linearTraj,parabolicTraj 

import utils_Getup as util
from assistClass import Assist

from abc import ABC, abstractmethod

from collections import defaultdict


#base (abstract) environment class where two agents interact with one helping the other along some trajectory
#move all trajectory related functions to this class
class DartAssist2Bot_Env(dart_env_2bot.DartEnv2Bot, ABC):

    def __init__(self, modelLocs, fs,  dt, helperBotLoc=None, obs_type="parameter", action_type="continuous", visualize=True, disableViewer=False, screen_width=80, screen_height=45):
        #set numpy printout precision options
        np.set_printoptions(precision=5)
        np.set_printoptions(suppress=True)
        np.set_printoptions(linewidth=220)       
        dart_env_2bot.DartEnv2Bot.__init__(self, modelLocs, fs, dt=dt, obs_type=obs_type, action_type=action_type, visualize=visualize, disableViewer=disableViewer, screen_width=screen_width, screen_height=screen_height)
        #set this so we can instance a duplicate for optimization process
        self.helperBotFullPath = None
        if helperBotLoc is not None :
            self.helperBotFullPath = self.setFullPath(helperBotLoc)        

        #set to None - will be instanced later if needed - multi-variate fleishman polynomial distribution generator
        self.MVFlDistObj=None
        #will be overridden by instancing class
        self.trajResetEndLoc = np.zeros(3)

        #override these flags in environment if necessary
        #connect human to constraint 
        self.connectHuman = False
        #connect bot to constraint
        self.connectBot = False

        #solve bot dynamic tracking using IK-SPD
        self.solveBotIK_SPD = False

        #whether or not to stop when trajectory is finished
        self.stopWhenTrajDone = False
        #display debug information regarding force application - turn off if training
        self.dbgAssistData = False             
        #display ANA reward dbg data - slow, make false if training
        self.dbgANAReward = False
        #calc and display post-step ANA eef force - make false if training
        self.dbgANAEefFrc = False
        #display full ANA force results if dbgANAEefFrc is true
        self.dbgANAFrcDet = False
        #True : ANA height to determine location along trajectory; False : kinematically move traj forward based on frame in rollout 
        self.useANAHeightForTrajLoc = False
        #whether this environment uses force as an assistive component or not - default is no
        self.assistIsFrcBased = False

        ####################################
        # external/interactive force initialization
        #init and clear all policy, experiment, and vfOptObj-related quantities so that they exist but are false/clear
        self.clearCurrPolExpDict()

    #make trajectory to follow
    #args is dictionary of arguments
    #if passive trajectory is not dynamic, will just stay fixed in space   
    def trackTrajFactory(self, trajTyp, isDynamic, isPassive):
        #traj = None
        #arguments shared with all trajectories
        args={}
        args['trajSpeed'] = .3
        args['trackObj'] = self.grabLink
        args['humanSkelHldr'] = self.skelHldrs[self.humanIdx]
        args['botSkelHldr'] = self.skelHldrs[self.botIdx]
        args['env']=self
        args['dt'] = self.dart_world.dt
        args['wander']=.5
        
        #whether traj is passive or not - if passive, will compute next desired location/displacement but will not advance there
        args['isPassive'] = isPassive       #if passive trajectory is not dynamic, will just stay fixed in space    
        #whether traj should move via SPD or kinematically
        args['isDynamic'] = isDynamic or isPassive
        args['kinematicFunc'] = "setBallPosKin"
        args['dynamicFunc'] = "setBallPosSPD"        
        
        #function to determine how traj should evolve - derive coefficients for equation to determine sim-step components of trajectory
        args['trajTargetFunc'] = "setCoeffsQuartic"
        #args['trajTargetFunc'] = "setCoeffsCubic" 
        #args['trajTargetFunc'] = "setCoeffsSmoothAccel"
        #manage accelerations/forces - function to determine final velocity 
        #args['accelManageFunc'] = "setV1_Mean"
        args['accelManageFunc'] = "setV1_Zero"
        #args['accelManageFunc'] = "setV1_ToGoal"
        args['trajType'] = trajTyp.lower()

        if ('circle' in trajTyp.lower()):
            #add circular values here
            #x radius, y radius, tilt above horizontal
            args['xRad'] = .30
            args['yRad'] = .10
            args['tiltRad'] = np.pi/5.0
            return circleTraj(args)
            
        elif ('linear' in trajTyp.lower()):
            #add linear values to args here
            #TODO make able to handle end point
            #args['endPoint'] = np.array([.5,1.1,0])
            #2nd point relative to start loc of traj
            args['trajSpeed'] = .7
            #standing with reasonably arranged elbow eef loc is 0.88957,  1.37131,  0.20306
            #seated (Starting) eef pos 0.47717,  0.66361,  0.01659
            #analysis showed that mean location rel to Standing COM is .15, .22, .17 of span of reasonable eef locs
            #standing COM is  0.5102 ,  0.91379,  0.00104]
            args['rel2ndPt']=np.array([.4,.7,.2])
            #trajectory length in meters
            args['length']=.65
            return linearTraj(args)

        elif ('gauss' in trajTyp.lower()):
            #add constraint-controlled trajectory here
            #scale factor - larger wander will give greater std in move direction and magnitude sampling
            return gaussTraj(args)

        elif ('servo'in trajTyp.lower()):  #velocity-driven joint using gauss traj
            from pydart2.joint import Joint
            #set joint to be linked to world - should this be SERVO (dynamic, tries to meet vel, respects constraints) or VELOCITY (kin, forces vel, constraints are broken)
            rootJoint = self.grabLink.joint(0)
            rootJoint.set_actuator_type(Joint.SERVO)
            #add constraint-controlled trajectory here
            #scale factor - larger wander will give greater std in move direction and magnitude sampling
            #use a different dynamic update for this type of constraint - servo joint
            args['dynamicFunc'] = "setBallVel"

            return gaussTraj(args)
            
        elif ('parabola' in trajTyp.lower()):
            #add parabolic values to args here
            #TODO
            return parabolicTraj(args)   
        else:
            print('Unknown Trajectory Type : {}'.format(trajTyp))
            return None 

    #initialize all assist force and trajectory values
    def initAssistTrajVals(self, extAssistSize, useANAHeightTraj,  trajTyp, setTrajDynamic, setTrajPassive):        
        #currently using 3 dof force and 3 dof frc loc : assisting component size for observation
        self.updateAssistDim(extAssistSize)        
        #True : ANA height to determine location along trajectory; False : kinematically move traj forward based on frame in rollout 
        self.useANAHeightForTrajLoc = useANAHeightTraj
        #turn on constraint ball to simulation
        if(setTrajDynamic):#don't reset to false if not necessary
            self.grabLink.set_mobile(True)

        #build trajectory object being used to evolve motion of constraint
        self.trackTraj = self.trackTrajFactory(trajTyp, setTrajDynamic, setTrajPassive) 


        #individual environment-related components of assist initialization
        self.initAssistTraj_indiv()

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

    #individual environment-related components of assist initialization
    @abstractmethod
    def initAssistTraj_indiv(self): pass

    #set essential state flags if training human policy
    #If training human (or purely consuming trained policy without assistant involvement) : 
    #   robot should be set to not mobile, not connected to constraint ball, and not simulated/solved (although IK to ball might be good for collision info with human)
    #   human needs to get assist force applied -every timestep-
    #_botSolving is type of solving Bot should engage in : 0 is IK, 1 is constraint optimization dyn, 2 is IK-SPD 
    #_solvingBot uses IK to traj loc if not Simulated, solves ID if is simulated, if false and dynamic bot arm is ragdoll
    #helpingBot is connected to human
    def setTrainAndInitBotState(self, _train, _solvingBot, _botSolving, _SPDGain):
        #human is always mobile
        self.skelHldrs[self.humanIdx].setSkelMobile(True)
        #whether we are training or not
        self.trainHuman = _train
        #bot will solve either IK of eef if not active or ID of force gen if active
        self.solvingBot = _solvingBot
        #no assist force for this skel, so no need to set this
        self.skelHldrs[self.humanIdx].setAssistFrcEveryTauApply = False
        #baseline does not solve for IK/SPD
        self.solveBotIK_SPD = False

        if (_train):
            # #set to false to enable robot to help, otherwise, set to true if training and applying specific force to robot
            # #must be set before apply_tau unless robot is actively helping
            #set mobility - turn off mobility of bot during training
            self.skelHldrs[self.botIdx].setSkelMobile(False) 
            #display ANA reward dbg data - slow, make false if training
            self.dbgANAReward = False
            #calc and display post-step ANA eef force - make false if training
            self.dbgANAEefFrc = False
            #display full ANA force results if dbgANAEefFrc is true
            self.dbgANAFrcDet = False
            #display debug information regarding force application - turn off if training
            self.dbgAssistData = False      
            #turn off all debug displays during training          
            for _,hndlr in self.skelHldrs.items():
                hndlr.debug=False      

        else : 
            #_dynamicBot bot is fwd simulated - set mobile if true, if immobile (false) either does nothing or solves IK
            if(_botSolving == 0) :  #IK to position
                isDynamic = False
            elif(_botSolving == 1) :    #optimization torque derivation to position
                isDynamic = _solvingBot
            elif(_botSolving == 2) :    #IK to find pose; SPD to find torques, to render position/displacement
                isDynamic = _solvingBot
                #if solving bot control via IK/SPD, setup initial matricies
                self.setSolveBotIK_SPD(_SPDGain)

            self.skelHldrs[self.botIdx].setSkelMobile(isDynamic)       

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

        self.trackTraj.debug = True
        self.constraintsBuilt = False
        #set human and helper bot starting poses - these are just rough estimates of initial poses - poses will be changed every reset
        self.skelHldrs[self.humanIdx].setToInitPose()
        #set states and constraints - true means set robot init pose, so that it can IK to eef pos
        self._resetEefLocsAndCnstrnts(True)

    #call this to configure assist bot to solve IK to find desired pose and then use SPD to generate torques to meet that pose
    def setSolveBotIK_SPD(self, SPDGain):
        self.solveBotIK_SPD = True
        self.skelHldrs[self.botIdx].buildSPDMats(SPDGain)
        #bot is helping
        self.stepBotAssist = True
    
    ############################################################
    # Sim Step-related functions
    #code to run before every frame loop
    #a : action to use
    #ANA : ref to ana's skel holder
    #setTarAssist : whether or not to send assist to ANA's skel holder - this is done for assist force so it can be applied in skel holder to reach hand eef
    def preStepSetUp(self, a, ANA):
        #per-frame ara of tuples of frcDictionaries for bot and ana, keyed by frame - diagnostic
        self.assistResDictList = [] 
        #send target force to ANA - this force must get applied whenever tau is applied unless ANA is coupled directly to helper bot
        self.preStepSetUp_indiv(ANA)
        #set up trajectory for this timestep - set target location to move to for this time step
        #MOVED TO END STEP SO FUTURE DISPLACEMENT IS AVAILABLE FOR TRAINING
        #self.calcNewTrajPos() 
        #must call prestep on human only before frameskip loop, to record initial COM quantities before sim cycle
        ANA.preStep(a)  

    #set target for trajectory - this needs to not change during ana control step.
    #MUST BE CALLED ON RESET TOO
    def calcNewTrajPos(self):
        #either use ana's height for trajectory location or move trajectory independently
        if(self.useANAHeightForTrajLoc) :
            #try use next step's predicted com location (using com vel) to get next trajLoc
            trajLoc = self.skelHldrs[self.humanIdx].getRaiseProgress()
            self.trackTraj.setTrackedPosition(trajLoc)
        else :            
            self.trackTraj.setNewPosition()
        #print("----------------------- startStepTraj  useANAHeightForTrajLoc : {} -----------------------------".format(self.useANAHeightForTrajLoc))   

    #step traj forward based on precalculated target position for trajectory.  this is called 1 time per frame_skip

    #derive target trajectory location using current ANA's height - let human height determine location of tracking ball along trajectory       
    #steps = # of steps to step trajectory if using preset trajectory step instead of ana's height 
    #this is called 1 time per
    def stepTraj(self, fr):
        done = self.trackTraj.advTrackedPosition(fr)
        return done    
    
    #return the acceleration seen from moving the ball unconnected with ANA - this is just the acceleration derived from the beginning and ending velocities of the constraint ball
    def getPerStepMoveExtForce(self):
        return self.trackTraj.getPerStepMoveExtForce()

    #this should be called 1 time per frame/step corresponding with how often stepTraj is called
    def endStepTraj(self, fr):
        #update all relevant info for trajectory
        self.trackTraj.endSimStep()
        #print("-----------------------endStepTraj {} of {} ----------------------------".format(fr, self.frame_skip))

    #this will synthesize the next control step's displacement - this will be used for next observation as well as actual displacement
    def endCntrlStepTraj(self):
        #get list of debug messages from traj, if any exist
        self.setTrajDbgDisplay( self.trackTraj.getCurTrajDbgMsgs())
        #set up trajectory for next control step - set target location to move to, 
        self.calcNewTrajPos() 
        #print("-----------------------endCntrlStepTraj - new location generated for next obs and next control step ----------------------------")

    #will return an optimal assistance prediction given an observation of Ana. 
    def getTargetAssist(self, obs):
        #assistance component of passed observation
        _, origAssist, _ = self.getObsComponents( obs)
        if(self.vfOptObj is None): #no vfOptObj so just use pre-set forces at locattion of eef, if used
            #TODO : target for trajectory is location of traj ball             
            return self.getTarAssist_indiv(False, 0, None, origAssist)
        else:
            #get target force and multiplier from vfOPT
            resDict = self.vfOptObj.findVFOptAssistForObs(obs)   
            #rawAssist is all assistive components
            val,rawAssist = list(resDict.items())[0]
            return self.getTarAssist_indiv(True, val, rawAssist, origAssist)

    @abstractmethod
    def getTarAssist_indiv(self, vfExists, val, rawAssist, origObs):pass
    @abstractmethod
    def preStepSetUp_indiv(self, ANA):pass
    @abstractmethod
    def stepBotForAssist(self, a, ANA):pass
    @abstractmethod
    def botPerFramePerStep_indiv(self):pass

    #base step function
    #needs to have different signature to support robot policy
    #a is list of two action sets - actions of robot, actions of human
    #a has been clipped before it reaches here to be +/- 1.0 - is this from normalized environment?
    def step(self, a): 
        ANA = self.skelHldrs[self.humanIdx]
        #set up initial step stuff - send force, initialize assistResDictList, if used, execute prestep for ana
        self.preStepSetUp(a, ANA)
        #if using policy to update action/observation, call this instead - do not use if training human policy
        if (self.stepBotAssist) and not self.trainHuman:
            return self.stepBotForAssist(a, ANA)

        #forward simulate 
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhldr':'None', 'reason':'OK', 'skelState':ANA.state_vector()}
        done = False
        fr = 0
        #iterate every frame step, until done
        while (not done) and (fr < self.frame_skip):
            fr += 1 
            #print("----------------------- Start frame skip {} of {} -----------------------------".format(fr, self.frame_skip))   
            #derive target trajectory location using current ANA's height or by evolving trajectory
            #let human height determine location of tracking ball along trajectory    
            self.doneTracking = self.stepTraj(fr)               
            #print("----------------------- stepTraj fr {} of {} done : {} -----------------------------".format(fr, self.frame_skip, done))   
            #init bot and solve for IK/ID if appropriate
            self.botPerFramePerStep_indiv()            
            #apply all torques to skeletons
            for _,v in self.skelHldrs.items():
                #tau is set in calling prestep function - needs to be sent to skeleton every sim step
                v.applyTau()              
            self.dart_world.step()   
            #after every sim step, run this
            done, self.assistResDictList = self.perFrameStepCheck(ANA,resDict, fr, self.solvingBot, self.dbgANAEefFrc,self.dbgANAFrcDet, self.assistResDictList, stopNow=(self.doneTracking and self.stopWhenTrajDone), botDict={})   
            #print("----------------------- End frame skip {} of {} -----------------------------".format(fr, self.frame_skip))   
        #print("----------------------- End Step -----------------------------")   

        return self.endStepChkANA(a,  done, resDict, self.dbgANAReward)

    #handle post-step robot assistant performance - return values only relevant if calculating a reward
    #debugRwd : whether to save debug reward data in dictionary dbgDict
    #dbgEefFrc : whether to query dynamic data to determine force at end effector and display results to console
    #build ana's frc result dictionary; build bot's dictionary before calling bot's post step
    def perFrameStepCheck(self, ANA, resDict, fr, dbgBot, dbgAna, dbgAnaEefDet, assistResDictList, stopNow, botDict={}):
        done = False
        #if updating every frame, or if frame is 1, then update info for trajectory
        self.endStepTraj(fr)
        #fDict holds info regarding debug monitoring of torques for ANA and bot
        fDict={'ana':{}, 'bot':botDict}
        #ana is only sim that might break - check to see if sim is broken each frame, for any skel handler, if so, return with broken flag, frame # when broken, and skel causing break
        brk, chkSt, reason = ANA.checkSimIsBroken()  
        if(brk):
            done = True #don't break out until sim data collected
            resDict={'broken':True, 'frame':fr, 'skelhldr':ANA.name, 'reason':reason, 'skelState':chkSt} 
        else :
            #aggregate quantities used in reward functions for ANA every sim step - these quantities are averaged over entire control step
            ANA.aggregatePerSimStepRWDQuantities(self.dbgANAReward)
            if(dbgAna):
                fDict['ana'] = ANA.monFrcTrqPostStep(dispFrcEefRes=True, dispEefDet=dbgAnaEefDet) 

        #if botDict != {} then already performed and passed as argument
        if((dbgBot) and (botDict=={})): 
            fDict['bot'] = self.skelHldrs[self.botIdx].monFrcTrqPostStep(dispFrcEefRes=True,dispEefDet=True)
            #poststep for bot handles if broken, 
            #returns obBot,rwdBot,doneBot, bot_DbgDict
            _,_,_,_= self.skelHldrs[self.botIdx].postStep(resDict) 
        assistResDictList.append(fDict)   

        #stop process if trajectory at end, or other criteria have been met external to this function?
        if stopNow:
            done=True
        #evolve sim time with timestep, save how much time has elapsed since reset
        self.timeElapsed += self.dart_world.dt          
        #return whether or not we are done
        return done, assistResDictList   

    #after all frames executed, call ANA's reward function and return performance, observation, whether or not done,and informational and dbg dictionaries
    #also preserve ana's state if 
    def endStepChkANA(self, a, done, resDict, dbgAnaRwrd): 
        #synthesize next displacement for constraint obj for next control step, so that observation remains accurate
        self.endCntrlStepTraj()
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

        #decay screen text display
        self.popScrDispTxt(calledFrom=0)

        return ob, reward, (done or ana_done), retDict  


    ############################################################
    # Reset functions

    #called at beginning of each rollout
    def reset_model(self):
        #reset assistive component - different per environment
        self._resetAssist_Indiv()

        #human pose must dominate on reset - remove constraints if exist and find new human pose ?
        #TODO verify this works -remove constraints to rebuild them after poses have been remade!   
        #if constraining, NEED TO REMOVE CONSTRAINTS HERE SO HUMAN CAN RE-POSE WITHOUT CONSTRAINT INTERFERENCE
        
        #self.dart_world.remove_all_constraints()
        #self.constraintsBuilt = False
        
        #print('reset_model called in standUp3d_2Bot.py')
        obsList = {}
        #first reset human pose with variation
        obsList[self.humanIdx] = self.skelHldrs[self.humanIdx].reset_model() 
        #reset models 
        #sampled target eef location
        self._resetEefLocsAndCnstrnts(False)
        obsList[self.botIdx] = self.skelHldrs[self.botIdx].reset_model()

        #return active model's observation
        return obsList[self.actRL_SKHndlrIDX]
        
    @abstractmethod
    def _resetAssist_Indiv(self):pass

    #reset trajectory location, 
    #called initially and when reset_model is called - initial poses must be set
    #   clear constraints if exist
    #   set pose of human
    #   set location of constraint in world coordinates relative to human pose
    #   set pose of robot relative to expected constraint location
    #   add constraint  at locations, if constraint is expected to be made
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
        self.trackTraj.initTraj(grabLocs['cnstrntCtr'], self.trajResetEndLoc)
        #set up first trajectory position and displacement, for observation
        self.calcNewTrajPos()
        #set world location for constraints for each skel holder  cPosInWorld, cBody, addBallConstraint, setConnect):           
        self.skelHldrs[self.humanIdx].initConstraintLocs(grabLocs['h_toCnstrnt'],self.grabLink.bodynodes[0])
        #IK to this position before setting it
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
                if(not setBotStart):
                    self.skelHldrs[self.botIdx].setToInitPose() 
                self.skelHldrs[self.botIdx].addBallConstraint()
            self.constraintsBuilt = True
        
        if(self.dbgAssistData):
            print('initPoseAndResetCnstrnts after IK : sphere pos : {}|\tTarget Robot Eff Pos: {}|\tActual Robot Eff Pos: {}\n'.format(self.grabLink.com(),grabLocs['r_toCnstrnt'], self.skelHldrs[self.botIdx].dbg_getEffLocWorld()))

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

    ####################################################
    # assist functions
    #build assist object
    def loadAssistObj(self, assistFileName):
        from assistClass import Assist
        try : 
            self.assistObj = Assist.loadAssist(assistFileName, self)
            print ('Assistance object loaded from {}!!'.format(assistFileName))
        except :      
            self._buildIndivAssistObj(assistFileName)
       
            # flags={'hasForce':True, 'useMultNotForce':True, 'usePresetAssist':False , 'dbgAssistData':True }
            
            # print ('Assistance object not found at {} - Building new object'.format(assistFileName))
            # assistDict = self.buildAssistObj(initVals=np.array([0.05, 0.05, 0.0]), cmpBnds = [[0.0,0.3,-0.001],[0.2, 0.8, 0.001],[],[]], flags=flags)
            # self.assistObj = assistDict['assistObj']
            # print ('Assistance object built and saved at {}!!'.format(assistDict['objFileName']))

    #load specific assist object for environment, if doesn't exist
    def _buildIndivAssistObj(self, assistFileName):
        raise NotImplementedError()

    #build default force-based assist object
    def _buildAssistObjFrcDefault(self, flags, cmpBnds=None):
        if (cmpBnds is not None):
            obj = self._buildAssistObj(dim=3, initVals=np.array([0.1, 0.2, 0.3]), cmpFuncs = [[[0,1,2],'uniform']],cmpBnds = cmpBnds,cmpNames = ['frcMult x', 'frcMult y', 'frcMult z'],frcIdxs=[0,1,2],useMultNotForce=True, flags=flags)
        else :
            obj = self._buildAssistObj(dim=3, initVals=np.array([0.1, 0.2, 0.3]), cmpFuncs = [[[0,1,2],'uniform']],cmpBnds = [[0.0,0.3,-0.001],[0.2, 0.8, 0.001],[],[]],cmpNames = ['frcMult x', 'frcMult y', 'frcMult z'],frcIdxs=[0,1,2],useMultNotForce=True, flags=flags)
        return obj
    
    def _buildAssistObjCnstrntDefault(self, flags):
        return self._buildAssistObj(dim=3, initVals=np.array([0.01, 0.01, 0.01]), cmpFuncs = [[[0,1,2],'gauss']],cmpBnds = [[],[],[],[]],cmpNames = ['delCnstrnt x', 'delCnstrnt y', 'delCnstrnt z'],frcIdxs=[],useMultNotForce=False, flags=flags)

    #build an assist class object and save it to a file, and return file name - subsequently just call this assist object via file name
    def _buildAssistObj(self,dim, initVals, cmpFuncs, cmpBnds, cmpNames,frcIdxs,useMultNotForce, flags):
        from assistClass import Assist
        a = Assist(dim=dim, assistVals=initVals, assistFuncs=cmpFuncs, assistBnds=cmpBnds, cmpNames=cmpNames, frcIdxs=frcIdxs,useMultNotForce=useMultNotForce, flags=flags, env=self)
        #build appropriate file and directory names to save the assist
        objDir, objFileName = a.buildObjFileDirAndName()
        #save assist to disk
        assistJsonStr = a.saveAssist(objFileName)
        return {'assistObj':a,'objFileName':objFileName,'jsonStr':assistJsonStr}
        
    #set policy dictionary, holding refs to trained policy and baseline
    #and set whether to updateAction or not every step with results of bot force calc
    def setCurrPolExpDict(self, polDict, expDict, initAssistVal, vfOptObj=None):
        self.currPolDict = polDict
        self.expDict = expDict
        self.useRndPol = expDict['useRndPol']
        self.ANAPolicy = polDict['policy']
        self.vfOptObj = vfOptObj
        #instance-specific settings
        self.setCurrPolExpDict_indiv(initAssistVal)
        #set boolean to perform bot assistance
        self.stepBotAssist = True

    @abstractmethod
    def setCurrPolExpDict_indiv(self, initAssistVal):pass
        
    #clear out all references to policy and experiment dictionaries 
    def clearCurrPolExpDict(self):
        #per-frame list of dictionaries (1 per bot) of dictionaries of frc gen results
        self.assistResDictList = [] 
        self.stepBotAssist = False
        #dictionary holding policy and baseline being used for this environment - if not none can use this to query actions
        self.currPolDict = {}
        self.expDict = {}
        self.useRndPol = None
        self.ANAPolicy = None
        self.vfOptObj = None

    ###########################
    #externally called functions
    #set target assists during rollout - valDefDict will vary based on env
    @abstractmethod
    def setAssistDuringRollout(self, val, valDefDict=defaultdict(int)): pass

    #also called internally by force-based assistive envs
    def getForceFromMult(self, frcMult):
        return self.skelHldrs[self.humanIdx].mg * frcMult

    #also called internally by force-based assistive envs
    def getMultFromFrc(self, frc):
        return frc/self.skelHldrs[self.humanIdx].mg  


    #configure class to use random forces on rollout reset instead of preset force value
    def unsetUsePresetAsssitFlag(self):
        self.usePresetAssist = False 
    
    #update observation dimensions of human and robot when force type/size changes
    #extAssistSize is number of components in assistive vector
    def updateAssistDim(self,extAssistSize):
        self.extAssistSize = extAssistSize      #set to 6 to use force location too
        #adding 3 to correspond to the location of the target in world space
        self.skelHldrs[self.humanIdx].setObsDim(self.skelHldrs[self.humanIdx].obs_dim + self.extAssistSize)
        #TODO robot observation dim needs to change with pose estimator - not going to have oracle into human's state
        self.skelHldrs[self.botIdx].setObsDim(self.skelHldrs[self.botIdx].obs_dim + self.skelHldrs[self.humanIdx].obs_dim) 
        #env needs to set observation and action space variables based on currently targetted skeleton
        self.updateObsActSpaces()
        
    #return a random assist value to initialize vf Opt
    @abstractmethod
    def getRandomAssist(self, assistBnds): pass

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

    #also called externally by testing mechanisms (? TODO: verify)
    def setToInitPose(self):
        self.activeRL_skelhldr.setToInitPose()

    #only called externally now, in vf optimization
    def _get_obs(self):
        return self.activeRL_skelhldr.getObs()

    #only called externally, in VF optimization - return the force part of the observation, either the multiplier or the force itself
    def _get_obsAssist(self):
        return self.activeRL_skelhldr.getObsAssist()

    #build random state   
    def getRandomInitState(self, poseDel=None):       
        return self.activeRL_skelhldr.getRandomInitState(poseDel)
    
    #set initial state externally - call before reset_model is called by training process
    def setNewInitState(self, _qpos, _qvel):       
        self.activeRL_skelhldr.setNewInitState(_qpos, _qvel)

    #debug assist component
    def setDebugMode(self, dbgOn):
        self.dbgAssistFrcData = dbgOn

    ############################################
    # standing eef locs and eef loc dist functions - fleishman polynomial dist object code

    #build a list of reach arm joint configs and eef locs for current pose (use with standing)
    def buildJntCnfgEefLocs(self, saveToFile, numSmplsPerDof=20):
        #save points relative to ana's COM
        #will generate numSmplsPerDim**4 points
        jConfigs, eefLocs = util.sweepStandingEefLocs(self.skelHldrs[self.humanIdx], numSmplsPerDof=numSmplsPerDof)
        fileName='unk'
        fileNameMMnts='unk'
        if (saveToFile):
            fileName,fileNameMMnts = util.saveEefLocs(jConfigs, eefLocs, homeBaseDir=self.getDartEnvBaseDir())
        return jConfigs, eefLocs, fileName, fileNameMMnts

    #return file name and location for mmnts data
    def getFileNameEefLocsMMnts(self, numSmplsPerDof=20):
        # num pts == 160000
        totNumPts = numSmplsPerDof**4        
        fileNameEefLocs, fileNameMMnts = util.getEefMmntsFileNames(totNumPts,homeBaseDir=self.getDartEnvBaseDir())
        return fileNameEefLocs, fileNameMMnts

    #import and construct a fleischman distribution simualtion which will build a distribution that exhibits the same behavior as 
    #the 1st 4 moments of the passed data
    #or the 4 moments if they are passed instead (isData = False)
    def buildFlDist(self, vals, showSimRes, isData):
        distObj =  util.buildFlDist(vals, showSimRes=showSimRes, isData=isData)
        return distObj
    #mutivariate
    def buildMVFlDist(self, vals, showSimRes, isData):
        distObj = util.buildMVFlDist(vals, showSimRes=showSimRes, isData=isData)
        return distObj

    #build a fleishman distribution of possible standing eef locations
    #use distribution to sample an end effector loc
    def sampleGoalEefRelLoc(self, fileNameMMnts=None, showSimRes=False, doReport=False, dbgMVSampling=False, useHull=True):
        #if doesn't exist then build this
        
        if not(hasattr(self, 'MVFlDistObj') and self.MVFlDistObj is not None):
            if (fileNameMMnts is None):
                fileNameMMnts = util.goalEefLocRelComMmntsFile
            mmntsDict = util.loadEefMmntsForMVFlDist(baseFileDir=self.getDartEnvBaseDir(),fileNameMMnts=fileNameMMnts)            
            self.MVFlDistObj = self.buildMVFlDist(mmntsDict, showSimRes=showSimRes, isData=False)
        dataRes = self.MVFlDistObj.genMVData(N=1, doReport=doReport, debug=dbgMVSampling, useHull=useHull)
        #add standing on ground COM location
        #dataResFromCom = dataRes[0] + self.skelHldrs[self.humanIdx].standOnGndCOMLoc
        #print('InitSample : {} | dataRes from COM {} | COM : {}'.format(dataRes[0], dataResFromCom,self.skelHldrs[self.humanIdx].standOnGndCOMLoc))
        #value is relative to standing COM - need to add stand COM to get value
        return dataRes[0]

    #global location of dart env base dir
    def getDartEnvBaseDir(self):
        return dart_env_2bot.glblDartEnvLoc


    #call if wanting to change existing reward components after skeleton holders are built.
    def setDesRwdComps(self, rwdList):
        #set static variable.  UGH.
        dart_env_2bot.DartEnv2Bot.getRwdsToUseBin(rwdList)
        #reset ANA's reward functions used
        self.activeRL_skelhldr.setRwdsToUse(dart_env_2bot.DartEnv2Bot.rwdCompsUsed)


    #build a new assist object from a json file
    def getAssistObj(self, fileName, env):
        return Assist.loadAssist(fileName, env)
        
    """
    This method is called when the viewer is initialized and after every reset
    Optionally implement this method, if you need to tinker with camera position
    and so forth.
    """    
    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5   

#object to hold screen display text
class scrDispText(object):
    def __init__(self, txt):
        self.text = txt
    
    def drawMe(self, ri, curXLoc, curYLoc):
        ri.draw_text([curXLoc, curYLoc], self.text)



#bodies in kima 
#[[BodyNode(0): pelvis_aux],
# [BodyNode(1): pelvis],
# [BodyNode(2): l-upperleg],
# [BodyNode(3): l-lowerleg],
# [BodyNode(4): l-foot],
# [BodyNode(5): r-upperleg],
# [BodyNode(6): r-lowerleg],
# [BodyNode(7): r-foot],
# [BodyNode(8): abdomen],
# [BodyNode(9): thorax],
# [BodyNode(10): head],
# [BodyNode(11): l-clavicle],
# [BodyNode(12): l-upperarm],
# [BodyNode(13): l-lowerarm],
# [BodyNode(14): r-clavicle],
# [BodyNode(15): r-upperarm],
# [BodyNode(16): r-lowerarm],
# [BodyNode(17): r-attachment]]

    
#root orientation 0,1,2
#root location 3,4,5

#left upper leg 6,7,8
#left shin, left heel(2) 9,10,11
#right thigh 12,13,14
#right shin, right heel(2) 15,16,17
#abdoment(2), spine 18, 19, 20
#bicep left(3) 21,22,23
#forearm left  24
#bicep right (3) 25, 26, 27
#forearm right 28    
        
#joints in kima -OLD KIMA
#[[TranslationalJoint(0): j_pelvis],
# [EulerJoint(1): j_pelvis2],
# [EulerJoint(2): j_thigh_left],
# [RevoluteJoint(3): j_shin_left],
# [UniversalJoint(4): j_heel_left],
# [EulerJoint(5): j_thigh_right],
# [RevoluteJoint(6): j_shin_right],
# [UniversalJoint(7): j_heel_right],
# [UniversalJoint(8): j_abdomen],
# [RevoluteJoint(9): j_spine],
# [WeldJoint(10): j_head],
# [WeldJoint(11): j_scapula_left],
# [EulerJoint(12): j_bicep_left],
# [RevoluteJoint(13): j_forearm_left],
# [WeldJoint(14): j_scapula_right],
# [EulerJoint(15): j_bicep_right],
# [RevoluteJoint(16): j_forearm_right],
# ---[WeldJoint(17): j_forearm_right_attach]]
        
#dofs in kima -OLD KIMA
#[[Dof(0): j_pelvis_x],
# [Dof(1): j_pelvis_y],
# [Dof(2): j_pelvis_z],
# [Dof(3): j_pelvis2_z],
# [Dof(4): j_pelvis2_y],
# [Dof(5): j_pelvis2_x],
# [Dof(6): j_thigh_left_z],             (0)
# [Dof(7): j_thigh_left_y],             (1)
# [Dof(8): j_thigh_left_x],             (2)
# [Dof(9): j_shin_left],                (3)
# [Dof(10): j_heel_left_1],             (4)
# [Dof(11): j_heel_left_2],             (5)
# [Dof(12): j_thigh_right_z],           (6)
# [Dof(13): j_thigh_right_y],           (7)
# [Dof(14): j_thigh_right_x],           (8)
# [Dof(15): j_shin_right],              (9)
# [Dof(16): j_heel_right_1],            (10)
# [Dof(17): j_heel_right_2],            (11)
# [Dof(18): j_abdomen_1],               (12)
# [Dof(19): j_abdomen_2],               (13)
# [Dof(20): j_spine],                   (14)
# [Dof(21): j_bicep_left_z],            (15)
# [Dof(22): j_bicep_left_y],            (16)
# [Dof(23): j_bicep_left_x],            (17)
# [Dof(24): j_forearm_left],            (18)
# [Dof(25): j_bicep_right_z],           (19)
# [Dof(26): j_bicep_right_y],           (20)
# [Dof(27): j_bicep_right_x],           (21)
# [Dof(28): j_forearm_right]]           (22)



#SKEL info for biped
#root orientation 0,1,2
#root location 3,4,5
#left thigh 6,7,8
#left shin, left heel(2), left toe 9,10,11,12
#right thigh ,13,14,15
#right shin, right heel(2), right toe 16,17,18,19
#abdoment(2), spine 20,21,22
#head 23,24
#scap left, bicep left(3) 25,26,27,28
#forearm left, hand left 29,30
#scap right, bicep right (3) 31,32,33,34
#forearm right.,hand right 35,36    
        #joints in biped
#[FreeJoint(0): j_pelvis]
        
#[EulerJoint(1): j_thigh_left]
#[RevoluteJoint(2): j_shin_left]
#[UniversalJoint(3): j_heel_left]
#[RevoluteJoint(4): j_toe_left]
        
#[EulerJoint(5): j_thigh_right]
#[RevoluteJoint(6): j_shin_right]
#[UniversalJoint(7): j_heel_right]
#[RevoluteJoint(8): j_toe_right]
        
#[UniversalJoint(9): j_abdomen]
#[RevoluteJoint(10): j_spine]
#[UniversalJoint(11): j_head]
        
#[RevoluteJoint(12): j_scapula_left]
#[EulerJoint(13): j_bicep_left]
#[RevoluteJoint(14): j_forearm_left]
#[RevoluteJoint(15): j_hand_left]
        
#[RevoluteJoint(16): j_scapula_right]
#[EulerJoint(17): j_bicep_right]
#[RevoluteJoint(18): j_forearm_right]
#[RevoluteJoint(19): j_hand_right]
        #dofs in biped
#[Dof(0): j_pelvis_rot_x]
#[Dof(1): j_pelvis_rot_y]
#[Dof(2): j_pelvis_rot_z]
#[Dof(3): j_pelvis_pos_x]
#[Dof(4): j_pelvis_pos_y]
#[Dof(5): j_pelvis_pos_z]
#[Dof(6): j_thigh_left_z]
#[Dof(7): j_thigh_left_y]
#[Dof(8): j_thigh_left_x]
#[Dof(9): j_shin_left]
#[Dof(10): j_heel_left_1]
#[Dof(11): j_heel_left_2]
#[Dof(12): j_toe_left]
#[Dof(13): j_thigh_right_z]
#[Dof(14): j_thigh_right_y]
#[Dof(15): j_thigh_right_x]
#[Dof(16): j_shin_right]
#[Dof(17): j_heel_right_1]
#[Dof(18): j_heel_right_2]
#[Dof(19): j_toe_right]
#[Dof(20): j_abdomen_1]
#[Dof(21): j_abdomen_2]
#[Dof(22): j_spine]
#[Dof(23): j_head_1]
#[Dof(24): j_head_2]
#[Dof(25): j_scapula_left]
#[Dof(26): j_bicep_left_z]
#[Dof(27): j_bicep_left_y]
#[Dof(28): j_bicep_left_x]
#[Dof(29): j_forearm_left]
#[Dof(30): j_hand_left]
#[Dof(31): j_scapula_right]
#[Dof(32): j_bicep_right_z]
#[Dof(33): j_bicep_right_y]
#[Dof(34): j_bicep_right_x]
#[Dof(35): j_forearm_right]
#[Dof(36): j_hand_right]