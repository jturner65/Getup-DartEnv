import numpy as np
#import pydart2 as pydart
from gym import utils
from gym.envs.dart import dart_env_2bot
from os import path

from followTraj import followTraj

#environment where ID robot helps RL human get up using force propossal from value function approximation provided by human policy baseline
#this environment is to duplicate GAE paper's getup process in dart
class DartStandUp3dGAE(dart_env_2bot.DartEnv2Bot, utils.EzPickle):
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
        #kr5Loc = path.join('KR5','KR5 sixx R650.urdf')      #for kr5 arm as assistant  
        #modelLocs = [kr5Loc,'getUpWithHelperBot3D_arm.skel']   #regular biped as ana
        #kima biped below with new kima skel - experienced ode issues which killed rllab, might be fixable with updating dart to 6.4.0+ using wenhao's modified lcp.cpp seems to address the issue.
        #modelLocs =  [kr5Loc,'kima/getUpWithHelperBot3D_armKima.skel']
        #kima biped below with kima skel from 2/18 - different joint limits, mapping of dofs and euler angle layout of root - MUST KEEP OLD in file name - used for both new experiments and ak's policy with constraint
        modelLocs = ['kima/getUpWithHelperBot3D_armKima_old.skel']        
        
        #set false if wanting to consume policy, true to train - this disables viewer
        #if True then using policy will yield error : AttributeError: 'NoneType' object has no attribute 'runSingleStep'
        trainPolicy = False
        print('\n!!!!!!!!!!!!!!!!! Viewer is disabled : {}\n'.format(trainPolicy))

        #!! move bot and box out of skel's way - back 2 m in skel file (in z)
        #set pose to be prone
        self.setProne = True
        #changed frameskip to 1 from 8 to match GAE
        dart_env_2bot.DartEnv2Bot.__init__(self, modelLocs, 1, dt=.01, disableViewer=trainPolicy)     
        
        #display ANA reward dbg data - slow, make false if training
        self.dbgANAReward = True
        #human is always mobile
        self.skelHldrs[self.humanIdx].setSkelMobile(True)
        #whether we are training or not
        self.trainHuman = trainPolicy
        if trainPolicy :
            #turn off all debug displays during training          
            for _,hndlr in self.skelHldrs.items():
                hndlr.debug=False   
        #set human and helper bot starting poses - these are just rough estimates of initial poses - poses will be changed every reset
        self.skelHldrs[self.humanIdx].setToInitPose()

                                
        utils.EzPickle.__init__(self)


    #specify this in instanced environment - handle initialization of ANA's skelhandler - environment-specific settings
    def _initAnaKimaSkel(self, skel, skelType, actSclBase,  basePose): 
        #used to modify kima's ks values when we used non-zero values - can be removed for ks==0
        ksVal = 0
        kdVal = 5.0
        #prone pose - override standard pose of "seated"
        basePose = self.getPoseVals('prone', skel, skelType)
        #use full action scale
        actSclIdxMult = []     
        self._fixJointVals(skel, kd=kdVal, ks=ksVal)  
        return actSclBase, actSclIdxMult, basePose

    #return assistive component of ANA's observation - put here so can be easily modified
    def getSkelAssistObs(self, skelHldr):
        return np.array([])
    #return the names of each dof of the assistive component for this environment - there are none, so string is empty
    def getSkelAssistObsNames(self, skelHldr):
        return ''
         
    #needs to have different signature to support robot policy
    #a is list of two action sets - actions of robot, actions of human
    #a has been clipped before it reaches here to be +/- 1.0 - is this from normalized environment?
    def step(self, a): 
        #print('StepNoAssist a on entry : {}'.format(a))
        self.skelHldrs[self.humanIdx].preStep(a)  
        #no assist force
        self.skelHldrs[self.humanIdx].setAssistFrcEveryTauApply = False
        
        #forward simulate 
        resDict = {'broken':False, 'frame':self.frame_skip, 'skelhldr':'None', 'reason':'FINE', 'skelState':self.skelHldrs[self.humanIdx].state_vector()}
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
                resDict={'broken':True, 'frame':fr, 'skelhldr':self.skelHldrs[self.humanIdx].name, 'reason':reason, 'skelState':chkSt}

            #evolve sim time with timestep, save how much time has elapsed since reset
            self.timeElapsed += self.dart_world.dt  

        return self.endStepChkHuman(a,  done, resDict, self.dbgANAReward)
    
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
        #total sim time for rollout
        self.timeElapsed = 0
        ## of steps
        self.sim_steps = 0
        
        #print('reset_model called in standUp3d_2Bot.py')
        obsList = {}
        #first reset human pose with variation
        obsList[self.humanIdx] = self.skelHldrs[self.humanIdx].reset_model() 

        #return active model's observation
        return obsList[self.actRL_SKHndlrIDX]
  
    
    #for external use only - return observation variable given passed state and state dots
    #obs is slightly different than pure q/qdot (includes height in world frame), requiring skel to be modified
    #restores skel pose when finished - make sure q is correctly configured
    #call to force using random force values - ignores initial foce value setting
    def getObsFromState(self, q, qdot):
        return self.activeRL_skelhldr.getObsFromState(q,qdot)
    
    #also called externally by testing mechanisms (? TODO: verify)
    def setToInitPose(self):
        self.activeRL_skelhldr.setToInitPose()

    #only called externally now, in vf optimization
    def _get_obs(self):
        return self.activeRL_skelhldr.getObs()

    #only called externally, in VF optimization - return the force part of the observation, either the multiplier or the force itself
    def _get_obsFrc(self):
        return self.activeRL_skelhldr.getObsForce()

    #build random state   
    def getRandomInitState(self, poseDel=None):       
        return self.activeRL_skelhldr.getRandomInitState(poseDel)
    
    #set initial state externally - call before reset_model is called by training process
    def setNewInitState(self, _qpos, _qvel):       
        self.activeRL_skelhldr.setNewInitState(_qpos, _qvel)
        
    """
    This method is called when the viewer is initialized and after every reset
    Optionally implement this method, if you need to tinker with camera position
    and so forth.
    """    
    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5   
