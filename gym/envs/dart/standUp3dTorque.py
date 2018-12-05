import numpy as np
#import pydart2 as pydart
from gym import utils
from gym.envs.dart import dart_env_2bot

class DartStandUp3dTrqueEnv(dart_env_2bot.DartEnv2Bot, utils.EzPickle):
    def __init__(self):
        #all skeleton init done in DartEnv2Bot
        
        dart_env_2bot.DartEnv2Bot.__init__(self, 'getUpWithHelperBot3D_damp.skel', 8, dt=.002, disableViewer=False)

        #list of skeleton holders is self.skelHldrs            
        #idx of human is self.humanIdx, idx of helper bot is self.botIdx  

        self.makeInitFBSitDownPose(self.skelHldrs[self.humanIdx])       
        if (self.hasHelperBot):
            self.makeInitStandUpPose(self.skelHldrs[self.botIdx])  
            #this will set joint constraints on sphere
            if(self.grabLink is not None) :
                self.addGrabConstraint()
        #human is mobile, robot is not
        self.skelHldrs[self.humanIdx].setSkelMobile(True)
        self.skelHldrs[self.botIdx].setSkelMobile(False)

        """
        !!! This class will manage force/torques always !!!
        put all relevant functionality in here, to keep it easily externally accessible
        """
       
        #display debug information regarding force application
        self.frcDebugMode = False
       
        #list of x,y assist force multipliers - init to .3, .3
        self.frcMult = np.array([.3,.3,0])
        #minimum allowed raise velocity
        self.minUpVel = .003
        self.useSetFrc = False
        #currently using 3 dof force 
        self.updateFrcType(3)
        
        #set height target of standing human, using COM of robot at init
        self.skelHldrs[self.humanIdx].standCOMHeight = self.skelHldrs[self.botIdx].skel.com()[1]
        #print('standing bot com : {}, foot com : {}'.format(self.skelHldrs[self.botIdx].skel.com(),self.skelHldrs[self.botIdx].skel.body("h_heel_left").com()))

        
        self.timeElapsed = 0
        utils.EzPickle.__init__(self)
                        
        """
        3d figure 
        body nodes
        idx i=0 name=h_pelvis
        idx i=1 name=h_thigh_left
        idx i=2 name=h_shin_left
        idx i=3 name=h_heel_left
        idx i=4 name=h_toe_left
        idx i=5 name=h_thigh_right
        idx i=6 name=h_shin_right
        idx i=7 name=h_heel_right
        idx i=8 name=h_toe_right
        idx i=9 name=h_abdomen
        idx i=10 name=h_spine
        idx i=11 name=h_head
        idx i=12 name=h_scapula_left
        idx i=13 name=h_bicep_left
        idx i=14 name=h_forearm_left
        idx i=15 name=h_hand_left
        idx i=16 name=h_scapula_right
        idx i=17 name=h_bicep_right
        idx i=18 name=h_forearm_right
        idx i=19 name=h_hand_right
        
        """

    #update observation dimensions of human and robot when force type/size changes
    #frcTrqSize is number of components in assistive vector.
    def updateFrcType(self, frcTrqSize):
        self.skelHldrs[self.humanIdx].setObsDim(self.skelHldrs[self.humanIdx].obs_dim + frcTrqSize)
        self.skelHldrs[self.botIdx].setObsDim(self.skelHldrs[self.botIdx].obs_dim + self.skelHldrs[self.humanIdx].obs_dim) 
        #env needs to set observation and action space variables based on currently targetted skeleton
        self.updateObsActSpaces()        
    
    #add the constraints so that the robot is grabbing the human's hand
    #using 2 ball joint constraints and a single body node - 
    #this body needs to not have a collision body attached to it
    def addGrabConstraint(self):
        initPosForBody = (self.skelHldrs[self.botIdx].getReachCOM() + self.skelHldrs[self.humanIdx].getReachCOM())* .5
        spherePos = np.copy(self.grabLink.q)
        spherePos[3:6] = initPosForBody
        self.grabLink.set_positions(spherePos)  
        #0.41482582  0.7649436   0.18959744
        #print('position after move : {}'.format(self.grabLink.q))
        
        self.skelHldrs[self.humanIdx].addBallConstraint(self.grabLink.bodynodes[0],spherePos[3:6])
        self.skelHldrs[self.botIdx].addBallConstraint(self.grabLink.bodynodes[0],spherePos[3:6])
        
        
    def makeInitFBSitDownPose(self, skelHldr):
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
        skelHldr.setInitPose(initPose)
        #set reach hand to be right hand - name of body node
        skelHldr.setReachHand('h_hand_right')

        
    def makeInitStandUpPose(self, skelHldr):
        initPose = np.zeros(skelHldr.skel.ndofs)
        #move to initial body position
        initPose[1] = 3.14
        initPose[3] = 1.0
        initPose[4] = .85
        #bend at waist
        initPose[21] = -.4
        #stretch out left arm at shoulder to align with hand
        initPose[26] = .25
        #stretch out left hand
        initPose[27] = -1.2        
        skelHldr.setInitPose(initPose)
        #set reach hand to be left hand - name of body node
        skelHldr.setReachHand('h_hand_left')
    
    
    #should only be called during rollouts consuming value function predictions
    #value function provides actual force given a state
    #NEED TO USE setFrcMultFromFrc when setting initial force before reset
    def setAssistForceDuringRollout(self, xfrc, yfrc, zfrc):
        self.assistForce = np.zeros(3)
        self.assistForce[0] = xfrc
        self.assistForce[1] = yfrc   
        self.assistForce[2] = zfrc   

       
    def initXYAssistForce(self):
        self.assistForce = np.zeros(3)
        self.assistForce[0] = self.getForceFromMult(self.frcMult[0])
        self.assistForce[1] = self.getForceFromMult(self.frcMult[1])
        self.assistForce[2] = self.getForceFromMult(self.frcMult[2])   
       
    #also called externally by testing mechanisms
    def getForceFromMult(self, frcMult):
        return (9.8 * self.skelHldrs[self.humanIdx].skel.mass())*frcMult
                 
    
    #needs to have different signature to support robot policy
    #a is list of two action sets - actions of robot, actions of human
    def _step(self, a):   
        #print('standing bot com : {}, foot com : {}'.format(self.skelHldrs[self.botIdx].skel.com(),self.skelHldrs[self.botIdx].skel.body("h_heel_left").com()))
       
        #build force from com frc + torque
        #this function takes a single vector, either force + torque or just force
        self.activeSkelHndlr.setDesiredExtForce(self.assistForce)
        
        self.activeSkelHndlr.preStep(a)       
        #TODO need to set this up to accept both skeletons, which means passing array of actions from 2 different policies.  need to modify ExpLite
        #if length 2 then this is the 2-element list of robot and human actions        
        
        #TODO Robot has been set to not mobile in robot skelhandler ctor for now, until we have a trained policy for human
        resDict = self.do_simulation(self.frame_skip)
        
        ob, reward, done, infoDict = self.activeSkelHndlr.postStep(resDict)   
        #infoDict actually defined as something in rllab code, so need to know what i can set it to
        return ob, reward, done, {}

    ###########################
    #externally called functions

    def setDebugMode(self, dbgOn):
        self.frcDebugMode = dbgOn
    
    #called externally to set force multiplier
    def setForceMag(self, frcMultX, frcMultY, frcMultZ):
        self.useSetFrc = True
        self.frcMult[0] = frcMultX
        self.frcMult[1] = frcMultY
        self.frcMult[2] = frcMultZ
            
    #given a specific force, set the force multiplier, which is used to set force
    #while this seems redundant, it is intended to preserve the single entry point 
    #to force multiplier modification during scene reset
    def setFrcMultFromFrc(self, frcX, frcY, frcZ):
        divVal = (9.8 * self.skelHldrs[self.humanIdx].skel.mass())
        frcMultX = frcX/divVal
        frcMultY = frcY/divVal 
        frcMultZ = frcZ/divVal 
        
        #print('frcMultx = {} | frcMulty = {}'.format(frcMultX,frcMultY))
        self.setForceMag(frcMultX, frcMultY, frcMultZ)
        
        
    #for external use only - return observation variable given passed state and state dots
    #obs is slightly different than pure q/qdot (includes height in world frame), requiring skel to be modified
    #restores skel pose when finished - make sure q is correctly configured
    def getObsFromState(self, q, qdot):
        return self.activeSkelHndlr.getObsFromState(q,qdot)
    
    #call to force using random force values - ignores initial foce value setting
    def unsetForceVal(self):
        self.useSetFrc = False
        
    #initialize assist force, either randomly or with force described by env consumer
    def _resetForce(self):
        if(not self.useSetFrc):  #using random force       
            #varying force in direction and magnitude
            self.frcMult[0] = self.np_random.uniform(0.0, 0.5)
            self.frcMult[1] = self.np_random.uniform(0.0, 0.5)
            self.frcMult[2] = self.np_random.uniform(-0.25, 0.25)
        #set assistive force vector for this rollout
        self.initXYAssistForce()
        if(self.frcDebugMode):
            print('_resetForce setting : multX : {:.3f} |multY : {:.3f}|multZ : {:.3f}\nforce vec : {}'.format(self.frcMult[0],self.frcMult[1],self.frcMult[2],['{:.3f}'.format(i) for i in self.assistForce[:3]]))

    #also called externally by testing mechanisms (? TODO: verify)
    def setToInitPose(self):
        self.activeSkelHndlr.setToInitPose()

    #only called externally now, if at all
    def _get_obs(self):
        return self.activeSkelHndlr.getObs()

    #build random state   
    def getRandomInitState(self, poseDel=None):       
        return self.activeSkelHndlr.getRandomInitState(poseDel)
    
    #set initial state externally - call before reset_model is called by training process
    def setNewInitState(self, _qpos, _qvel):       
        self.activeSkelHndlr.setNewInitState(_qpos, _qvel)
     
    #called at beginning of each rollout
    def reset_model(self):
        self.dart_world.reset()
        #initialize assist force, either randomly or with force described by env consumer
        self._resetForce()

        obsList = []
        for i in range(len(self.skelHldrs)):
            obsList.append(self.skelHldrs[i].reset_model())
        #return active model's observation
        return obsList[self.actSKHndlrIDX]

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5
        
    def dbgCnstrntBodyDisp(self):
        spherePos = np.copy(self.grabLink.q)
        print('position before move : {}'.format(spherePos))
        spherePos[4] += .01
        self.grabLink.set_positions(spherePos)  
        print('position after move : {}'.format(self.grabLink.q))
        

           
    def dbgShowSkelList(self, ara, araType):
        print(araType)
        for i in range(len(ara)):
            print('idx i={} name={}'.format(i,ara[i].name))
