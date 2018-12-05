#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:24:52 2018

@author: akankshabindal
"""

import numpy as np
#import pydart2 as pydart
from gym import utils
from gym.envs.dart import dart_env_2bot_kima

#class for 2d standup agent, using 3 element assistive force vector- xy force + torque scalar (in x dir)
class DartKimaStandUpEnv(dart_env_2bot_kima.DartEnv2BotKima, utils.EzPickle):
    def __init__(self):
        #all skeleton init done in DartEnv2Bot
        dart_env_2bot_kima.DartEnv2BotKima.__init__(self, 'kima/kima_human_edited_2bot.skel', 8, dt=.002, disableViewer=False)

        #list of skeleton holders is self.skelHldrs            
        #idx of human is self.humanIdx, idx of helper bot is self.botIdx  
        #self.grabLink = None
        self.makeInitFBSitDownPose(self.skelHldrs[self.humanIdx])      
        if (self.hasHelperBot):
            self.makeInitStandUpPose(self.skelHldrs[self.botIdx])  
            #this will set joint constraints on sphere
            if(self.grabLink is not None) :
                self.addGrabConstraint()
        """
        !!! This class will manage force/torques always !!!
        put all relevant functionality in here, too keep it easily externally accessible
        """
        
        #display debug information regarding force application
        self.frcDebugMode = True
       
        #list of x,y assist force multipliers - init to .3, .3
        self.frcMult = np.array([.3,.3, 0])
        #minimum allowed raise velocity
        self.minUpVel = .003
        self.useSetFrc = False
        #currently using 3 dof force 
        self.updateFrcType(3)
        
        self.timeElapsed = 0
        #added by JT to save frames
        import datetime
        #save datetime for timestamping images
        self.datetimeStr = str(datetime.datetime.today())
        self.datetimeStr =  self.datetimeStr.replace('.','-').replace(' ','-').replace(':','-')        

        utils.EzPickle.__init__(self)
                        
        """
        3d figure 
        body nodes

        
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
    def returnHelperPoseMoveArm(self):
        h1, x1, y1, z1 = -0.815, 0.51, -0.14, 0.172
        h2, x2, y2, z2 = 0, 0.61, 0.607, 0.177
        
        slope_x = (x2-x1)/(h2-h1)
        slope_y = (y2-y1)/(h2-h1)
        slope_z = (z2-z1)/(h2-h1)
        
        bend_position_x = (self.skelHldrs[self.humanIdx].skel.q[1] - h1)*slope_x + x1
        bend_position_y = (self.skelHldrs[self.humanIdx].skel.q[1] - h1)*slope_y + y1
        bend_position_z = (self.skelHldrs[self.humanIdx].skel.q[1] - h1)*slope_z + z1

        initPosForHelper = np.array([bend_position_x, bend_position_y, bend_position_z])
        initPosForHelper = 2*initPosForHelper - self.skelHldrs[self.humanIdx].getReachCOM()
        q = self.skelHldrs[self.botIdx].skel.q
        q[22:24] = initPosForHelper[::-1]
        input()
        self.skelHldrs[self.botIdx].skel.set_positions(q)
        
        initPosForBody = np.array([bend_position_x, bend_position_y, bend_position_z])
        return initPosForBody
    
    def returnHelperPoseMoveWaist(self):
        #x - waist bend
        #y - forearm bend
        h1, x1, y1 = -0.815, -0.5, 0
        h2, x2, y2 = 0, 0.6, 1.5
        
        slope_x = (x2-x1)/(h2-h1)
        waist_bend_position = (self.skelHldrs[self.humanIdx].skel.q[1] - h1)*slope_x + x1
        initPosForBody = (self.skelHldrs[self.humanIdx].getReachCOM() + self.skelHldrs[self.botIdx].getReachCOM())*0.5
        
        initPosForBody[1] = waist_bend_position
        
        
        slope_y = (y2-y1)/(h2-h1)
        arm_bend_position = (self.skelHldrs[self.humanIdx].skel.q[1] - h1)*slope_y + y1
        #initPosForHelper = initPosForBody
        #initPosForHelper = 2*initPosForHelper - self.skelHldrs[self.humanIdx].getReachCOM()
        q = self.skelHldrs[self.botIdx].skel.q
        q[19] = waist_bend_position
        q[24] = arm_bend_position
        self.skelHldrs[self.botIdx].skel.set_positions(q)
        initPosForBody = (self.skelHldrs[self.humanIdx].getReachCOM() + self.skelHldrs[self.botIdx].getReachCOM())*0.5
        
        return initPosForBody
        
        
       
    def setPositionForGrabLink(self, calledBy):
        #depending on height of human move the sphere by a percentage
        # perc: current_height+1/1
        # get current_bend_position *= perc
        #-0.8 - -0.5 and 0 corrs to 0.5
        #bent down hand com
        #[0.72102723 0.04412303 0.20626841]
        #up hand com
        #[0.89865305 0.47223968 0.20668525]
        #x1,y1 = -0.815, -0.5
        #x2, y2 = 0, 0.5
        
        
        #self.skelHldrs[self.botIdx].skel.set_positions(q)
        #print(initPosForBody)
        
        #initPosForBody[1] = bend_position
        #print(self.skelHldrs[self.botIdx].skel.states()[:29])
        #input()
        
        #if bend_position>0:
            #print(initPosForBody)
            #input()
        
        #[ 0.51668771 -0.14062229  0.1720193 ] down
        #[0.61388003 0.06973473 0.17719223] up 
        initPosForBody = self.returnHelperPoseMoveWaist()
        #added because ground is down .9175
        print('[{:4f},{:4f},{:4f}],'.format(initPosForBody[0], initPosForBody[1]+.9175, initPosForBody[2]))
        spherePos = np.copy(self.grabLink.q)
        spherePos[3:6] = initPosForBody
        self.grabLink.set_positions(spherePos)  
        return spherePos
        
    def addGrabConstraint(self):

        spherePos = self.setPositionForGrabLink(calledBy='AddConstraint')
        #0.41482582  0.7649436   0.18959744
        #print('position after move : {}'.format(self.grabLink.states()[:6]))
        #print('human not connected to constraint')
        #policy only works if constraint is set to be imobile or if connected to robot
        #self.grabLink.set_mobile(False)
        self.skelHldrs[self.humanIdx].addBallConstraint(self.grabLink.bodynodes[0],spherePos[3:6])
        #print('robot not connected to constraint')
        self.skelHldrs[self.botIdx].addBallConstraint(self.grabLink.bodynodes[0],spherePos[3:6])
        
        #print(self.skelHldrs[self.botIdx].skel.body('head').com())
        #print(self.skelHldrs[self.humanIdx].skel.body('head').com())
        #print(self.skelHldrs[self.botIdx].skel.q[1])
        #input()
        #Second hand
        '''
        initPosForBody = (self.skelHldrs[self.botIdx].getReachCOM() + self.skelHldrs[self.humanIdx].skel.body('l-lowerarm').com())* .5
        spherePos = np.copy(self.grabLink.q)
        spherePos[3:6] = initPosForBody
        self.grabLink.set_positions(spherePos) 
        self.skelHldrs[self.humanIdx].addBallConstraint(self.grabLink.bodynodes[0],spherePos[3:6])
        self.skelHldrs[self.botIdx].addBallConstraint(self.grabLink.bodynodes[0],spherePos[3:6])
        '''
        
    def makeInitFBSitDownPose(self, skelHldr):
        initPose = [#sitting with knees bent - attempting to minimize initial collision with ground to minimize NAN actions
           #root location 0,1,2
            0, -0.815, 0, 
            #goes from -8 to 0
            #this is correct now we neet to bring this down
            #root orientation 3,4,5 Z Y X
            # Z is clocksise dir on the paper top to bottom
            #Y is clockwise direction coming out of paper. Guy looks in direction of arrow
            1.57, 0, 0,
            
                        
            #left thigh 6,7,8 : 6 is bend toward chest, 7 is twist along thigh axis, 8 is spread
             0.87,0, 0,
            #left shin, left heel(2) 9,10,11
            2.0, -0.5, 0,  
            #right thigh ,12,13,14: 12 is bend toward chest, 13 is twist along thigh axis, 14 is spread
            0.87,0, 0,
            #right shin, right heel(2)  15,16,17
            2.0, -0.5, 0, 
            #abdoment(2), spine 18,19, 20
            0,-1.5, 0,
            #bicep left(3) 21,22,23
            0.3,-0.6,0,
            #0.6,-1.6,0,
            #forearm left,24
            0.6,
            # bicep right (3) 25, 26, 27
            0.6,-1.6,0,
            #forearm right. 28
            0.6]
        skelHldr.setInitPose(initPose)
        #set reach hand to be right hand - name of body node
        skelHldr.setReachHand('r-lowerarm')
        
    def makeInitStandUpPose(self, skelHldr):
        initPose = np.zeros(skelHldr.skel.ndofs)
        #move to initial body position
        initPose[1] = 0
        initPose[2] = 1.2
        initPose[4] = 3.14
        #initPose[3] = 1.0
        #initPose[4] = .85
        #goes from -0.5 to 0.5
        #bend at waist
        initPose[19] = -0.5
        #stretch out left arm at shoulder to align with hand
        initPose[24] = 0
        #stretch out left hand
        initPose[22] = -1.2        
        skelHldr.setInitPose(initPose)
        #set reach hand to be left hand - name of body node
        skelHldr.setReachHand('l-lowerarm')
    
    
    #should only be called during rollouts consuming value function predictions
    #value function provides actual force given a state
    #NEED TO USE setFrcMultFromFrc when setting initial force before reset
    def setAssistForceDuringRollout(self, xfrc, yfrc, zfrc):
        self.assistForce = np.zeros(3)
        self.assistForce[0] = xfrc
        self.assistForce[1] = yfrc   
        self.assistForce[2] = zfrc   
        for hdlr in self.skelHldrs:
            hdlr.setDesiredExtForce(self.assistForce)
       
    def initXYAssistForce(self):
        self.assistForce = np.zeros(3)
        self.assistForce[0] = self.getForceFromMult(self.frcMult[0])
        self.assistForce[1] = self.getForceFromMult(self.frcMult[1])
        self.assistForce[2] = self.getForceFromMult(self.frcMult[2])
        for hdlr in self.skelHldrs:
            hdlr.setDesiredExtForce(self.assistForce)
            
       
    #also called externally by testing mechanisms
    def getForceFromMult(self, frcMult):
        return (9.8 * self.skelHldrs[self.humanIdx].skel.mass())*frcMult
    
    def getTau(self):
        return self.skelHldrs[self.humanIdx].tau           
    
    #needs to have different signature to support robot policy
    #a is list of two action sets - actions of robot, actions of human
    def _step(self, a):      
#        self.assistForce = [0,0,0]
       
        #print('assist force : {}'.format(self.assistForce))
        self.activeSkelHndlr.setDesiredExtForce(self.assistForce)
#        self.grabLink.bodynodes[0].add_ext_force([0, -self.grabLink.mass()*9.8, 0])
#          print(self._get_obs())
        self.activeSkelHndlr.preStep(a)
        
        if self.grabLink is not None:
             self.setPositionForGrabLink(calledBy='_step')
        #TODO need to set this up to accept both skeletons, which means passing array of actions from 2 different policies.  need to modify ExpLite
        #if length 2 then this is the 2-element list of robot and human actions        
        
        #TODO Robot has been set to not mobile in robot skelhandler ctor for now, until we have a trained policy for human
        resDict = self.do_simulation(self.frame_skip)
        #return velocities in all directions
#        vx, vy, vz, ob, reward, done, infoDict = self.activeSkelHndlr.postStep(resDict)  
        ob, reward, done, infoDict = self.activeSkelHndlr.postStep(resDict)  
        #infoDict actually defined as something in rllab code, so need to know what i can set it to
#        return vx, vy, vz, ob, reward, done, {}
        return ob, reward, done, infoDict

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
    
    def getMultFromForce(self, frc):
        divVal = (9.8 * self.skelHldrs[self.humanIdx].skel.mass())
        return frc/divVal
        
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
#        print('Resetting force', self.frcMult)
#        self.frcMult = [0.3, 0.3, 0]
        if(not self.useSetFrc):  #using random force       
            #varying force in direction and magnitude
            self.frcMult[0] = self.np_random.uniform(0.0, 0.5)
            self.frcMult[1] = self.np_random.uniform(0.0, 0.5)
            self.frcMult[2] = self.np_random.uniform(-0.1, 0.1)
        #set assistive force vector for this rollout
        self.initXYAssistForce()
        if(self.frcDebugMode):
            print('_resetForce setting : multX : {:.3f} |multY : {:.3f}| multZ : {:.3f}\nforce vec : {}'.format(self.frcMult[0],self.frcMult[1],self.frcMult[2], ['{:.3f}'.format(i) for i in self.assistForce[:3]]))

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
        if(self.grabLink is not None) :
            self.setPositionForGrabLink(calledBy='reset_model')
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
        print('Type', araType)
        for i in range(len(ara)):
            print('idx i={} name={}'.format(i,ara[i].name))
    
    ###########################
    #externally called functions - Added by JT to save frames for video clips

    #return a string name to use as a directory name for a simulation run
    def getRunDirName(self):
        nameStr = self.skelHldrs[self.botIdx].skel.name + '_' + self.datetimeStr 
        nameStr = nameStr.replace(' ','_')
        return nameStr
        
    #return a name for the image
    def getImgName(self):        
        return self.skelHldrs[self.botIdx].skel.name

    ####################################