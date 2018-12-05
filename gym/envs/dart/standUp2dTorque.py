import numpy as np
#import pydart2 as pydart
from gym import utils
from gym.envs.dart import dart_env_2bot
#from pydart2.marker import Marker

#class for 2d standup agent, using 3 element assistive force vector- xy force + torque scalar (in x dir)
class DartStandUp2dTrqueEnv(dart_env_2bot.DartEnv2Bot, utils.EzPickle):
    def __init__(self):
        #all skeleton init done in DartEnv2Bot
        #dart_env_2bot.DartEnv2Bot.__init__(self, 'getUp2dWithHelperBot2.skel', 4, disableViewer=False)
        dart_env_2bot.DartEnv2Bot.__init__(self, 'getUp2dWithHelperBot3D.skel', 4, disableViewer=False)

        #list of skeleton holders is self.skelHldrs            
        #idx of human is self.humanIdx, idx of helper bot is self.botIdx        
        self.makeInitFBSitDownPose(self.skelHldrs[self.humanIdx])
        #self.makeInitLieDownPose(self.skelHldrs[self.humanIdx])
        self.makeInitStandUpPose(self.skelHldrs[self.botIdx])   
        
        
        
        
        
        #initialize assist force
        #self.initXYAssistForce()
        #print('assist force : {} mass : {}'.format(self.assistForce,self.humanAssist.mass()))
        #don't use preset value for force, use random value

        #display debug information regarding force application
        self.frcDebugMode = False
#        #use the mean state only if set to false, otherwise randomize initial state
#        self.randomizeInitState = True
#        #use these to hold preset initial states - only used if randomizeInitState is false
#        self.loadedInitStateSet = False
#        self.initQPos = None
#        self.initQVel = None
        #+/- perturb amount of intial pose and vel for random start state
        self.poseDel = .005
        
        #list of x,y assist force multipliers - init to .3, .3
        self.frcMult = np.array([.3,.3])
        #minimum allowed raise velocity
        self.minUpVel = .003
        self.useSetFrc = False
        #location of force application on pelvis body -> coincides with top edge
        frcLoc = [0,.25,0]
        self.frcOffset = np.array(frcLoc)
        self.timeElapsed = 0
        utils.EzPickle.__init__(self)
        
                
        """
        assist bot
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
        
        assist bot dofs
        
        
        humanoid 
        body nodes
        idx i=0 name=h_pelvis_aux2
        idx i=1 name=h_pelvis_aux
        idx i=2 name=h_pelvis
        idx i=3 name=h_thigh
        idx i=4 name=h_shin
        idx i=5 name=h_foot
        idx i=6 name=h_thigh_left
        idx i=7 name=h_shin_left
        idx i=8 name=h_foot_left
        dofs
        idx i=0 name=j_pelvis_x
        idx i=1 name=j_pelvis_y
        idx i=2 name=j_pelvis_rot
        idx i=3 name=j_thigh
        idx i=4 name=j_shin
        idx i=5 name=j_foot
        idx i=6 name=j_thigh_left
        idx i=7 name=j_shin_left
        idx i=8 name=j_foot_left
        # skel q size : 9
        """
    #configure dofs to be prone, with pelvis up
    def makeInitLieDownPose(self, skelHldr):
        #self.dbgDisplayState()
        initPose = np.zeros(skelHldr.skel.ndofs)
        #height should be .1
        initPose[1] = -1.0#-1.2
        #rotation needs to be pi/2
        initPose[2] = 0 #1.57
        #bend knees
        #rotate at hips
        initPose[3] = 2.44#.87
        initPose[6] = 2.44#.87
        #rotate at knees
        initPose[4] = -1.6
        initPose[7] = -1.6
        #bend ankles
        initPose[5] = -.85
        initPose[8] = -.85       
        skelHldr.setInitPose(initPose)
        
    def makeInitFBSitDownPose(self, skelHldr):
        initPose = [#sitting with knees bent - attempting to minimize initial collision with ground to minimize NAN actions
            #root orientation 0,1,2
            0,0,1.57,
            #root location 3,4,5
            0,0.05,0,            
            #left thigh 6,7,8
            0.87,0,0,
            #left shin, left heel(2), left toe 9,10,11,12
            -1.85,-0.6,0, 0,
            #right thigh ,13,14,15
            0.87,0,0,
            #right shin, right heel(2), right toe 16,17,18,19
            -1.85,-0.6,0, 0,
            #abdoment(2), spine 20,21,22
            0,-1.5,0,
            #head 23,24
            0,0,
            #scap left, bicep left(3) 25,26,27,28
            0,0,-0.5,0,
            #forearm left, hand left 29,30
            0.5,0,
            #scap right, bicep right (3) 31,32,33,34
            0,.6,1.8,0,
            #forearm right.,hand right 35,36
            0.5,0]
        skelHldr.setInitPose(initPose)
        
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
    
    
    #should only be called during rollouts consuming value function predictions
    #value function provides actual force given a state
    #NEED TO USE setFrcMultFromFrc when setting initial force before reset
    def setAssistForceDuringRollout(self, xfrc, yfrc):
        self.assistForce = np.zeros(3)
        self.assistForce[0] = xfrc
        self.assistForce[1] = yfrc   

       
    def initXYAssistForce(self):
        xfrc = self.getForceFromMult(self.frcMult[0])
        yfrc = self.getForceFromMult(self.frcMult[1])
        self.assistForce = np.zeros(3)
        self.assistForce[0] = xfrc
        self.assistForce[1] = yfrc
        
    #also called externally by testing mechanisms
    def getForceFromMult(self, frcMult):
        return (9.8 * self.skelHldrs[self.humanIdx].skel.mass())*frcMult
           
    #also called externally by testing mechanisms (? TODO: verify)
    def setToInitPose(self):
        #initPose[]
        self.activeSkelHndlr.skel.set_positions(self.initHPose)
       

    def advance(self, a):
        #TODO
        #split action into two components for human and robot
        #action is list of each skel's proposed action
        #check len - if len == 1 then only set action for human
        
        #clamps and saves torques in skelHndlr
        self.skelHldrs[self.humanIdx].setTau(a)
        #self.skelHldrs[self.botIdx].setTau(a)       
        self.do_simulation(self.frame_skip)
        
        
    
    #needs to have different signature to support robot policy
    #a is list of two action sets - actions of robot, actions of human
    def _step(self, a):
        #build force from com frc + torque
        
        #set skeleton handler for this step TODO needs to
        skelHdlr = self.skelHldrs[self.humanIdx]
#        CF = np.concatenate([self.assistTau, self.assistForce])
#        Jacobian = self.robot_skeleton.body('h_pelvis').world_jacobian()
#        T = np.matmul(Jacobian.T, CF)
#        self.robot_skeleton.set_forces(T)

        skelHdlr.skel.body('h_pelvis').add_ext_force(self.assistForce, _offset=self.frcOffset)
        linAcc = self.humanAssist.body('h_pelvis').to_world(self.humanAssist.body('h_pelvis').com_linear_acceleration())
        trans = self.humanAssist.body('h_pelvis').T
        angAcc = self.humanAssist.body('h_pelvis').com_spatial_acceleration()     
                
        print('frc : {:.5f},{:.5f} | {} | {}'.format(self.assistForce[0],self.assistForce[1],linAcc,trans)) 
               
        heightBefore =  self.humanAssist.bodynodes[2].com()[1]
        posbefore, angBefore = self.humanAssist.q[0,2]
        #for work calc
        #forceLocBefore = self.humanAssist.bodynodes[2].to_world(self.frcOffset)
        
        #TODO need to set this up to accept both skeletons, which means passing array of actions from 2 different policies.  need to modify ExpLite
        #if length 2 then this is the 2-element list of robot and human actions
        
        self.advance(a)
        #for work calc 
        #forceLocAfter = self.humanAssist.bodynodes[2].to_world(self.frcOffset)
                
        posafter,ang = self.humanAssist.q[0,2]
        skelCom = self.humanAssist.bodynodes[2].com()
        height = skelCom[1]
        contacts = self.dart_world.collision_result.contacts
        #total_force_mag = 0
        footContacts = 0
        leftContacts = 0
        rightContacts = 0
        nonFootContacts = 0
        for contact in contacts:
            #print('body 1 : {} | body 2 : {} '.format(contact.bodynode1.name, contact.bodynode2.name))
            if ('foot' in contact.bodynode2.name) or ('foot' in contact.bodynode1.name):#ground is usually body 1
                footContacts +=1
                if ('left' in contact.bodynode2.name) or ('left' in contact.bodynode1.name):
                    leftContacts +=1
                else:
                    rightContacts +=1
            else:
               nonFootContacts+=1 
            #total_force_mag += np.square(contact.force).sum()
        
        # reward function calculation
        alive_bonus = 1.0
        vel = (posafter - posbefore) / self.dt
        velScore = -abs(vel-2)+2
        raiseVel = (height - heightBefore) / self.dt
        #peak centered at velocity == 2 + self.minUpVel, peak is 2 high, so spans self.minUpVel to 4+sel.minUpVel
        raiseVelScore = -abs(raiseVel-(2+ self.minUpVel)) + 2
        #print('upVel : {}'.format(raiseVel) )
        #ang vel should be negative to get up
        #angVelPen = (ang - angBefore)/self.dt
        #foot distance - penalize feet being far apart - want to end up standing with feet together
        #foot dist varies from 0 to ~1 in actual rollout
#        lfootCom = self.humanAssist.body('h_foot_left').com()
#        rfootCom = self.humanAssist.body('h_foot').com()
#        #can't use contacts, body seems to leave the ground often in simulation (0 contacts)
#        if(footContacts > 0):
#            if (leftContacts == 0):                
#                lfootCom = self.humanAssist.body('h_foot').com()
#            if (rightContacts == 0):
#                rfootCom = self.humanAssist.body('h_foot_left').com()
#            minFoot = min(rfootCom[0],lfootCom[0])
#            maxFoot = max(rfootCom[0],lfootCom[0])
#            #com projection on ground should be between lowest and highest foot x
#            if(skelCom[0]< minFoot) :
#                comOvAvgFootPen = 5.0* (minFoot - skelCom[0])
#            elif (skelCom[0] >  maxFoot):
#                comOvAvgFootPen = 5.0* (skelCom[0] - maxFoot)
#            else:
#                #between feet - in support rect
#                comOvAvgFootPen =0       
#        else :
#            #both feet off ground - shouldn't be possible
#            if(nonFootContacts == 0) :
#                comOvAvgFootPen=11
#            else:                    
#                comOvAvgFootPen=5*nonFootContacts
#            
#        
#        #reward feet being close together
#        footDist = abs(lfootCom[0] - rfootCom[0])#np.square(lfootCom - rfootCom).sum()
#        ftRwdBs = (1.25*(.8-footDist))#makes +/- 1 for good/bad distance
#        ftRwd = ftRwdBs#(2 * ftRwdBs)#linear penalty
        #reward how high the character's com is
        #possibly causing pelvis to push on the ground?
        height_rew = float(10 * height)
        #reward the feet staying on the ground, to minimize jumping
        contactRew = footContacts
        
        #minimize torques
        actionPenalty = float(1e-3 * np.square(a).sum())

        #reward = alive_bonus + raiseVel + vel + contactRew + height_rew - actionPenalty #- angVelPen #- contactPen
        reward = alive_bonus + raiseVelScore + velScore + contactRew + height_rew - actionPenalty #- angVelPen #- contactPen
        if self.frcDebugMode:
            #work is force through displacement - force dotted with displacement vector
            #work of assistive force 
            #delWork = np.dot(self.assistForce,(forceLocAfter - forceLocBefore))
            #reward close feet, penalize distant feet
#            ftRwdCube = (ftRwdBs**3)
#            rewardNew = reward + ftRwd - comOvAvgFootPen
            #(1e2 * actionPenalty) 
            
            print('upV:{:.5f}| fwdV:{:.5f}| ctct rew:{} | ht_rew:{:.3f}| act:-{:.5f}| reward:{:.5f}'.format(raiseVel, vel , contactRew , height_rew , -actionPenalty,reward))
#            print('upV:{:.5f}| fwdV:{:.5f}| ftDist:{:.5f}|'.format(raiseVel,vel,footDist)+\
#                    'ftRwd:{:.3f}| ftRwdCube:{:.3f}| ht_rew:{:.3f}|'.format(ftRwd,ftRwdCube,height_rew)+\
#                    #'delWork:{:.3f}|  '.format(delWork)+\
#                    'comOvFt:-{:.5f}| act:-{:.5f}| rwrdOld:{:.5f}| rwrdWithExtra:{:.5f}'.format(comOvAvgFootPen,actionPenalty,reward,rewardNew))
#        else :
#            #modify original reward to account for foot location/distance and com over support 
#            reward = reward + ftRwd - comOvAvgFootPen
            
            pass
        #pelvis is ~1.25  high
        # uncomment to enable knee joint limit penalty
        '''joint_limit_penalty = 0
        for j in [-2, -5]:
            if (self.humanAssist.q_lower[j] - self.humanAssist.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.humanAssist.q_upper[j] - self.humanAssist.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)

        reward -= 5e-1 * joint_limit_penalty'''

        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    #(footContacts > 0) and
                    (raiseVelScore >= 0) and
#                    (raiseVel > self.minUpVel) and #(raiseVel > 0.001) and #needs to be slightly greater than 0 or ends up holding still without improving
#                    (raiseVel < (4.0 + self.minUpVel)) and
                    #((leftContacts > 0) or (rightContacts > 0)) and  #end if feet leave ground
                    (height < 1.7) #and 
                    )
#        if (done and not(raiseVelScore >=0)) :
#            print('done with rollout : raiseVelScore : {} : height : {} '.format(raiseVelScore,height))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        skelState = self.activeSkelHndlr.getObs()
        state =  np.concatenate([
                skelState,            #self.frcMult                    
            self.assistForce[:2] #using assist force too high magnitude?
        ])
        #print('st[0] bfr : {} | aftr : {}'.format(state[0],self.humanAssist.bodynodes[2].com()[1]))
        #TODO : why is this here? y value in Q might not be in world coords
        state[0] = self.activeSkelHndlr.skel.bodynodes[2].com()[1]

        return state
    
#override default rendering?   
#    def _render(self, mode='human', close=False):
#        if not self.disableViewer:
#            self._get_viewer().scene.tb.trans[0] = -self.dart_world.skeletons[self.track_skeleton_id].com()[0]*1
#        if close:
#            if self.viewer is not None:
#                self._get_viewer().close()
#                self.viewer = None
#            return
#
#        if mode == 'rgb_array':
#            data = self._get_viewer().getFrame()
#            return data
#        elif mode == 'human':
#            self._get_viewer().runSingleStep()
        
#    #disable/enable viewer
#    def setViewerDisabled(self, vwrDisabled):
#         self.disableViewer = vwrDisabled
    #
    def setDebugMode(self, dbgOn):
        self.frcDebugMode = dbgOn
    
    #called externally to set force multiplier
    def setForceMag(self, frcMultX, frcMultY):
        self.useSetFrc = True
        self.frcMult[0] = frcMultX
        self.frcMult[1] = frcMultY
            
    #given a specific force, set the force multiplier, which is used to set force
    #while this seems redundant, it is intended to preserve the single entry point 
    #to force multiplier modification during scene reset
    def setFrcMultFromFrc(self, frcX, frcY):
        divVal = (9.8 * self.skelHldrs[self.humanIdx].skel.mass())
        frcMultX = frcX/divVal
        frcMultY = frcY/divVal 
        #print('frcMultx = {} | frcMulty = {}'.format(frcMultX,frcMultY))
        self.setForceMag(frcMultX, frcMultY)
        
        
    #for external use only - return observation variable given passed state and state dots
    #obs is slightly different than pure q/qdot (includes height in world frame), requiring skel to be modified
    #restores skel pose when finished - make sure q is correctly configured
    def getObsFromState(self, q, qdot):
        #save current state
        curQ = [x for x in self.humanAssist.q]
        curQdot = [x for x in self.humanAssist.dq]
        #set passed state
        self.set_state(np.asarray(q, dtype=np.float64), np.asarray(qdot, dtype=np.float64))
        #get obs - INCLUDES FORCE VALUE - if using to build new force value, need to replace last 3 elements
        obs = self._get_obs()        
        #return to original state
        self.set_state(np.asarray(curQ, dtype=np.float64), np.asarray(curQdot, dtype=np.float64))
        return obs
    
    #call to force using random force values - ignores initial foce value setting
    def unsetForceVal(self):
        self.useSetFrc = False
        
    #initialize assist force, either randomly or with force described by env consumer
    def _resetForce(self):
        if(not self.useSetFrc):  #using random force       
            #varying force in direction and magnitude
            self.frcMult[0] = self.np_random.uniform(0.0, 0.5)
            self.frcMult[1] = self.np_random.uniform(0.0, 0.5)
        #set assistive force vector for this rollout
        self.initXYAssistForce()
        if(self.frcDebugMode):
            print('_resetForce setting : multX : {:.3f} |multY : {:.3f}\nforce vec : {}'.format(self.frcMult[0],self.frcMult[1],['{:.3f}'.format(i) for i in self.assistForce[:2]]))

    #build random state   
    def getRandomInitState(self, poseDel=None):
        if(poseDel is None):
            poseDel=self.poseDel
            
        return s
        #set walker to be laying on ground
        self.setToInitPose()
        #perturb init state and statedot
        qpos = self.humanAssist.q + self.np_random.uniform(low=-poseDel, high=poseDel, size=self.humanAssist.ndofs)
        qvel = self.humanAssist.dq + self.np_random.uniform(low=-poseDel, high=poseDel, size=self.humanAssist.ndofs)
        return qpos, qvel
    
    #set initial state externally - call before reset_model is called by trainin process
    def setNewState(self, _qpos, _qvel):
        #set to false to use specified states
        self.randomizeInitState = False
        self.initQPos = np.asarray(_qpos, dtype=np.float64)
        self.initQVel = np.asarray(_qvel, dtype=np.float64)
        self.loadedInitStateSet = True
    
    #called at beginning of each rollout
    def reset_model(self):
        self.dart_world.reset()
        #initialize assist force, either randomly or with force described by env consumer
        self._resetForce()
        
        if(self.randomizeInitState):#if not random, setInitPos will set the pose
            qpos, qvel = self.getRandomInitState()
            self.set_state(qpos, qvel)
        else:
            #set walker to be laying on ground
            self.setToInitPose()
            if (self.loadedInitStateSet):
                if(self.frcDebugMode):
                    print('DartStandUp2dEnv Notice : Setting specified init q/qdot/frc')
                    print('initQPos : {}'.format(self.initQPos))
                    print('initQVel : {}'.format(self.initQVel))
                    print('initFrc : {}'.format(self.assistForce))
                self.set_state(self.initQPos, self.initQVel)
                self.loadedInitStateSet = False
            else:
                print("DartStandUp2dEnv Warning : init skel state not randomized nor set to precalced random state")

        return self._get_obs()

    def viewer_setup(self):
        self._get_viewer().scene.tb.trans[2] = -5.5

       
    #display the state of the skeleton bodies and joints
    def dbgDisplayState(self):
        for i in range(len(self.humanAssist.dofs)):
            #print('idx i={} name={} : {}'.format(i,self.humanAssist.dofs[i].name, self.humanAssist.q[i]))
            print('{}'.format(self.humanAssist.q[i]))
            
    def dbgShowSkelList(self, ara, araType):
        print(araType)
        for i in range(len(ara)):
            print('idx i={} name={}'.format(i,ara[i].name))
