import numpy as np
#import pydart2 as pydart
from gym import utils
from gym.envs.dart import dart_env
#from pydart2.marker import Marker


class DartStandUp2dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        self.control_bounds = np.array([[1.0]*6,[-1.0]*6])
        self.action_scale = np.array([100, 100, 20, 100, 100, 20])
        obs_dim = 19
        #display sphere at point of force application
        self.useSphereDBG = False
        #display debug information regarding force application
        self.frcDebugMode = False
        #use varying force in 2d
        self.use2Dforce = True
        #use the mean state only if set to false, otherwise randomize initial state
        self.randomizeInitState = True
        #use these to hold preset initial states - only used if randomizeInitState is false
        self.loadedInitStateSet = False
        self.initQPos = None
        self.initQVel = None
        #+/- perturb amount of intial pose and vel for random start state
        self.poseDel = .005
        #list of x,y assist force multipliers - init to .3, .3
        self.frcMult = np.array([.3,.3])
        #minimum allowed raise velocity
        self.minUpVel = .003
        
        
        if(self.useSphereDBG):
            dart_env.DartEnv.__init__(self, 'walker2d_withSphere.skel', 4, obs_dim, self.control_bounds, disableViewer=False)
            self.initDbgFrcLoc()
        else:
            dart_env.DartEnv.__init__(self, 'walker2d.skel', 4, obs_dim, self.control_bounds, disableViewer=False)
        #self.dbgShowSkelList(self.robot_skeleton.bodynodes, 'body nodes')
        #print('skel q size : ' + str(np.size(self.robot_skeleton.q)))
        #print('left foot loc : ' + str(self.robot_skeleton.bodynode('h_foot_left').C))
        #print('pelvis loc : ' + str(self.robot_skeleton.bodynode('h_pelvis').C))
        #build initial pose
        self.makeInitPose()
        #initialize assist force
        self.initXYAssistForce()
        #print('assist force : {} mass : {}'.format(self.assistForce,self.robot_skeleton.mass()))
        #don't use preset value for force, use random value
        self.useSetFrc = False
        #location of force application on pelvis body -> coincides with top edge
        frcLoc = [0,.25,0]
        self.frcOffset = np.array(frcLoc)
        self.timeElapsed = 0
        utils.EzPickle.__init__(self)
        
                
        """
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
    def makeInitPose(self):
        #self.dbgDisplayState()
        self.initPose = np.zeros(self.robot_skeleton.ndofs)
        #height should be .1
        self.initPose[1] = -1.2
        #rotation needs to be pi/2
        self.initPose[2] = 1.57
        #bend knees
        #rotate at hips
        self.initPose[3] = .87
        self.initPose[6] = .87
        #rotate at knees
        self.initPose[4] = -1.6
        self.initPose[7] = -1.6
        #bend ankles
        self.initPose[5] = -.85
        self.initPose[8] = -.85       
    

    #should only be called during rollouts consuming value function predictions
    #value function provides actual force given a state
    #NEED TO USE setFrcMultFromFrc when setting initial force before reset
    def setAssistForce(self, xfrc, yfrc):
        self.setFrcMultFromFrc(xfrc, yfrc)
        self.initXYAssistForce()  

       
    def initXYAssistForce(self):
        xfrc = self.getForceFromMult(self.frcMult[0])
        yfrc = self.getForceFromMult(self.frcMult[1])
        self.assistForce = np.zeros(3)
        self.assistForce[0] = xfrc
        self.assistForce[1] = yfrc   
    
    #to extract multiplier from given force value
    def getMultFromFrc(self, frc):
        return frc/(9.8 * self.robot_skeleton.mass())
    
    #also called externally by testing mechanisms
    def getForceFromMult(self, frcMult):
        return (9.8 * self.robot_skeleton.mass())*frcMult
           
    #also called externally by testing mechanisms (? TODO: verify)
    def setInitPose(self):
        #initPose[]
        self.robot_skeleton.set_positions(self.initPose)

    def advance(self, a, stIdx):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[stIdx:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)
    
    def _step(self, a):
        if(self.useSphereDBG):
            self.dbgFrcLoc()
            
        self.robot_skeleton.body('h_pelvis').add_ext_force(self.assistForce, _offset=self.frcOffset)
        
        heightBefore =  self.robot_skeleton.bodynodes[2].com()[1]
        posbefore, angBefore = self.robot_skeleton.q[0,2]
        #for work calc
        #forceLocBefore = self.robot_skeleton.bodynodes[2].to_world(self.frcOffset)
        
        self.advance(a, 3)
        #for work calc 
        #forceLocAfter = self.robot_skeleton.bodynodes[2].to_world(self.frcOffset)
                
        posafter,ang = self.robot_skeleton.q[0,2]
        skelCom = self.robot_skeleton.bodynodes[2].com()
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
#        lfootCom = self.robot_skeleton.body('h_foot_left').com()
#        rfootCom = self.robot_skeleton.body('h_foot').com()
#        #can't use contacts, body seems to leave the ground often in simulation (0 contacts)
#        if(footContacts > 0):
#            if (leftContacts == 0):                
#                lfootCom = self.robot_skeleton.body('h_foot').com()
#            if (rightContacts == 0):
#                rfootCom = self.robot_skeleton.body('h_foot_left').com()
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
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
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
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            #np.clip(self.robot_skeleton.dq,-10,10),
            self.robot_skeleton.dq,
            #self.frcMult                    
            self.assistForce[:2] 
        ])
        #print('st[0] bfr : {} | aftr : {}'.format(state[0],self.robot_skeleton.bodynodes[2].com()[1]))
        #TODO : why is this here?
        state[0] = self.robot_skeleton.bodynodes[2].com()[1]

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
        
    #disable/enable viewer
    def setViewerDisabled(self, vwrDisabled):
         self.disableViewer = vwrDisabled
    #
    def setDebugMode(self, dbgOn):
        self.frcDebugMode = dbgOn
        
    #set force value to be added as assistive force
    #frcVal is vector of values for force
    #isMult : true means this is mg multiplier, false means actual force
    #def setForceVal(self, frcVal, isMult):
    
    #called externally to set force multiplier
    def setForceMult(self, frcMultX, frcMultY=None):
        self.useSetFrc = True
        if(frcMultY==None):
        #if no frcY then frxX is magnitude : hyp of pi/4 force; we want x and y values of magnitude
            xyVal = frcMultX/1.41421356237
            self.frcMult[0] = xyVal
            self.frcMult[1] = xyVal
        else :
            self.frcMult[0] = frcMultX
            self.frcMult[1] = frcMultY
            
    #given a specific force, set the force multiplier, which is used to set force
    #while this seems redundant, it is intended to preserve the single entry point 
    #to force multiplier modification during scene reset
    def setFrcMultFromFrc(self, frcX, frcY=None):
        frcMultX = self.getMultFromFrc(frcX)
        frcMultY = None if (frcY is None) else self.getMultFromFrc(frcY)
        #print('frcMultx = {} | frcMulty = {}'.format(frcMultX,frcMultY))
        self.setForceMult(frcMultX, frcMultY)
        
        
    #for external use only - return observation variable given passed state and state dots
    #obs is slightly different than pure q/qdot (includes height in world frame), requiring skel to be modified
    #restores skel pose when finished
    def getObsFromState(self, q, qdot):
        #save current state
        curQ = [x for x in self.robot_skeleton.q]
        curQdot = [x for x in self.robot_skeleton.dq]
        #set passed state
        self.set_state(np.asarray(q, dtype=np.float64), np.asarray(qdot, dtype=np.float64))
        #get obs - INCLUDES FORCE VALUE - if using to build new force value, need to replace last 2 elements
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
            if(self.use2Dforce): 
                #varying force in direction and magnitude
                self.frcMult[0] = self.np_random.uniform(0.0, 0.5)
                self.frcMult[1] = self.np_random.uniform(0.0, 0.5)
            else :
                #setting desired magnitude of force, applied at pi/4, to be 0->.6 * mg (set x and y components of equalateral right triangle with hyp == mag)
                forceMag = self.np_random.uniform(0.0, 0.6)/1.41421356237
                self.frcMult[0] = forceMag
                self.frcMult[1] = forceMag
        #set assistive force vector for this rollout
        self.initXYAssistForce()
        if(self.frcDebugMode):
            print('_resetForce setting : multX : {:.3f} |multY : {:.3f}\nforce vec : {}'.format(self.frcMult[0],self.frcMult[1],['{:.3f}'.format(i) for i in self.assistForce[:2]]))

    #build random state   
    def getRandomInitState(self, poseDel=None):
        if(poseDel is None):
            poseDel=self.poseDel
        #set walker to be laying on ground
        self.setInitPose()
        #perturb init state and statedot
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-poseDel, high=poseDel, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-poseDel, high=poseDel, size=self.robot_skeleton.ndofs)
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
            self.setInitPose()
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

    #load and show sphere at position of force application
    def initDbgFrcLoc(self):
        #sphere is 2nd object in world(walker is last)
        self.sphere = self.dart_world.skeletons[1]
        self.sphere.bodynodes[0].set_collidable(False)
        self.sphere.set_mobile(False)
        self.dbgShowSkelList(self.sphere.bodynodes, 'sphere body nodes')
        self.dbgShowSkelList(self.sphere.dofs, 'sphere dofs')
        
    def dbgFrcLoc(self):
        spherePos = self.sphere.positions()
        #print('{}'.format(['{:.3f}'.format(i) for i in spherePos]))
        spherePos[3:] = self.robot_skeleton.bodynodes[2].to_world(self.frcOffset)
        #spherePos[3] +=1
        print('Sphere Position : {}'.format(['{:.3f}'.format(i) for i in spherePos[3:]]))
        self.sphere.set_positions(spherePos)
        
    #display the state of the skeleton bodies and joints
    def dbgDisplayState(self):
        for i in range(len(self.robot_skeleton.dofs)):
            #print('idx i={} name={} : {}'.format(i,self.robot_skeleton.dofs[i].name, self.robot_skeleton.q[i]))
            print('{}'.format(self.robot_skeleton.q[i]))
            
    def dbgShowSkelList(self, ara, araType):
        print(araType)
        for i in range(len(ara)):
            print('idx i={} name={}'.format(i,ara[i].name))
