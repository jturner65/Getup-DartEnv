

import numpy as np
from gym import utils
from gym.envs.dart import dart_env

#3d full body skel to getup
class DartStandUpFullBody3dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        #full skel - 6 root dofs
        # number of controllable dofs
        numActDofs = 31
        #clips control to be within 1 and -1
        self.control_bounds = np.array([[1.0]*numActDofs,[-1.0]*numActDofs])
        #need to change this to spd-based ?
        
        #scales control to be between 2 and -2
        self.action_scale = np.array([1.0]*numActDofs)
##        #thighs 
#        self.action_scale[[0,1,2,7,8,9]] = 100
##        #knees
#        self.action_scale[[3,10]] = 100
#        #abdomen and spine
#        self.action_scale[[14,15,16]] = 50
#        #shoulders/biceps
#        self.action_scale[[19,20,21,22,25,26,27,28]] = 15
        #causes bipedJT to explode but seems to need this
        #scale feet actions less
#        self.action_scale[[4,5,6,11,12,13]] = 5
        #scale thigh more (?)
        #self.action_scale[[0, 1, 2, 7, 8 ,9]] = 150
        #q and qdot == 2 * numDofs
        #q[1:] + qdot + 6 frc/trq dofs
        obs_dim = 36 + 37 + 6#79
        #vector of "assisting" force applied to left or right hand
        self.assistForce = np.zeros(3)
        self.assistTorque = np.zeros(3)
        self.assistBody = 'h_hand_left' #or h_hand_right

        self.forceMag = 150
        self.assistForce[1] = self.forceMag
        #choice should be forceMag in 0 for left or 1 for right
        self.forceChoice = [self.forceMag,0]
        
        self.t = 0

        dart_env.DartEnv.__init__(self, 'bipedJT.skel', 8, obs_dim, 
                                  self.control_bounds, dt=0.001,
                                  disableViewer=False)
        #bipedJT has 37 dofs, of which 6 are root -> use 31
        #self.dbgShowSkelList(self.robot_skeleton.bodynodes, 'body nodes')
        #self.dbgShowSkelList(self.robot_skeleton.dofs, 'dofs')
        #print('# skel q size : ' + str(np.size(self.robot_skeleton.q)))

        self.robot_skeleton.set_self_collision_check(True)
        self.makeInitPose()
        #ground skel
        for i in range(1, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(0)
            
        utils.EzPickle.__init__(self)
        
    def makeInitPose(self):
        #self.displayState()
        #self.initPose = np.zeros(self.robot_skeleton.ndofs)

        tmpPose = [#sitting with knees bent - attempting to minimize initial collision with ground
            #root orientation
            0,0,1.57,
            #root location
            0,-0.842,0,
            
            #left thigh
            0.87,0,0,
            #left shin, left heel(2), left toe
            -1.85,-0.6,0, 0,
            #right thigh
            0.87,0,0,
            #right shin, right heel(2), right toe
            -1.85,-0.6,0, 0,
            #abdoment(2), spine
            0,-1.5,0,
            #head
            0,0,
            #scap left, bicep left(3)
            0,0,-0.5,0,
            #forearm left, hand left
            0.5,0,
            #scap right, bicep right (3)
            0,0,0.5,0,
            #forearm right.,hand right
            0.5,0]
        self.initPose = np.array(tmpPose)

    def setInitPose(self):
        #initPose[]
        self.robot_skeleton.set_positions(self.initPose)

    #display the state of the skeleton bodies and joints
    def displayState(self):
        for i in range(len(self.robot_skeleton.dofs)):
            #print('idx i={} name={} : {}'.format(i,self.robot_skeleton.dofs[i].name, self.robot_skeleton.q[i]))
            print('{}'.format(self.robot_skeleton.q[i]))
        
    def dbgShowSkelList(self, ara, araType):
        print(araType)
        for i in range(len(ara)):
            print('idx i={} name={}'.format(i,ara[i].name))

    def advance(self, a):
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        tau = np.zeros(self.robot_skeleton.ndofs)
        tau[6:] = clamped_control * self.action_scale

        self.do_simulation(tau, self.frame_skip)

    #added by JT
    def procOrient(self, orientAxes):
        oVec = np.array(orientAxes)
#        a = self.robot_skeleton.bodynodes[0].to_world(oVec)
#        b = self.robot_skeleton.bodynodes[0].to_world(np.array([0, 0, 0]))
        oVec_W = self.robot_skeleton.bodynodes[0].to_world(oVec) - self.robot_skeleton.bodynodes[0].to_world(np.array([0, 0, 0]))
#        tmpVec = np.array(oVec_W)
        norm = np.linalg.norm(oVec_W)
        if(norm == 0):
            return 10
        oVec_W /= norm
        ang_cos = np.arccos(np.dot(oVec, oVec_W))
#        print('{}|{}  {} | {} | {}' .format(a,b,tmpVec, oVec_W,ang_cos))
#        print('')
        return ang_cos
    
    def _step(self, a):
        #add force to skeleton before advancingBodyNode
        #add_ext_force has _offset == np.zeros(3)
        #self.assistBody is either h_hand_left or h_hand_right
        self.robot_skeleton.body(self.assistBody).add_ext_force(self.assistForce)
        
        comBefore = self.robot_skeleton.bodynodes[0].com()
        self.advance(a)
        comAfter = self.robot_skeleton.bodynodes[0].com()
        comDiff = comAfter - comBefore
        
        #angular terms
        ang_cos_swd = self.procOrient([0, 0, 1])
        ang_cos_uwd = self.procOrient([0, 1, 0])
        ang_cos_fwd = self.procOrient([1, 0, 0])

        contacts = self.dart_world.collision_result.contacts
        total_force_mag = 0
        for contact in contacts:
            total_force_mag += np.square(contact.force).sum()

        joint_limit_penalty = 0
        for j in [-3, -9]:
            if (self.robot_skeleton.q_lower[j] - self.robot_skeleton.q[j]) > -0.05:
                joint_limit_penalty += abs(1.5)
            if (self.robot_skeleton.q_upper[j] - self.robot_skeleton.q[j]) < 0.05:
                joint_limit_penalty += abs(1.5)
        
        #rewards
        alive_bonus = 1.0
        up_vel_rew = .01 * (comDiff[1]) / self.dt  #y dir
        height_rew = 5.0 /(2.0 - comAfter[1])
        #penalties
        action_pen = 1e-3 * np.square(a).sum()
        joint_pen = 2e-1 * joint_limit_penalty
        #correct? only look at displacement from 0
        fwd_deviation_pen = 1e-3 * abs(comAfter[0])
        side_deviation_pen = 1e-3 * abs(comAfter[2])
        reward = up_vel_rew + alive_bonus + height_rew - action_pen - joint_pen - side_deviation_pen - fwd_deviation_pen

        self.t += self.dt
        
        #need to redo this
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (comAfter[1] < 1.05) and #(height > 1.05) and (height < 2.0) and 
                    (comDiff[1] > 0) and
                    (comDiff[2] < .1) and
                    (comDiff[0] < .1) and
                    
                    (abs(ang_cos_uwd) < np.pi) and 
                    (abs(ang_cos_fwd) < np.pi) and
                    (abs(ang_cos_swd) < np.pi) 
                    )

        if done:
            reward = 0

        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10),
            self.assistForce,
            self.assistTorque
        ])
        #print ('state size : ' + str(np.size(state)) + '|' + str(np.size(self.robot_skeleton.q[1:])) + '|' + str(np.size(self.robot_skeleton.dq)))

        return state

    def reset_model(self):
        self.dart_world.reset()
        self.setInitPose()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.t = 0
        #set force to be applied to hand
        #choice should be forceMag in 0 for left or 1 for right
#        val = self.np_random.randint(0,2)# either 0 or 1 #(2 * self.np_random.randint(0,2)) - 1 #-1 or 1
#        if(val > 0):#left
#            #upward force on "left"
#            self.forceChoice = [self.forceMag,0]
#            self.assistBody = 'h_hand_left'
#        else :#right
#            #upward force on "right"
#            self.forceChoice = [0,self.forceMag]
#            self.assistBody = 'h_hand_right'

        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5


"""
bipedJT stuff : 
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
dofs
idx i=0 name=j_pelvis_rot_x
idx i=1 name=j_pelvis_rot_y
idx i=2 name=j_pelvis_rot_z
idx i=3 name=j_pelvis_pos_x
idx i=4 name=j_pelvis_pos_y
idx i=5 name=j_pelvis_pos_z

idx i=6 name=j_thigh_left_z
idx i=7 name=j_thigh_left_y
idx i=8 name=j_thigh_left_x
idx i=9 name=j_shin_left
idx i=10 name=j_heel_left_1
idx i=11 name=j_heel_left_2
idx i=12 name=j_toe_left
idx i=13 name=j_thigh_right_z
idx i=14 name=j_thigh_right_y
idx i=15 name=j_thigh_right_x
idx i=16 name=j_shin_right
idx i=17 name=j_heel_right_1
idx i=18 name=j_heel_right_2
idx i=19 name=j_toe_right
idx i=20 name=j_abdomen_1
idx i=21 name=j_abdomen_2
idx i=22 name=j_spine
idx i=23 name=j_head_1
idx i=24 name=j_head_2
idx i=25 name=j_scapula_left
idx i=26 name=j_bicep_left_z
idx i=27 name=j_bicep_left_y
idx i=28 name=j_bicep_left_x
idx i=29 name=j_forearm_left
idx i=30 name=j_hand_left
idx i=31 name=j_scapula_right
idx i=32 name=j_bicep_right_z
idx i=33 name=j_bicep_right_y
idx i=34 name=j_bicep_right_x
idx i=35 name=j_forearm_right
idx i=36 name=j_hand_right
# skel q size : 37
#collapsed tmp pose
#        tmpPose = [
#            #root orientation
#            -0.03899879803664157, 0.26267662126263625, -0.37517055603957367,
#            #root location
#            -0.15563517625753198,-0.6919700788956382,-0.02370566609014574,
#            #left thigh
#            1.7821873667628814, -0.011980422054887906,0.22010792894775735,
#            #left shin, left heel(2), left toe
#            -3.000241608523221,1.1745579096358858, 0.0, 0.3974793580913019,
#            #right thigh
#            1.789496515162747,-0.00837634810685024,-0.5926692727257583,
#            #right shin, right heel(2), right toe
#            -3.00025237384968,1.1754005983839533,0.0,0.395405648124891,
#            #abdoment(2), spine
#            0.026625601256421612,-1.3996918843847406,0.9522299963455227,
#            #head
#            -1.2096540619734608,1.2169492686885983,
#            #scap left, bicep left(3)
#            0.6148244479137597, 1.5723191214698384,0.7891229895287946,-1.7203613478101425,
#            #forearm left, hand left
#            2.0264581693690795, -1.5786886716410031,
#            #scap right, bicep right (3)
#            0.05855995428410524,-3.717776734491244, -0.7255442988369272, -3.4973063529649346,
#            #forearm right.,hand right
#            1.4203802551915548, 0.6049590709229998]

"""