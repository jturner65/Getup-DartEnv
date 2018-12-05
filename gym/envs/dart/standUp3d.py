

import numpy as np
from gym import utils
from gym.envs.dart import dart_env

#3d skel to getup
class DartStandUp3dEnv(dart_env.DartEnv, utils.EzPickle):
    def __init__(self):
        numDofs = 15
        """
        node names:
        node i=0 name=h_torso_aux
        node i=1 name=h_torso
        node i=2 name=h_pelvis
        node i=3 name=h_thigh
        node i=4 name=h_shin
        node i=5 name=h_foot
        node i=6 name=h_thigh_left
        node i=7 name=h_shin_left
        node i=8 name=h_foot_left
        
        dofs:
        idx i=0 name=j_torso_x
        idx i=1 name=j_torso_y
        idx i=2 name=j_torso_z
        idx i=3 name=j_torso(1)_z
        idx i=4 name=j_torso(1)_y
        idx i=5 name=j_torso(1)_x
        
        idx i=6 name=j_pelvis_x_x
        idx i=7 name=j_pelvis_x_y
        idx i=8 name=j_pelvis_x_z
        idx i=9 name=j_thigh_z
        idx i=10 name=j_thigh_y
        idx i=11 name=j_thigh_x
        idx i=12 name=j_shin
        idx i=13 name=j_foot_1
        idx i=14 name=j_foot_2
        idx i=15 name=j_thigh_left_z
        idx i=16 name=j_thigh_left_y
        idx i=17 name=j_thigh_left_x
        idx i=18 name=j_shin_left
        idx i=19 name=j_foot_left_1
        idx i=20 name=j_foot_left_2
        """
        self.control_bounds = np.array([[1.0]*numDofs,[-1.0]*numDofs])
        self.action_scale = np.array([100.0]*numDofs)
        #scale feet actions mult less
        self.action_scale[[7,8,13,14]] = 20
        #scale pelvis more 
        self.action_scale[[0, 1, 2]] = 150
        #20 q (?), 21 qdot, 2 w (force choice)
        obs_dim = 2 * numDofs + 5 + 6 + 2#43
        #vector of force to apply to h_torso
        self.torso_force = np.zeros(3)
        self.forceMag = 50
        #force is upward
        self.torso_force[1] = self.forceMag
        self.forceOffset = np.zeros(3)
        self.forceOffset[2] = .22
        #choice vertical force applied on either left or right
        self.forceChoice = [self.forceMag,0]
        #set values to 50,50 or 50,-50
        self.t = 0

        dart_env.DartEnv.__init__(self, 'walker3d_waist.skel', 4, obs_dim, self.control_bounds, disableViewer=False)
        #walker3d has 21 dofs, of which 6 are root (i.e. cannot include in action space)
#        self.dbgShowSkelList(self.robot_skeleton.bodynodes, 'body nodes')
#        self.dbgShowSkelList(self.robot_skeleton.dofs, 'dofs')

        self.robot_skeleton.set_self_collision_check(True)
        #ground skel
        for i in range(1, len(self.dart_world.skeletons[0].bodynodes)):
            self.dart_world.skeletons[0].bodynodes[i].set_friction_coeff(0)

        utils.EzPickle.__init__(self)
        
    #display skeleton components idx and name
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
        oVec_W = self.robot_skeleton.bodynodes[0].to_world(oVec) - self.robot_skeleton.bodynodes[0].to_world(np.array([0, 0, 0]))
        oVec_W /= np.linalg.norm(oVec_W)
        ang_cos = np.arccos(np.dot(oVec, oVec_W))
        return ang_cos
    
    def _step(self, a):
        #add force to skeleton before advancing
        #add_ext_force has _offset == np.zeros(3)
        self.robot_skeleton.body('h_torso').add_ext_force(self.torso_force, _offset=self.forceOffset)
        
        posbefore = self.robot_skeleton.bodynodes[0].com()[0]
        self.advance(a)

        posafter = self.robot_skeleton.bodynodes[0].com()[0]
        height = self.robot_skeleton.bodynodes[0].com()[1]
        side_deviation = self.robot_skeleton.bodynodes[0].com()[2]
        
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

        #need to redo this
        alive_bonus = 1.0
        vel_rew = 1.0 * (posafter - posbefore) / self.dt
        action_pen = 1e-3 * np.square(a).sum()
        joint_pen = 2e-1 * joint_limit_penalty
        deviation_pen = 1e-3 * abs(side_deviation)
        reward = vel_rew + alive_bonus - action_pen - joint_pen - deviation_pen

        self.t += self.dt
        
        #need to redo this
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > 1.05) and (height < 2.0) and (abs(ang_cos_uwd) < 0.84) and (abs(ang_cos_fwd) < 0.84))

        if done:
            reward = 0

        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        state =  np.concatenate([
            self.robot_skeleton.q[1:],
            np.clip(self.robot_skeleton.dq,-10,10),
            self.forceChoice,
        ])
        #print ('state size : ' + str(np.size(state)) + '|' + str(np.size(self.robot_skeleton.q[1:])) + '|' + str(np.size(self.robot_skeleton.dq)))

        return state

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.q + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        qvel = self.robot_skeleton.dq + self.np_random.uniform(low=-.005, high=.005, size=self.robot_skeleton.ndofs)
        self.set_state(qpos, qvel)
        self.t = 0
        #set force to be applied to skeleton here randomly choosing -1 or 1, and pass result with observation
        
        val = self.np_random.randint(0,2)# either 0 or 1 #(2 * self.np_random.randint(0,2)) - 1 #-1 or 1
        if(val > 0):
            #upward force on "left"
            self.forceChoice = [self.forceMag,0]
            #force application placement on torso
            self.forceOffset[2] = .22 
        else :
            #upward force on "right"
            self.forceChoice = [0,self.forceMag]
            #force application placement on torso
            self.forceOffset[2] = -.22 
          
        return self._get_obs()

    def viewer_setup(self):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[2] = -5.5
