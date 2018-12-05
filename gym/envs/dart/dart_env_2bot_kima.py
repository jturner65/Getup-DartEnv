#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:25:18 2018

@author: akankshabindal
"""

# Modified version of dart_env.py to support 2 skeletons

import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six

from collections import defaultdict
import nlopt
from math import sqrt

from gym.envs.dart.static_window import *
import scipy.misc

try:
    import pydart2 as pydart
    from pydart2.gui.trackball import Trackball
    pydart.init()
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install pydart2.)".format(e))


class DartEnv2BotKima(gym.Env):
    """Superclass for dart environment with 2 bots contained in one skel file
    """

    def __init__(self, model_paths, frame_skip,  \
                 dt=0.002, obs_type="parameter", action_type="continuous", visualize=True, disableViewer=False,\
                 screen_width=80, screen_height=45):
        assert obs_type in ('parameter', 'image')
        assert action_type in ("continuous", "discrete")
        print('pydart initialization OK')
        self.viewer = None

        if len(model_paths) < 1:
            raise StandardError("At least one model file is needed.")

        if isinstance(model_paths, str):
            model_paths = [model_paths]
            
        #list of all skelHolders - convenience class holding all relevant info about a skeleton
        self.skelHldrs = []
        #load skels, make world
        self.loadAllSkels(model_paths, dt)                
        #build list of slkeleton handlers
        self._buildSkelHndlrs()
        #set which skeleton is active, which also sets action and observation space
        self.setActiveSkelHndlr(True)
        
        
        self._obs_type = obs_type
        self.frame_skip= frame_skip
        self.visualize = visualize  #Show the window or not
        self.disableViewer = False or disableViewer

        # random perturbation
        self.add_perturbation = False
        self.perturbation_parameters = [0.05, 5, 2] # probability, magnitude, bodyid, duration
        self.perturbation_duration = 40
        self.perturb_force = np.array([0, 0, 0])
        # initialize the viewer, get the window size
        # initial here instead of in _render in image learning
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._get_viewer()

        self._seed()
        #self._seed = 5794
        #print(self._seed)
        
        #overriding env stuff
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : int(np.round(1.0 / self.dt))
        }
        
        
    #set which skeleton we are currently using for training
    #also sets action and observation variables necessary for env to work
    def setActiveSkelHndlr(self, isHumanSkel=True):
        if(isHumanSkel):
            self.actSKHndlrIDX = self.humanIdx
        else :
            self.actSKHndlrIDX = self.botIdx
           
        self.activeSkelHndlr = self.skelHldrs[self.actSKHndlrIDX]
        self.updateObsActSpaces()

    #call this to update the observation and action space dimensions if they change
    #either because active skeleton handler has changed, or because dimension of assist
    #force has changed
    def updateObsActSpaces(self):    
        self.obs_dim = self.activeSkelHndlr.obs_dim
        self.act_dim = self.activeSkelHndlr.action_dim
        self.action_space = self.activeSkelHndlr.action_space
        self.observation_space = self.activeSkelHndlr.observation_space
                
        
    #set skeleton values for 2D waist-down skeleton
    def _setSkel2DWaistDown(self, skelH, widx):
        print('Skel {} is in _setSkel2DWaistDown'.format(skelH.skel.name))
        
        skelH.setCntlBnds(np.array([[1.0]*6,[-1.0]*6]))
        skelH.setActionScale(np.array([100, 100, 20, 100, 100, 20]))
        # observation dimension == # of q (8) , qdot (9) components + # of assistive force components (3)
        skelH.setObsDim(20)
        return skelH

        
    #3d version of waist down bot
    def _setSkel3DWaist(self, skelH, widx):
        print('Skel {} is in _setSkel3DWaist'.format(skelH.skel.name))
        #'walker3d_waist.skel'
        action_scale = np.array([100.0]*15)
        action_scale[[-1,-2,-7,-8]] = 20
        action_scale[[0, 1, 2]] = 150
        skelH.setCntlBnds(np.array([[1.0]*15,[-1.0]*15]))
        skelH.setActionScale(action_scale)
        # observation dimension == # of q (8) , qdot (9) components + # of assistive force components (3)
        #TODO in 3d this is wrong, root dofs are 3 orient, 3 loc , first idx is orient around x, not x pos like in 2d
        skelH.setObsDim(41)
        return skelH
    
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
        #joints in kima
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
# [WeldJoint(17): j_forearm_right_attach]]
        
        #dofs in kima
#[[Dof(0): j_pelvis_x],
# [Dof(1): j_pelvis_y],
# [Dof(2): j_pelvis_z],
# [Dof(3): j_pelvis2_z],
# [Dof(4): j_pelvis2_y],
# [Dof(5): j_pelvis2_x],
# [Dof(6): j_thigh_left_z],
# [Dof(7): j_thigh_left_y],
# [Dof(8): j_thigh_left_x],
# [Dof(9): j_shin_left],
# [Dof(10): j_heel_left_1],
# [Dof(11): j_heel_left_2],
# [Dof(12): j_thigh_right_z],
# [Dof(13): j_thigh_right_y],
# [Dof(14): j_thigh_right_x],
# [Dof(15): j_shin_right],
# [Dof(16): j_heel_right_1],
# [Dof(17): j_heel_right_2],
# [Dof(18): j_abdomen_1],
# [Dof(19): j_abdomen_2],
# [Dof(20): j_spine],
# [Dof(21): j_bicep_left_z],
# [Dof(22): j_bicep_left_y],
# [Dof(23): j_bicep_left_x],
# [Dof(24): j_forearm_left],
# [Dof(25): j_bicep_right_z],
# [Dof(26): j_bicep_right_y],
# [Dof(27): j_bicep_right_x],
# [Dof(28): j_forearm_right]]
    
    
    
    def _setSkel3DFullBody(self, skelH, widx):
        print('Skel {} is in _setSkel3DFullBody'.format(skelH.skel.name))                
        #numdofs - 6 root dofs
        numActDofs = 23
        #clips control to be within 1 and -1
        skelH.setCntlBnds(np.array([[1.0]*numActDofs,[-1.0]*numActDofs]))        
        #scales control to be between 150 and -150
        action_scale = np.array([100.0]*numActDofs)
        #these are dof idxs -6 (we never apply actions to first 6 dofs)
        #irrelevant dofs set to action scale 1/100th to minimize instablity caused by self-collision
        #thigh twist and spread 
        #action_scale[[0,3,6,9, 12, 13, 14, 16, 20]]*= 0.2
        action_scale[[1, 2, 7, 8]] *= .1
        #head 
        #action_scale[[21,23, ]] *= .01
        #left shoulders, 2 of bicep, and hands
        action_scale[[15,17, 19, 21]]*= .05
        #right shoulders, 2 of bicep, hands
        #action_scale[[25,26]]*= .01
        #shoulders/biceps
        #action_scale[[19,20,21,22,25,26,27,28]] *= .75
        #scale ankles actions less
        action_scale[[4,5, 10, 11]]*= .20
        #scale feet and forearms much less
        action_scale[[18, 22]]*= .1
                
        print('action scale : {}'.format(action_scale))
        skelH.setActionScale(action_scale)
        #input()
        #2 * numDofs  : can't ignore 1st element in q in 3d.
        skelH.setObsDim(2*skelH.skel.num_dofs())
        return skelH
    
    def _initLoadedSkel(self, skel, widx, isHuman, isFullRobot, skelHldrIDX):       
        #3/16/18 problem is with skeleton
        #need to set skeleton joint stiffness and damping, and body friction
        #maybe even coulomb friction for joints.
        #set for every joint except world joint - joint limits, stiffness, damping
        #print('Set Joints for Skel {}'.format(skel.name))
        for jidx in range(skel.njoints):
            j = skel.joint(jidx)
            #don't mess with free joints
            if ('Free' in str(j)):
                continue
            nDof = j.num_dofs()
            for didx in range(nDof):
                if j.has_position_limit(didx):
                    j.set_position_limit_enforced(True)             
                
                j.set_damping_coefficient(didx, 10.)
                j.set_spring_stiffness(didx, 50.)
                   
                    
                    
        #eventually check for # dofs in skel to determine init
        if(skel.ndofs==9):#2D
            stIDX = 3
        elif(skel.ndofs==37):
            stIDX = 6
        elif(skel.ndofs==29):
            stIDX = 6
        else :
            print('DartEnv2BotKima::_initLoadedSkel : Unknown skel type based on # of dofs {} for skel {}'.format(skel.ndofs,skel.name))
            return None

        if(isHuman):
            skelH = humanSkelHolder(self, skel, widx,stIDX, skelHldrIDX)  
        elif(isFullRobot):
            skelH = robotSkelHolder(self, skel, widx,stIDX, skelHldrIDX)  
        else:
            skelH = robotArmSkelHolder(self, skel, widx,stIDX, skelHldrIDX)  

        return self._setSkel3DFullBody(skelH, widx)
        
    #get skeleton objects from list in dart_world  
    def _buildSkelHndlrs(self):
        numBots = self.dart_world.num_skeletons()
        idxSkelHldrs = 0
        self.hasHelperBot = False
        self.grabLink = None
        for idx in range(numBots):
            skelType = ''
            skel = self.dart_world.skeletons[idx]
            skelName = skel.name
            
        
#        for i in range(self.robot_skeleton.njoints-1):
#           #print("joint:%d"%(i),skel.joint(i).name)
#           #print("position limit",skel.joint(i).position_upper_limit(0))
#           self.robot_skeleton.joint(i).set_position_limit_enforced(True)
#           j = self.robot_skeleton.dof(i)
#           j.set_damping_coefficient(10.)
#           #j.set_spring_stiffness(2.)
#           #print("joint name",j.name)
#           #print("stiffness",j.spring_stiffness)        
#        for body in self.robot_skeleton.bodynodes+self.dart_world.skeletons[0].bodynodes:
#           body.set_friction_coeff(20.)
                        
            
            
            #this is humanoid to be helped up
            if ('getUpHumanoid' in skelName) or ('biped' in skelName) :
                self.skelHldrs.append(self._initLoadedSkel(skel, idx, True, False, idxSkelHldrs))
                self.humanIdx = idxSkelHldrs
                idxSkelHldrs += 1
                skelType = 'Human'
                #track getup humanoid in rendering
                self.track_skeleton_id = idx
            elif 'helperBot' in skelName :
                self.skelHldrs.append(self._initLoadedSkel(skel, idx, False, True, idxSkelHldrs))
                self.botIdx = idxSkelHldrs
                idxSkelHldrs += 1
                skelType = 'Robot'
                self.hasHelperBot = True
            elif 'helperBotArm' in skelName :
                self.skelHldrs.append(self._initLoadedSkel(skel, idx, False, False, idxSkelHldrs))
                self.botIdx = idxSkelHldrs
                idxSkelHldrs += 1
                skelType = 'Robot'
                self.hasHelperBot = True    
            elif 'sphere_skel' in skelName :
                self.grabLink = skel
                skelType = 'Sphere'
                self.grabLink.bodynodes[0].set_collidable(False)

#            print('{} index : {} #dofs {} #nodes {} root loc : {}'.format(skelType,idx, skel.ndofs, skel.nbodies, skel.states()[:6]))
        #give robot a ref to human skel handler
        if (self.hasHelperBot):
            self.skelHldrs[self.botIdx].setHelpedSkelH(self.skelHldrs[self.humanIdx])
        
        

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    #return jacobians of either robot or human contact point
    #for debugging purposes to examine jacboian
    def getOptVars(self, useRobot=True):
        if(useRobot):
            skelhldr = self.skelHldrs[self.botIdx]
        else:
            skelhldr = self.skelHldrs[self.humanIdx]
        
        return skelhldr.getOptVars()
    
    # methods to override:
    # ----------------------------
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------
    def set_state(self, qpos, qvel):
        self.activeSkelHndlr.set_state(qpos, qvel)

    def set_state_vector(self, state):
        self.activeSkelHndlr.set_state_vector(state)        
        
    def state_vector(self):
        return self.activeSkelHndlr.state_vector()


    @property
    def dt(self):
        return self.dart_world.dt * self.frame_skip
    
    def do_simPerturb(self, n_frames):
        self.checkPerturb()
        #apply their respective torques to all skeletons
        for fr in range(n_frames):
            for i in range(len(self.skelHldrs)) :
                self.skelHldrs[i].add_perturbation( self.perturbation_parameters[2], self.perturb_force)  
                #tau is set in calling step routine
                self.skelHldrs[i].applyTau()  
                
            self.dart_world.step()
#            #check to see if sim is broken each frame, if so, return with broken flag, frame # when broken, and skel causing break
            chk,resDict = self.checkWorldStep(fr)
            if(chk):
                return resDict
                
        return {'broken':False, 'frame':n_frames, 'skelhndlr':None}    
    #need to perform on all skeletons that are being simulated
    def do_simulation(self, n_frames):
        if self.add_perturbation:
            return self.do_simPerturb(n_frames)
        
        for fr in range(n_frames):
            for i in range(len(self.skelHldrs)):
                #tau is set in calling step routine
                self.skelHldrs[i].applyTau()  
#            self.dart_world.skeletons[1].bodynodes[0].add_ext_force([0, self.dart_world.skeletons[1].mass()*9.8, 0])
#        print(self._get_obs())
            self.dart_world.step()
            #check to see if sim is broken each frame, if so, return with broken comment
            for i in range(len(self.skelHldrs)) :
                brk, chkSt = self.skelHldrs[i].checkSimIsBroken()
                if(brk):  #means sim is broken, end fwd sim                 
                    return {'broken':True, 'frame':fr}

        return {'broken':False, 'frame':n_frames}                      
            
            
    def checkPerturb(self):
        if self.perturbation_duration == 0:
            self.perturb_force *= 0
            if np.random.random() < self.perturbation_parameters[0]:
                axis_rand = np.random.randint(0, 2, 1)[0]
                direction_rand = np.random.randint(0, 2, 1)[0] * 2 - 1
                self.perturb_force[axis_rand] = direction_rand * self.perturbation_parameters[1]

        else:
            self.perturbation_duration -= 1
   #Take total moments, total force, return a suitable point of application of that force to provide that moment     
    #COP_tau = COPval cross COP_ttlFrc ==> need to constrain possible COPvals
    #so set COPval.y == 0 since we want the COP at the ground, and then solve eqs :      
    def calcCOPFromTauFrc(self,COP_tau, COP_ttlFrc):
        COPval = np.zeros(3)        
        COPval[0] = COP_tau[2]/COP_ttlFrc[1]
        COPval[2] = COP_tau[1]/COP_ttlFrc[0] + (COP_ttlFrc[2] * COPval[0]/COP_ttlFrc[0])
        return COPval            
            
    #return a dictionary arranged by skeleton, of 1 dictionary per body of contact info
    #the same contact might be referenced multiple times
    def getContactInfo(self):
        contacts = self.dart_world.collision_result.contacts
        #dictionary of skeleton-keyed body node colli
        cntInfoDict = {}
        for i in range(len(self.dart_world.skeletons)):
            cntInfoDict[self.dart_world.skeletons[i].name] = defaultdict(contactInfo)
        
        for contact in contacts:
            cntInfoDict[contact.bodynode1.skeleton.name][contact.bodynode1.name].addContact(contact,contact.bodynode1,contact.bodynode2)
            cntInfoDict[contact.bodynode2.skeleton.name][contact.bodynode2.name].addContact(contact,contact.bodynode2,contact.bodynode1)
            
        return cntInfoDict
            
    ######################       
    #rendering stuff
    
    def _render(self, mode='human', close=False):
        if not self.disableViewer:
            self._get_viewer().scene.tb.trans[0] = -self.dart_world.skeletons[self.track_skeleton_id].com()[0]*1
        if close:
            if self.viewer is not None:
                self._get_viewer().close()
                self.viewer = None
            return

        if mode == 'rgb_array':
            data = self._get_viewer().getFrame()
            return data
        elif mode == 'human':
            self._get_viewer().runSingleStep()

    def getViewer(self, sim, title=None):
        # glutInit(sys.argv)
        win = StaticGLUTWindow(sim, title)
        #Currently scene is defined for first person view. Check in pydart/gui/opnegl/kima_firstpersonvieww
        
        #rot 4th component is along the ground
        #rot 2nd componend moves the ground 
        #rot 3rd component moves ground clockwise
#        win.scene.add_camera(Trackball(theta=0.0, phi = 0.0, rot=[0, 0.0, 0, 0.8], trans=[-0.0, 0.5, 0], zoom=0.1), 'gym_camera')
#        win.scene.add_camera(Trackball( theta = 0.0, phi = 0.0, rot=[0.02, 0.71, -0.02, 0.71],
#                                      zoom=1.5), 'gym_camera')
#        win.scene.add_camera(Trackball(theta=0.0, phi = 0.0, rot=[0.02, 0.71, -0.02, 0.71], trans=[0.02, 0.09, 0.39],
#                                      zoom=0.1), 'gym_camera')
#        win.scene.set_camera(win.scene.num_cameras()-1)
#        win.scene = 
        # to add speed,
        if self._obs_type == 'image':
            win.run(self.screen_width, self.screen_height, _show_window=self.visualize)
        else:
            win.run(_show_window=self.visualize)
#        img = win.getGrayscale(128, 128)
#       scipy.misc.imsave('image/shot1.png', img)
        return win

    def _get_viewer(self):
        if self.viewer is None and not self.disableViewer:
            self.viewer = self.getViewer(self.dart_world)
            self.viewer_setup()
        return self.viewer
    
        #disable/enable viewer
    def setViewerDisabled(self, vwrDisabled):
         self.disableViewer = vwrDisabled


    ####################
    # "private" methods
    def _reset(self):
        self.perturbation_duration = 0
        ob = self.reset_model()
        return ob

    #load all seletons and sim world
    def loadAllSkels(self, model_paths, dt):
        # convert everything to fullpath
        full_paths = []
        for model_path in model_paths:
            if model_path.startswith("/"):
                fullpath = model_path
            else:
                fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
            if not path.exists(fullpath):
                raise IOError("File %s does not exist"%fullpath)
            full_paths.append(fullpath)

        if full_paths[0][-5:] == '.skel':
            self.dart_world = pydart.World(dt, full_paths[0])
        else:
            self.dart_world = pydart.World(dt)
            for fullpath in full_paths:
                self.dart_world.add_skeleton(fullpath)
                
#class to hold reference for all contact info for a particular skeleton's body node
#needs to be held in a collection indexed by body node, so that it doesn't get overwritten
#this class is a bit weird - doesn't have body node set in ctorbecause used in a default dict               
class contactInfo():
    def __init__(self) :
        self.ttlfrc = np.zeros(3)
        self.COPloc = np.zeros(3)
        #np 3 dim arrays for point location and force value
        self.cntctPt = list()
        self.cntctFrc = list()
        self.colBodies= list()
        self.body = None
    
    #if there's a contact with this body
    #thisBody is this node's body node - should not be able to change
    def addContact(self, contact, thisBody, otrBody):
        if(None == self.body) :
            self.body = thisBody
            self.skel = thisBody.skeleton
        elif (self.skel.name != thisBody.skeleton.name) or (self.body.name != thisBody.name):
            print('Error in contactInfo:addContact : attempting to reassign from skel {} body {} to new skel {} body {}'.format(self.skel.name, self.body.name,thisBody.skeleton.name, thisBody.name))
        self.colBodies.append(otrBody)
        self.cntctPt.append(np.copy(contact.point))
        self.cntctFrc.append(np.copy(contact.force))
        self.setCopLoc()
        
    #recalculate average location of contacts - called internally
    def setCopLoc(self):
        self.ttlfrc = np.zeros(3)
        ttlTau = np.zeros(3)
        self.COPloc = np.zeros(3)
        numPts = len(self.cntctPt)
        
        for i in range(numPts):
            self.ttlfrc += self.cntctFrc[i]
            ttlTau += np.cross(self.cntctPt[i],self.cntctFrc[i])
        #coploc is location of cop in world coords
        self.COPloc[0] = ttlTau[2]/self.ttlfrc[1]
        self.COPloc[2] = ttlTau[1]/self.ttlfrc[0] + (self.ttlfrc[2] * self.COPloc[0]/self.ttlfrc[0])
    
    #for optimization just want jacobian for each body
    def getCntctJacob(self):
        #want jacobian expressed in world frame, since forces are expressed in world frame
        Jtrans = np.transpose(self.body.world_jacobian(offset=self.body.to_local(self.COPloc)))
        return Jtrans        
        
    #This will get the contact force in generalized coordinates
    #using J(self.COPloc)_transpose * self.ttlfrc
    def getCntctTauGen(self):
        #want jacobian expressed in world frame, since forces are expressed in world frame
        Jtrans = self.getCntctJacob()
        #Jtrans is dofs x 6 : first 3 of 6 are angular components, 2nd 3 of 6 are linear
        frcVec = np.zeros(6)
        #no angular component
        frcVec[3:]=self.ttlfrc
        #numpy matrix * vector -> vector
        return Jtrans.dot(frcVec)

from abc import ABC, abstractmethod

#base convenience class holding relevant info and functions for a skeleton
class skelHolder(ABC):   
    #env is owning environment
    #skel is ref to skeleton
    #widx is index in world skel array
    #stIdx is starting index in force array for tau calc
    def __init__(self, env, skel, widx, stIdx, skelHldrIDX):
#        print("making skel : {}".format(skel.name))
        #ref to owning environment
        self.env = env
        #ref to skel object
        self.skel = skel
        #index in owning env's skeleton holder list
        self.skelHldrIDX = skelHldrIDX
        #index in world
        self.worldIdx = widx
        #start index for action application in tau array
        self.stIdx = stIdx
        self.initQPos = None
        self.initQVel = None
        #number of dofs - get rid of length calc every time needed
        self.ndofs = self.skel.ndofs

        #timestep
        self.timestep = env.dart_world.dt
        
        #state flags
        #use the mean state only if set to false, otherwise randomize initial state
        self.randomizeInitState = True
        #use these to show that we are holding preset initial states - only used if randomizeInitState is false
        self.loadedInitStateSet = False
        #desired force/torque has been set
        self.desForceTorqueSet = False
        #initial torques
        self.tau = np.zeros(self.skel.ndofs)
        #TODO force and torque dimensions get from env
        
        #+/- perturb amount of intial pose and vel for random start state
        self.poseDel = .005
        
        #set initial value of force/torque - to be overridden by calling env - temporary placeholder
        #val=(.3 * 9.8 * self.skel.mass())
        #self.setDesiredExtForce(np.array([val,val,0]))
        #hand reaching to other agent
        self.reach_hand = None
        self.cnstrntLoc = np.zeros(3)    
        #list of body node names for feet
        self.feetBodyNames = ['r-foot', 'l-foot']
    
#        for i in range(len(self.skel.joints)-1):
#           #print("joint:%d"%(i),skel.joint(i).name)
#           #print("position limit",skel.joint(i).position_upper_limit(0))
#           self.skel.joint(i).set_position_limit_enforced(True)
#           j = self.skel.dof(i)
#           j.set_damping_coefficient(2.)
#           j.set_spring_stiffness(2.)
           #print("joint name",j.name)
           #print("stiffness",j.spring_stiffness)

        for body in self.skel.bodynodes+self.env.unwrapped.dart_world.skeletons[0].bodynodes:
            body.set_friction_coeff(100.)
        self.desFrcTrqVal = [0,0,0]
        
    def setSkelMobile(self, val):
        self.skel.set_mobile(val)            
        
    def getReachCOM(self):
        return self.skel.body(self.reach_hand).com()
        
    #add a ball constraint between this skeleton's reaching hand and the passed body
    #the order component is the order to add the object
    def addBallConstraint(self,body1, bodyPos):
        body2 = self.skel.body(self.reach_hand)
        pos = .5*(bodyPos + body2.com())
        self.lclCnstLoc = body2.to_local(pos)
        print('Name : {} skelHandler ball world loc {} ball lcl loc : {} | cnst world loc : {} and lcl loc : {} on body {}'.format(self.skel.name,bodyPos,body2.to_local(bodyPos), pos,self.lclCnstLoc,body2.name))
        constraint = pydart.constraints.BallJointConstraint(body1, body2, pos)
        constraint.add_to_world(self.env.dart_world)

    #set the name of the body node reaching to other agent
    def setReachHand(self, _rchHand):
        self.reach_hand = _rchHand
    
    #set the desired external force and torque for this skel, to be applied at 
    #COM (TODO)
    #(either the force applied to the human, or the force being applied by the robot)
    def setDesiredExtForce(self, desFrcTrqVal):
        self.lenFrcVec = len(desFrcTrqVal)
        self.desFrcTrqVal = desFrcTrqVal
        self.desForceTorqueSet = True                
   
    #set control bounds for this skeleton
    def setCntlBnds(self, control_bounds):
        self.control_bounds = control_bounds
        self.action_dim = len(control_bounds[0])
        self.action_space = spaces.Box(control_bounds[1], control_bounds[0])
        
    #set action scaling for this skeleton
    def setActionScale(self, action_scale):
        self.action_scale = action_scale
    
    #set observation dimension
    def setObsDim(self, obsDim):
        self.obs_dim = obsDim
        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        
    #set what this skeleton's init pose should be
    def setInitPose(self, _ip):
        self.initPose = _ip
        self.setToInitPose()
#        self.postPoseInit()
    
    #reset this skeleton to its initial pose
    def setToInitPose(self):
        self.skel.set_positions(self.initPose)        
        
    #set initial state externally - call before reset_model is called by training process
    #Does not override initPose
    def setNewInitState(self, _qpos, _qvel):       
        #set to false to use specified states
        self.randomizeInitState = False
        self.initQPos = np.asarray(_qpos, dtype=np.float64)
        self.initQVel = np.asarray(_qvel, dtype=np.float64)
        self.loadedInitStateSet = True
                
    #set this skeleton's state
    def state_vector(self):        
        return np.concatenate([
            self.skel.q,
            self.skel.dq
        ])
    #sets skeleton state to be passed position and velocity
    def set_state(self, qpos, qvel):
#        print(qpos.shape)
#        print(qpos)
        #assert shouldnt be there
#        assert qpos.shape == (self.skel.ndofs,) and qvel.shape == (self.skel.ndofs,)
        self.skel.set_positions(qpos)

        self.skel.set_velocities(qvel)

    #sets skeleton state to be passed state vector, split in half (for external use)
    def x(self, state):
        numVals = int(len(state)/2.0)
        self.skel.set_positions(state[0:numVals])
        self.skel.set_velocities(state[numVals:])  

    #called in do_simulate if perturbation is true
    def add_perturbation(self, nodes, frc):
        self.skel.bodynodes[nodes].add_ext_force(frc)        
                 
    #build tau from a, using control clamping, for skel, starting at stIdx
    #and action scale for skeleton
    def setTau(self, a):
        self.a = a
        clamped_control = np.array(a)
        for i in range(len(clamped_control)):
            if clamped_control[i] > self.control_bounds[0][i]:
                clamped_control[i] = self.control_bounds[0][i]
            if clamped_control[i] < self.control_bounds[1][i]:
                clamped_control[i] = self.control_bounds[1][i]
        self.tau = np.zeros(self.skel.ndofs)
        
        self.tau[self.stIdx:] = clamped_control * self.action_scale 
        # print(self.tau)
        return self.tau              
    #get dictionary of optimization variables at constraint location (where applicable)
    #debug function
    def getOptVars(self):
        res = {}
        body=self.skel.body(self.reach_hand)
        res['M']=self.skel.M
        res['CfrcG']=self.skel.coriolis_and_gravity_forces()
        res['jacobian']=body.jacobian(offset=self.cnstrntLoc)
        res['world_jacobian']=body.world_jacobian(offset=self.cnstrntLoc)
        res['linear_jacobian']=body.linear_jacobian(offset=self.cnstrntLoc)
        res['angular_jacobian']=body.angular_jacobian()       
        
        return res   
    #for some reason this is called after perturbation is set
    #would be better if called from setTau
    def applyTau(self):
        pass



    #return a random initial state for this skeleton
    def getRandomInitState(self, poseDel=None):
        if(poseDel is None):
            poseDel=self.poseDel
        #set walker to be laying on ground
        self.setToInitPose()
        #perturb init state and statedot
        qpos = self.skel.q + self.env.np_random.uniform(low=-poseDel, high=poseDel, size=self.skel.ndofs)
        qvel = self.skel.dq + self.env.np_random.uniform(low=-poseDel, high=poseDel, size=self.skel.ndofs)
        return qpos, qvel
      
        
    #calculate orientation along certain orientation axes
    def procOrient(self, orientAxes):
        oVec = np.array(orientAxes)
        oVec_W = self.skel.bodynodes[0].to_world(oVec) - self.skel.bodynodes[0].to_world(np.array([0, 0, 0]))
        norm = np.linalg.norm(oVec_W)
        if(norm == 0):#should never happen, since this is used as a marker of failing, a large value will signal done
            return 10
        oVec_W /= norm
        ang_cos = np.arccos(np.dot(oVec, oVec_W))
        return ang_cos


    def dispResetDebug(self, notice=''):
        print('{} Notice : Setting specified init q/qdot/frc'.format(notice))
        print('initQPos : {}'.format(self.initQPos))
        print('initQVel : {}'.format(self.initQVel))
#        print('initFrc : {}'.format(self.assistForce))
        
    #for external use only - return observation list given passed state and state dots
    #obs is slightly different than pure q/qdot (includes height in world frame), requiring skel to be modified
    #restores skel pose when finished - make sure q is correctly configured
    def getObsFromState(self, q, qdot):
        #save current state so can be restored
        oldState = self.state_vector()
        oldQ = self.skel.q
        oldQdot = self.skel.dq
        #set passed state
        self.set_state(np.asarray(q, dtype=np.float64), np.asarray(qdot, dtype=np.float64))
        #get obs - INCLUDES FORCE VALUE - if using to build new force value, need to replace last 3 elements
        obs = self.getObs()        
        #return to original state
        self.set_state(oldQ, oldQdot)
        return obs  
    
    #called at beginning of each rollout - resets this model, resetting its state
    def reset_model(self, dispDebug=False):  
#        print('Randomize state', self.randomizeInitState)
#        print('Reset')         
        self.randomizeInitState=True
        if(self.randomizeInitState):#if random, set random perturbation from initial pose
            qpos, qvel = self.getRandomInitState()

            self.set_state(qpos, qvel)
        else:
            #reset to be in initial pose
            self.setToInitPose()
#            print('Loaded state',self.loadedInitStateSet)
#            input()
            #resetting to pre-set initial pose
            if ( self.loadedInitStateSet):
                if(dispDebug):
                    self.dispResetDebug('skelHolder::reset_model')
                self.set_state(self.initQPos, self.initQVel)
                self.loadedInitStateSet = False
            else:
                print('skelHolder::reset_model Warning : init skel state not randomized nor set to precalced random state')

        return self.getObs()

    
    #init to be called after skeleton pose is set
    @abstractmethod
    def postPoseInit(self):
        pass
    #functionality necessary before simulation step is executed
#    @abstractmethod
#    def preStep(self, a, cnstLoc=np.zeros(3)):
#        pass        
   
    @abstractmethod
    def resetIndiv(self, dispDebug):
        pass   
    
   
    
    #functionality after sim step is taken - 
    #calculate reward, determine if done(temrination conditions) and return observation
    #and return informational dictionary
    def postStep(self, resDict):
#        vx, vy, vz, rwd, done, d = self.calcRewardAndCheckDone(resDict)
        rwd, done, d = self.calcRewardAndCheckDone(resDict)
        #want this to be cleared for every non-broken iteration or if it is broken but also done
        if ((d['broke_sim'] == False) or done):
            self.numBrokenIters = 0
        obs = self.getObs()
        #ob, reward, done, infoDict
#        return vx, vy, vz, obs, rwd, done, d
        return obs, rwd, done, d
    
    #return skeleton qdot - maybe clipped, maybe not
    def getSkelqDot(self):
        return np.clip(self.skel.dq, -10, 10)
    
    #base check goal functionality - this should be same for all agents,
    #access by super()
    def checkSimIsBroken(self):
        s = self.state_vector()
        _broken = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() )
        return _broken, s
    
    #check body node name to see if part of foot
    def checkBNFoot(self, name):
        return ('foot' in name)
    
    #check if passed body nodes are on two different, non-ground, skeletons - return true
    #don't want this - skeletons should not contact each other except through the ball joint constraint
    def checkBNKickOtr(self, bn1, bn2):
        if ('ground' in bn1.name) or ('ground' in bn2.name):
            return False
        #if != then they are different skeletons
        return (bn1.skel.name != bn2.skel.name)
    
    #returns true only if one body node is a foot on this skeleton and the other is the ground in a contact
    def checkMyFootWithGround(self, bn1, bn2):
        return (('ground' in bn1.name) and self.checkBNFoot(bn2.name) and (self.skel.name == bn2.skel.name)) or \
                (('ground' in bn2.name) and self.checkBNFoot(bn1.name) and (self.skel.name == bn1.skel.name)) 
    
       
      
      
    #calculate foot contact count and other terms if we want to use them for reward calc
    def calcFootContactRew(self):
        contacts = self.env.dart_world.collision_result.contacts        
        contactDict = defaultdict(float)
        #sum of foot contact forces in 3 dirs
        contactDict['cFrcX'] = 0
        contactDict['cFrcY'] = 0
        contactDict['cFrcZ'] = 0
        
        COPval = np.zeros(3)
        COP_tau = np.zeros(3)
        COP_ttlFrc = np.zeros(3)
        #tau = loc x frc
        #COP calculation is the location that, when crossed with total force, will give total moment
        #we can calculate total moment by crossing all contact forces with all contact locations
        #we can get total force, and from this we can find the location that would produce the total 
        #torque given the total force (by first constraining one of the dimensions of the point in question)
        #we choose to constrain the y coordinate to be 0 since we want the cop on the ground
       
        for contact in contacts:
            if (self.skel.name != contact.bodynode1.skeleton.name ) and (self.skel.name != contact.bodynode2.skeleton.name ) :
                #contact not from this skeleton
                continue
            #penalize contact between the two skeletons - getup-human should not touch assistant bot
            #only true if one skel contacts other
            if self.checkBNKickOtr(contact.bodynode1, contact.bodynode2):
                contactDict['kickBotContacts'] +=1            
            #only true if feet are contacting skeleton - kicking self
            if (contact.bodynode1.skel.name == contact.bodynode2.skel.name):
                contactDict['tripFeetContact'] +=1
            #this is a foot contact
            #if (self.checkBNFoot(contact.bodynode2.name) or self.checkBNFoot(contact.bodynode1.name)):#ground is usually body 1
            if (self.checkMyFootWithGround(contact.bodynode1,contact.bodynode2)):
                #print('With Ground : contact body 1 : {} skel : {} | contact body 2 : {} skel : {}'.format(contact.bodynode1,contact.bodynode1.skeleton.name, contact.bodynode2,contact.bodynode2.skeleton.name))
                contactDict['footGroundContacts'] +=1
                #find total moment of all contacts
                COP_tau += np.cross(contact.point, contact.force)
                COP_ttlFrc += contact.force
                if ('left' in contact.bodynode2.name) or ('left' in contact.bodynode1.name):
                    contactDict['leftContacts']+=1                       
                else:
                    contactDict['rightContacts']+=1
                    
                contactDict['cFrcX'] += contact.force[0]
                contactDict['cFrcY'] += contact.force[1]
                contactDict['cFrcZ'] += contact.force[2]
            else:
                #print('Not With Ground : contact body 1 : {} skel : {} | contact body 2 : {} skel : {}'.format(contact.bodynode1,contact.bodynode1.skeleton.name, contact.bodynode2,contact.bodynode2.skeleton.name))
                contactDict['nonFootContacts']+=1
        
        #determines COP based on foot contacts with ground
        if(0 < contactDict['footGroundContacts']):
            #COP_tau = COPval cross COP_ttlFrc ==> need to constrain possible COPvals -> set COPval.y == 0 since we want the COP at the ground, and then solve eqs : 
            COPval = self.env.calcCOPFromTauFrc(COP_tau, COP_ttlFrc)
        else :  #estimate COP as center of both feet body node com locations         
            COPval = np.zeros(3)
            COPval += self.skel.body('r-foot').com()
            COPval += self.skel.body('l-foot').com()
  
            COPval /= 2.0

        return contactDict, COPval

    
    #this will calculate a bounded velocity as a reward 
    #passed are the velocities that get max reward, == the max reward given 
    # and an offset equivalent to 
    #the minimum value to get a positive reward
    #shaped like an inverted parabola
    def calcVelRwd(self, vel, vMaxRwd, minVel, maxRwd):
        #return (-abs(vel-(vMaxrwd + minVel)) + maxRwd)
        a = maxRwd/(vMaxRwd * vMaxRwd)
        cval = (vel-(vMaxRwd + minVel))        
        return (-a *(cval * cval) + maxRwd)
        
   
    #get the state observation from this skeleton - concatenate to 
    #whatever extra info we are sending as observation
    @abstractmethod
    def getObs(self):
        pass
    
    #calculate reward for this agent, see if it is done, and return informational dictionary (holding components of reward for example)
    @abstractmethod
    def calcRewardAndCheckDone(self,resDict):
        pass
    

                 
#class for skeleton holder specifically for the getup human
class humanSkelHolder(skelHolder):
                 
    def __init__(self, env, skel, widx, stIdx, skelHldrIDX):          
        skelHolder.__init__(self,env, skel,widx,stIdx, skelHldrIDX)
        #set this so that bot doesn't get stuck in limbo
        self.minUpVel = .001
        #must set force    
    
    #called after pose is set
    def postPoseInit(self):
        pass
    def getObs(self):
        state =  np.concatenate([
            self.skel.q,
            self.skel.dq,#self.getSkelqDot(),
            #need force as part of observation!
            self.desFrcTrqVal
        ])
        #assign COM to state
        state[3:6] = self.skel.com()
        return state
    
    def applyTau(self):
        self.skel.body(self.reach_hand).set_ext_force(self.desFrcTrqVal)
        self.skel.set_forces(self.tau)   
    #get-up skel's individual per-reset settings
    def resetIndiv(self, dispDebug):
        #initial copcom distance - want this to always get smaller
        #set initial COMCOPDist for velocity calcs
        print('In COMCOP')
        input()
        self.COMCOPDist = self.calcCOMCOPDist()

    #functionality necessary before simulation step is executed for the human needing assistance
    def preStep(self, a):
        #self.lenFrcVec = len(desFrcTrqVal)
        #self.desFrcTrqVal = desFrcTrqVal
        #just add force at reach hand's COM
        self.skel.body(self.reach_hand).add_ext_force(self.desFrcTrqVal)
        #self.skel.body('l-lowerarm').add_ext_force(self.desFrcTrqVal)
        #get x position and height before forward sim
        com = self.skel.body('head').com()
        self.posBefore = com[0]        
        self.heightBefore = com[1] 
        self.sideBefore = com[2]
        #state before sim
        self.oldState = self.state_vector()
                     
        #set torques
        self.tau=self.setTau(a)
    
    def calcCOMCOPDist(self, COPVal=np.zeros(3)):
        if(np.count_nonzero(COPVal) == 0):
            #COPVal will either be center of foot contacts with ground, or if no foot contacts then center of feet node COMs
            _,COPVal = self.calcFootContactRew()
        COMatFeet = self.skel.com()
        COMatFeet[1] = COPVal[1]        
        #distance between COM projected on ground and COP value is bad
        return np.square(COMatFeet - COPVal).sum()**(.5)
        
    #calculate distance of COM projection at feet to center of feet
    #and velocity of convergence
    def calcCOMCOPRew(self,COPVal):
        COMCOPDist = self.calcCOMCOPDist(COPVal)
        #new dist needs to be smaller than old dist - this needs to be positive
        COMCOP_vel = (self.COMCOPDist - COMCOPDist)  / self.env.dt  
        #bounding convergence speed so we don't get the baseline diverging
        #vel, vMaxRwd, minVel, maxRwd
        COMCOP_VelScore = self.calcVelRwd(COMCOP_vel, 1.5, self.minUpVel, 5.0)
        
        #keep around for next timestep
        self.COMCOPDist = COMCOPDist
        
        return COMCOP_VelScore    


    #calculate reward for this agent, see if it is done, and return informational dictionary (holding components of reward for example)
    #called after fwd sim step
    def calcRewardAndCheckDone(self,resDict):
        #resDict holds whether the sim was broken or not - if sim breaks, we need 
        #holds these values : {'broken':False, 'frame':n_frames, 'stableStates':stblState}   
        #check first if sim is broken - illegal actions or otherwise exploding
        
        broke_sim = resDict['broken']
        #s = self.state_vector()
        #forceLocAfter = self.skel.body(self.reach_hand).to_world(self.frcOffset)
        #get x position and height before forward sim                   
        com = self.skel.body('head').com()
        #print(com)
        posAfter = com[0]        
        heightAfter = com[1]  
        sideVel = (com[2] - self.sideBefore)/self.env.dt
        #tight
#        sideVelScore = -1/0.2*(sideVel)**2+0.2
        #medium bound
        sideVelScore = -1/1*(sideVel)**2+1
#        sideVelScore = -abs(sideVel)+1
#        sideVelScore = float(1e-2*(sideVel))

        #angular terms : rotating around z, y and x axis
#            ang_cos_swd = self.procOrient([0, 0, 1])
#            ang_cos_uwd = self.procOrient([0, 1, 0])
#            ang_cos_fwd = self.procOrient([1, 0, 0])
#                  
           
        # reward function calculation
        alive_bonus = 1.0
        vel = (posAfter - self.posBefore) / self.env.dt

#        velScore = -(vel)**2 + 1.0
        #medium
        valX = 1.5
        velScore = -1/valX*(vel)**2+valX
#        velScore = -abs(vel-2) + 2
        raiseVel = (heightAfter - self.heightBefore) / self.env.dt
        
        #Keeping raiseVel as square is breaking the same. Goes negative very soon
#        raiseVelScore = -abs(raiseVel - (2+ self.minUpVel)) + 2
        #peak centered at velocity == 2 + self.minUpVel, peak is 2 high, so spans self.minUpVel to 4+sel.minUpVel
#        valY = 1.2/2
        valY = 1.75/2
        raiseVelScore = -1/valY*(raiseVel-(valY+ self.minUpVel))**2 + valY + self.minUpVel
        #ground is at -0.5
        height_rew = heightAfter + 0.5
        
        contacts = self.env.dart_world.collision_result.contacts
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

        
        
        #minimize requested torques- should this be requested or actually used? TODO
        actionPenalty = float(1e-1* np.square(self.a).sum())

        #reward = alive_bonus + raiseVelScore + velScore + contactRew + height_rew - actionPenalty #- angVelPen #- contactPen
        reward = raiseVelScore + velScore + float(1e1* (height_rew)) + float(footContacts) - actionPenalty - 1e-1*sideVelScore#- angVelPen #- contactPen
#        print('Upward Velocity: {}, Forward velocity:{} sideL{} reward{} '.format(raiseVel, vel, sideVel, reward))
        s = self.state_vector()
        #if broke_sim:
        #    reward = reward - 10
#        print(vel, velScore)
        done = broke_sim or not(
                np.isfinite(s).all() and 
                #(np.abs(s[2:]) < 100).all() and
                raiseVelScore>self.minUpVel and  #NOT 0
                #maybe not good, need to check
#                    (abs(ang_cos_uwd) < np.pi) and 
#                    (abs(ang_cos_fwd) < np.pi) and
#                    (abs(ang_cos_swd) < np.pi) and 
                (heightAfter>-0.4) and
                          
                (heightAfter < 0.61) and
                velScore > 0 and # allowing for slight negative velocities
                sideVelScore > 0 # Not allowing more deviation
                
                
            )
#        print(self.desFrcTrqVal)
#        input()
        #If Force values are extreme, check the returns
#        if(self.desFrcTrqVal[0]>365 and self.desFrcTrqVal[1]>360):
#        print("Both forces are high and reward is", self.desFrcTrqVal, reward)
#            input()
        
#        if done:
#            
#            if(broke_sim):
#                print('Broken Sim')
#            elif(not np.isfinite(s).all()):
#                print('Nans')
#            elif(not(heightAfter>-0.4)):
#                print('Height too less')
#            elif(not(heightAfter < 0.61)):
#                print('Height too much')
#            elif(not raiseVelScore>self.minUpVel):
#                print('Raise Vel negative')
#            elif(not sideVelScore>0):
#                print('Side velocity unbounded')
#            elif (not velScore > 0 ):
#                print(vel)
#                print('Forward velocity unbounded')
#            else:
#                print(raiseVel )
#                print('Something else')
#                input()
         
#        print('humanSkelHolder : heightAfter : {}'.format(heightAfter))
#        if heightAfter>0.57:
#            print('humanSkelHolder : heightAfter : {}'.format(heightAfter))
#            input()
        
        #info
        dct = {'broke_sim': False,'vy': raiseVel, 'raiseVelScore': raiseVelScore, 'vx': vel, 'velScore': velScore, 'vz': sideVel, 'height_rew':height_rew,
                'actionPenalty':actionPenalty, 'is_done': done}
        
        return reward, done, dct
    
    
                 
                 
#class to hold assisting robot
class robotSkelHolder(skelHolder):
                 
    def __init__(self, env, skel, widx, stIdx, skelHldrIDX):          
        skelHolder.__init__(self,env, skel,widx,stIdx, skelHldrIDX)
        #the holder of the human to be helped - 
        self.helpedSkelH = None
        #this skeleton is not mobile TODO remove this once we have a policy for him
        self.skel.set_mobile(False)
        
    #called after initial pose is set
    def postPoseInit(self):
        self.initOpt()    
    
     #set bounds for optimizer, if necessary/useful
    #TODO see if this will help, probably algorithm dependent
    #idxAra : idxs of y component in fcntct part of bounds array(needs to be offset by ndofs)
    def initOptBoundCnstrnts(self, idxAra, n):
        #qdotPrime = first ndofs
        #fcntc = next 12
        #tau = last ndofs
        #end idx of contact forces
        idxFend = (self.ndofs+12)
        lbndVec = np.zeros(n)
        ubndVec = np.zeros(n)
        #set accel min and max
        lbndVec[0:self.ndofs] = -30.0
        ubndVec[0:self.ndofs] = 30.0
        #set contact force lower and upper bounds
        lbndVec[self.ndofs:idxFend] = -100.0
        #set lowerbound for y component to be 0
        offsetYIdxs =(idxAra + self.ndofs)
        lbndVec[offsetYIdxs] = 0
        ubndVec[self.ndofs:idxFend] = 100.0
        #set min/max torque limits as in program
        lbndVec[idxFend:] = self.torqueLims * -1
        ubndVec[idxFend:] = self.torqueLims
        #set min bounds
        self.optimizer.set_lower_bounds(lbndVec)
        #set max bounds
        self.optimizer.set_upper_bounds(ubndVec)        
        
    #initialize nlopt optimizer variables
    def initOpt(self):
        # n is dim of optimization parameters/decision vars-> 2 * ndofs + 12
        n=2*self.ndofs + 12
        #index array of locations in fcntct array of y component of contact force - should always be positive
        y_idxAra = np.array([1,4,7,10])
        #idx aras to isolate variables in constraint and optimization functions
        self.qdotIDXs = np.arange(self.ndofs)
        self.fcntctIDXs = np.arange(self.ndofs,(self.ndofs+12))
        self.tauIDXs = np.arange((self.ndofs+12), n)
        
        #helper values - always calculate 1 time
        #always constant - calc 1 time
        
        #dof idxs to be used for pose matching optimization calculations
        #ignore waist, and reach hand dofs and root location
        #root orientation 0,1,2; root location 3,4,5
        #left thigh 6,7,8 : 6 is bend toward chest, 7 is twist along thigh axis, 8 is spread
        #left shin, left heel(2), left toe 9,10,11,12
        #right thigh ,13,14,15: 13 is bend toward chest, 14 is twist along thigh axis, 15 is spread
        #right shin, right heel(2), right toe 16,17,18,19
        #abdoment(2), spine 20,21,22; head 23,24
        #scap left, bicep left(3) 25,26,27,28 ; forearm left, hand left 29,30
        #scap right, bicep right (3) 31,32,33,34 ; forearm right.,hand right 35,36
        if('h_hand_right' in self.reach_hand) :
            self.optPoseUseIDXs = np.array([0,1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,23,24,25,26,27,28])
        else :#left hand reaching, match pose of right hand
            self.optPoseUseIDXs = np.array([0,1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,23,24,31,32,33,34])
            
        #this is pose q of dofs we wish to match
        self.matchPose = self.initPose[(self.optPoseUseIDXs)]
               
        #TODO magic number, no idea if this is correct choice
        self.kPose = 100
        self.tSqrtKpose = 2 * sqrt(self.kPose)

        #tau filter- force first 6 dofs to 0
        rootZeros = np.zeros(6)
        tauOnes = np.ones(self.ndofs-6)
        self.tauFltr = np.concatenate([rootZeros,tauOnes])
        #torqueLimits
        print('{}'.format(self.action_scale))
        self.torqueLims = np.concatenate([rootZeros,self.action_scale])

        #up vector for all 4 contact forces
        self.fcntctUp = np.zeros(12)
        self.fcntctUp[y_idxAra] = 1

        #for derivs
        self.zeroA = np.zeros(self.ndofs)
        self.oneA = np.ones(self.ndofs)
        self.oneAOvTS = self.oneA/self.timestep
        
        self.zeroTau = np.zeros(self.ndofs)
        #negative because used in MA equality constraint
        self.tauDot = np.ones(self.ndofs) * -1
     
        self.zeroFrict = np.zeros(12)
        
#        #derive of contact force eq - setting to negative since difference from cone val
        self.fcntctDot = np.ones(12)
        self.fcntctDot[y_idxAra] *= -1
        
        
        #create optimizer
        #only LD_MMA and LD_SLSQP support nonlinear inequality constraints
        #only LD_SLSQP supports nonlinear equality cosntraints
        self.optimizer = nlopt.opt(nlopt.LN_COBYLA, n)  #NLOPT_LN_COBYLA
        #set bounds - not sure if needed/wise
        self.initOptBoundCnstrnts(y_idxAra,n)
        #set initial guess to be all ones TODO better choice?
        self.nextGuess= np.ones(n)
        #set root dof initial guess to be 0
        stRoot=(self.ndofs+12)
        self.nextGuess[stRoot:(stRoot+6)] = np.zeros(6)
        
        #tolerances
        self.cnstTol = np.ones(n) * 1e-8
        
    def applyTau(self):
        self.skel.set_forces(self.tau)       
        
    #this is the skelHolder for the assisting robot, so set the human to be nonNull
    def setHelpedSkelH(self, skelH):
        self.helpedSkelH = skelH
        #reset observation dim to match this skel and human skel's combined obs
        self.setObsDim(self.obs_dim + skelH.obs_dim)        
        
    #robot uses combined human and robot state, along with force observation
    def getObs(self):
        stateHuman = self.helpedSkelH.getObs()
        #stateHuman has current force as part of observation, need to replace with local desired force TODO       
        
        
        state =  np.concatenate([
            self.skel.q,
            self.getSkelqDot(),
            stateHuman,
        ])            
        
        #assign COM to observation
        state[3:6] = self.skel.com()
        return state

       #assistant robot skel's individual per-reset settings
    def resetIndiv(self, dispDebug):
#        print('bot head height : {} '.format(self.skel.body('h_head').com()[1] ))
#        print('bot foot height : {} '.format(self.skel.body('h_heel_left').com()[1] ))
#        print('bot com  : {} '.format(self.skel.com() ))
        pass    

    #functionality before sim step is executed on this skel
    def preStep(self, a):
#        self.lenFrcVec = len(desFrcTrqVal)
        #self.desFrcTrqVal = [-1] * self.helpedSkelH.desFrcTrqVal
        #self.skel.body(self.reach_hand).add_ext_force(self.desFrcTrqVal)
        #get x position and height before forward sim
        com = self.skel.com()
        self.posBefore = com[0]        
        self.heightBefore = com[1] 
        self.sideBefore = com[2]
        #state before sim
        self.oldState = self.state_vector()
        
        self.tau=self.setTau(a)

    #calculate reward for this agent, see if it is done, and return informational dictionary (holding components of reward for example)
    def calcRewardAndCheckDone(self,resDict):        
        #resDict holds whether the sim was broken or not - if sim breaks, we need 
        #holds these values : {'broken':False, 'frame':n_frames, 'stableStates':stblState}   
        #check first if sim is broken - illegal actions or otherwise exploding
        
        broke_sim = resDict['broken']
        com = self.skel.com()
        posAfter = com[0]        
        heightAfter = com[1]        
        
        sideVel = (com[2] - self.sideBefore)/self.env.dt
        #angular terms : rotating around z, y and x axis
#            ang_cos_swd = self.procOrient([0, 0, 1])
#            ang_cos_uwd = self.procOrient([0, 1, 0])
#            ang_cos_fwd = self.procOrient([1, 0, 0])
#                  
           
        # reward function calculation
        alive_bonus = 1.0
        vel = (posAfter - self.posBefore) / self.env.dt
        velScore = -abs(vel-2)+2
        raiseVel = (heightAfter - self.heightBefore) / self.env.dt
        #peak centered at velocity == 2 + self.minUpVel, peak is 2 high, so spans self.minUpVel to 4+sel.minUpVel
        raiseVelScore = -abs(raiseVel-(2+ self.minUpVel)) + 2

        height_rew = float(10 * heightAfter)

        #minimize requested torques- should this be requested or actually used? TODO
        actionPenalty = float(1e-3 * np.square(self.a).sum())

        reward = alive_bonus - actionPenalty 
        
        done = broke_sim or not(
                (raiseVelScore >= 0) and
                #maybe not good, need to check
#                    (abs(ang_cos_uwd) < np.pi) and 
#                    (abs(ang_cos_fwd) < np.pi) and
#                    (abs(ang_cos_swd) < np.pi) and                   
                (heightAfter < 1.7) 
            )       
        #print('RobotSkelHolder : heightAfter : {}'.format(heightAfter))
            
        dct = {'broke_sim': broke_sim, 'raiseVelScore': raiseVelScore, 'height_rew':height_rew,
                    'actionPenalty':actionPenalty, 'is_done': done}   
        return vel, raiseVel, sideVel, reward, done, dct
    
##class to hold assisting robot arm (not full body biped robot)
class robotArmSkelHolder(skelHolder):
                 
    def __init__(self, env, skel, widx, stIdx, skelHldrIDX):          
        skelHolder.__init__(self,env, skel,widx,stIdx, skelHldrIDX)
        #the holder of the human to be helped - 
        self.helpedSkelH = None
        

    #called after initial pose is set
    def postPoseInit(self):
        self.initOpt()
        
    
    #set bounds for optimizer, if necessary/useful
    #TODO see if this will help, probably algorithm dependent
    #no contacts forces
    def initOptBoundCnstrnts(self, n):
        lbndVec = np.zeros(n)
        ubndVec = np.zeros(n)
        #TODO 
        #set min bounds
        self.optimizer.set_lower_bounds(lbndVec)
        #set max bounds
        self.optimizer.set_upper_bounds(ubndVec)        
        
    #initialize nlopt optimizer variables
    def initOpt(self):
        # n is dim of optimization parameters/decision vars-> 2 * ndofs + 12
        n=2*self.ndofs

        #idx aras to isolate variables in constraint and optimization functions
        self.accelIDXs = np.arange(self.ndofs)        
        self.tauIDXs = np.arange(self.ndofs, n)
        
        #helper values - always calculate 1 time
        #always constant - calc 1 time
         
        #TODO magic number, no idea if this is correct choice
        self.kPose = 100
        self.tSqrtKpose = 2 * sqrt(self.kPose)

        #tau filter- force first 6 dofs to 0
        rootZeros = np.zeros(6)
        tauOnes = np.ones(self.ndofs-6)
        self.tauFltr = np.concatenate([rootZeros,tauOnes])
        #torqueLimits
        #print('{}'.format(self.action_scale))
        self.torqueLims = np.concatenate([rootZeros,self.action_scale])

        #for derivs
        self.zeroA = np.zeros(self.ndofs)
        self.oneA = np.ones(self.ndofs)
        
        self.zeroTau = np.zeros(self.ndofs)
        self.tauDot = np.ones(self.ndofs) * -1
    
        
        #create optimizer
        self.optimizer = nlopt.opt(nlopt.LD_SLSQP, n)
        #set bounds - not sure if needed/wise
        self.initOptBoundCnstrnts(n)
        
        
        
    #this is the skelHolder for the assisting robot, so set the human to be nonNull
    def setHelpedSkelH(self, skelH):
        self.helpedSkelH = skelH
        #reset observation dim to match this skel and human skel's combined obs
        self.setObsDim(self.obs_dim + skelH.obs_dim)        
        
    #robot uses combined human and robot state, along with force observation
    def getObs(self):
        stateHuman = self.helpedSkelH.getObs()
        #stateHuman has current force as part of observation, need to replace with local desired force TODO    
        state =  np.concatenate([
            self.skel.q,
            self.skel.dq,#self.getSkelqDot(),
            stateHuman,
        ])            
        
        #assign COM to observation
        state[3:6] = self.skel.com()
        return state

    #assistant robot skel's individual per-reset settings
    def resetIndiv(self, dispDebug):
#        print('bot head height : {} '.format(self.skel.body('h_head').com()[1] ))
#        print('bot foot height : {} '.format(self.skel.body('h_heel_left').com()[1] ))
#        print('bot com  : {} '.format(self.skel.com() ))
        pass

        
    #return body torques to provide self.desFrcTrqVal at toWorld(self.constraintLoc)
    #provides JtransFpull component of equation
    def getPullTau(self, reachBody):
        #self.desFrcTrqVal
        wrldTrqFrc = np.zeros(6)
        #TODO need to manage when self.desFrcTrqVal has 6 vals
        wrldTrqFrc[3:]=self.desFrcTrqVal
        #find jacobian transpose
        Jtrans = np.transpose(reachBody.world_jacobian(offset=self.cnstrntLoc))
        return Jtrans.dot(wrldTrqFrc)
    
    #initialize these values every time step - as per Abe' paper they are constant per timestep
    def setSimVals(self):
        #Mass Matrix == self.skel.M
        #M is dof x dof 
        self.M = self.skel.M
        #CfG == C(q,dq) + G; it is dof x 1 vector
        self.CfG = self.skel.coriolis_and_gravity_forces()
        #torque cntrol desired to provide pulling force at contact location        
        self.Tau_JtFpull = self.getPullTau(self.skel.body(self.reach_hand))
        #pose jacobian - all pose elements?
        #this is x in paper - world location of all dofs? bodynodes?
        #use q/qdot
        self.curMatchPose = self.skel.q[(self.optPoseUseIDXs)]
        self.curMatchPoseDot = self.skel.qdot[(self.optPoseUseIDXs)]
    
    
    #def getObjAccelDiff(self, )
    
    #returns scalar value
    #grad is dim n
    def objectiveFunc(self, x, grad):
        a = x[self.accelIDXs]
        tau = x[self.tauIDXs]
        
        #diff_a = world_accel - desired_accel
        #TODO calculate a from xdd and desired accel
        #pose can be matched to rest pose for all dofs except reaching hand and waist
        #ignore dofs for reach arm and waist
        #world_accel = J*a + Jdot * qdot 
        diff_a = np.ones(self.ndofs)
        g_a = l2_norm(diff_a)
        dg_a = 2 * (diff_a).dot(self.oneA)
        
        #diff
        #diff_cntc
        
        #desire pose acceleration TODO : this is in pose space, not world space 
        desPoseAccel = (self.matchPose - self.curMatchPose) * self.kPose - self.tSqrtKpose * self.curMatchPoseDot
        
        funcRes =0
        
        
        
        if grad.size > 0:
            #grad calc - put grad in place here
            grad[self.accelIDXs] = dg_a
     
        
        return funcRes
        
#        self.accelIDXs = np.arange(self.ndofs)
#        self.tauIDXs = np.arange((self.ndofs+12), n)
 
    
    #current timestep physics values must be refreshed before this is called
    #equality constraint == 0
    def MAcnst(self, x, grad):   
        a = x[self.accelIDXs]
        tau = x[self.tauIDXs]
        #force first dims of tau to be 0
        tau[0:6] = 0
        
        maRes = self.M.dot(a) + self.CfG 
        tauPull = self.Tau_JtFpull - tau
        
        ttlVec = maRes + tauPull        
        
        if grad.size > 0:
            #grad calc - put grad in place - grad is n x 1 vector 
            grad[self.accelIDXs] = self.M.dot(self.oneA)
            #taudot is vector of size tau all -1
            grad[self.tauIDXs] = self.tauDot        

        return np.sum(np.absolute(ttlVec))
    
   
    
    #solving quadratic objectives : 
    # min (a, F(cntct), tau)
    
    #   w1 * | pose_fwd_accel - pose_des_accel | + 
    #   w2 * | COM_fwd_accel - COM_des_accel |
    
    #subject to 
    # : M a + (C(q,dq) + G) + Tau_JtF(cntct) + Tau_JtF(grab) = tau 
    # : ground forces being in the coulomb cone
    # : tau being within torque limits
    # : subject to no slip contacts on ground
    
    
    def calcOptTau(self):
        #set sim values used by optimization routine
        self.setSimVals()
        
        #initialize optimizer - objective function is minimizer
        self.optimizer.set_min_objective(self.objectiveFunc)
        #set constraints
        
        #TODO
        
        #run optimizer - use last result as guess
        self.nextGuess = self.opt.optimize(self.nextGuess)

        return self.nextGuess[self.tauIDXs]
    
        

    #functionality before sim step is executed on this skel
    #here is where we calculate the robot's control torques
    def preStep(self, actions, cnstLoc=np.zeros(3)):
        #actions is just placeholder, used as initial guess for optimizer maybe? 
        com = self.skel.com()
        self.posBefore = com[0]        
        self.heightBefore = com[1] 
        
        #state before sim
        self.oldState = self.state_vector()
        
        self.tau=self.setTau(a)
#        self.tau = np.zeros(self.ndofs)
        
        #tau must be set via optimization
        
        #set sim values used by optimization routine
#        self.tau = self.calcOptTau()
            
       

    #perform post-step calculations for robot - no reward for ID
    def calcRewardAndCheckDone(self,resDict):        
        #resDict holds whether the sim was broken or not - if sim breaks, we need 
        #holds these values : {'broken':False, 'frame':n_frames, 'stableStates':stblState}   
        #check first if sim is broken - illegal actions or otherwise exploding
        
        broke_sim = resDict['broken']
        com = self.skel.com()
        posAfter = com[0]        
        heightAfter = com[1]        

        #angular terms : rotating around z, y and x axis
#            ang_cos_swd = self.procOrient([0, 0, 1])
#            ang_cos_uwd = self.procOrient([0, 1, 0])
#            ang_cos_fwd = self.procOrient([1, 0, 0])
#                  
           
        # reward function calculation
        alive_bonus = 1.0
        vel = (posAfter - self.posBefore) / self.env.dt
        velScore = -abs(vel-2)+2
        raiseVel = (heightAfter - self.heightBefore) / self.env.dt
        #peak centered at velocity == 2 + self.minUpVel, peak is 2 high, so spans self.minUpVel to 4+sel.minUpVel
        raiseVelScore = -abs(raiseVel-(2+ self.minUpVel)) + 2

        height_rew = float(10 * heightAfter)

        #minimize requested torques- should this be requested or actually used? TODO
        actionPenalty = float(1e-3 * np.square(self.a).sum())

        reward = alive_bonus - actionPenalty 
        
        #dictionary holds contact information
        contactDict, COPval = self.calcFootContactRew()
        
        print ('total foot contact force : [{},{},{}]'.format(contactDict['cFrcX'],contactDict['cFrcY'],contactDict['cFrcZ']))
        
        

        done = broke_sim or not(
                (raiseVelScore >= 0) and
                #maybe not good, need to check
#                    (abs(ang_cos_uwd) < np.pi) and 
#                    (abs(ang_cos_fwd) < np.pi) and
#                    (abs(ang_cos_swd) < np.pi) and                   
                (heightAfter < 1.7) 
            )        
            
        dct = {'broke_sim': broke_sim, 'raiseVelScore': raiseVelScore, 'height_rew':height_rew,
                    'actionPenalty':actionPenalty, 'is_done': done}   
        return reward, done, dct
    

                 
                 
                 
                 
                 
                 
                 