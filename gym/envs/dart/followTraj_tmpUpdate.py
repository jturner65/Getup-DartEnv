#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 12:37:19 2018

@author: john

"""

import numpy as np

from abc import ABC, abstractmethod


#class to hold trajectories to follow
class followTraj(ABC):
    """
    args is dictionary of arguments to define trajectory
    
    trajSpeed : speed of object - how far it moves per timestep
    trackObj : reference to tracked object
    dt : timestep
    humanSkelHldr : skel holder for human
    botSkelHldr : skel holder for robot
    
    """
    
    def __init__(self, args):
        self.trajSpeed = args['trajSpeed']
        self.trackObj = args['trackObj']
        self.track_nDofs = self.trackObj.ndofs
        self.dt = args['dt']

        self.humanSkelHldr = args['humanSkelHldr']
        self.botSkelHldr = args['botSkelHldr']
        self.env = args['env']
        #whether traj should evolve dynamically (via SPD) or kinematically
        self.isDyanmic = args['isDynamic']
        #trajectory starting location
        self.stLoc = None
        #next location of trajectory
        self.nextPos = None
        #difference between current and next location of trajectory
        self.nextDispVec = None
        self.trajStep = 0
        #constant traj speed - overridden by gaussian traj
        self.trajIncrBase = self.dt * self.trajSpeed
        self.trajIncr = self.trajIncrBase
        self.debug = False
        #traj length multiplier -> used for setting location within trajectory relative to end points for linear this will be 1, for circular will be 2pi.
        self.trajLenMultiplier = 1.0
        self.buildSPDMats()

    def buildSPDMats(self):
        ndofs = self.track_nDofs
        dt = self.dt
        self.Kp_const = 100000
        self.Kd_const = dt * self.Kp_const
        self.Kp = self.Kp_const * np.identity(ndofs)
        #self.Kp[0:3,0:3] = 0
        self.Kd = self.Kd_const * np.identity(ndofs)
        #self.Kd[0:3,0:3] = 0
        self.Kd_ts = dt * self.Kd

    def setDebug(self, _debug):
        self.debug = _debug
    
    #initialize trajectory with new starting and ending location
    #initLoc : new starting postion
    #endLoc : either ending location or direction vector(?)
    def initTraj(self, initLoc, endLoc):
        self.trajStep = 0
        #set start location of trajectory
        self.stLoc = np.copy(initLoc)
        self.curLoc = np.copy(initLoc)
        self.curDisp = self.curLoc - self.stLoc
        self.oldLoc = np.copy(initLoc)
        self.nextDispVec = self.stLoc - self.oldLoc
        #put ball at this location
        self._returnToStart()
        #set up initial trajectory constants and functional components - also advances trajStep
        self.initTrajIndiv(endLoc)
        #pre-calc initial next-step trajectory - note : trajIncr must be 
        self.calcNewPosIndiv(self.trajStep)

    #return traj object to start location of trajectory if one exists
    def _returnToStart(self):
        if (self.stLoc is not None):
            self._setBallPos(self.stLoc, self.curDisp)
        else:
            print('\nfollowTraj::returnToStart : no stLoc set to return to.' )

    @abstractmethod
    def initTrajIndiv(self,dirVec):
        pass

    #return components of trajectory that would be used for an observation - next step displacement vector or next step position
    def getTrajObs(self, _typ):
        if ('disp' in _typ.lower()):
            return np.copy(self.nextDispVec)
        else :
            print("followTraj::getTrajObs : Unknown observation type query {} : returning displacement vector".format(_typ))
            return np.copy(self.nextDispVec)
    
    #save current state of trajectory
    def saveCurVals(self):
        self.savedLoc = np.copy(self.curLoc)
        self.savedCurDisp = np.copy(self.curDisp)
        self.savedNextLoc = np.copy(self.nextPos)
        self.savedNextDispLoc = np.copy(self.nextDispVec)
        self.savedOldLoc = np.copy(self.oldLoc)
        self.savedTrajStep = self.trajStep
        
    #restore a trajectory to previous saved state-restored between application of current q/qdot and next calced q/qdot
    def restoreSavedVals(self):
        self.oldLoc = self.savedOldLoc
        self.nextPos = self.savedNextLoc
        self.nextDispVec = self.savedNextDispLoc
        self._setBallPos(self.savedLoc,self.savedCurDisp)  #this will set curLoc and curDisp
        self.trajStep = self.savedTrajStep

    #call at beginning of simulation step - calculate displacement per frame.  This is called after self.nextPos is calculated
    def startSimStepFS(self, framesPerStep=1.0):
        self.curLoc = self.trackObj.com()
        self.curDisp = self.curLoc - self.oldLoc
        #divide expected per-step movement into # frames per step pieces.
        self.nextFrameDispVec = self.nextDispVec/(1.0 * framesPerStep)
        self.nextFrameIncr =  self.trajIncr /(1.0 * framesPerStep)

        if(self.debug):
            print('\n')
            if(not np.allclose(self.oldLoc, self.curLoc, 1e-10)):
                print('!!!!followTraj::advTrackedPosition : Tracked Obj moved by simulation interactions : expected old loc : {}|\t actual old loc : {}'.format(self.oldLoc,self.curLoc))
            print('followTraj::advTrackedPosition : Tracked Obj COM after Update : {}|\told loc : {}|\tnewLoc used to update per step :{} with delta = {} \n'.format(self.trackObj.com(),self.oldLoc,self.nextPos,self.trajStep))

    def startSimStep(self):
        self.curLoc = self.trackObj.com()
        self.curDisp = self.curLoc - self.oldLoc
        #print('cur loc : {}'.format(self.curLoc))
        
        self.nextFrameDispVec = self.nextDispVec
        self.nextFrameIncr =  self.trajIncr

        if(self.debug):
            print('\n')
            if(not np.allclose(self.oldLoc, self.curLoc, 1e-10)):
                print('!!!!followTraj::advTrackedPosition : Tracked Obj moved by simulation interactions : expected old loc : {}|\t actual old loc : {}'.format(self.oldLoc,self.curLoc))
            print('followTraj::advTrackedPosition : Tracked Obj COM after Update : {}|\told loc : {}|\tnewLoc used to update per step :{} with delta = {} \n'.format(self.trackObj.com(),self.oldLoc,self.nextPos,self.trajStep))

    def calcPerFrameDisp(self, fr):
        nextFramePos = self.curLoc + fr * self.nextFrameDispVec
        return nextFramePos

    #advance per frame
    #1st/only frame should have fr == 1
    def advTrackedPosition(self, fr):
        self.nextFramePos = self.calcPerFrameDisp(fr)
        #self.nextDispVec = newLoc - oldLoc#calced now in self.calcNewPosIndiv
        if (self.isDyanmic):
            self._setBallPosSPD(self.nextFramePos, self.nextFrameDispVec)
        else :
            self._setBallPos(self.nextFramePos, self.nextFrameDispVec)
    
        #advance trajStep variable, check if finished
        doneTraj = self.incrTrajStepIndiv(self.nextFrameIncr)  

        # #save current location as next step's old location, to verify that this functionality is only component responsible for ball displacement
        # self.oldLoc = np.copy(self.trackObj.com())
        # #calculate next step's location (changes nextPos)
        # self.calcNewPosIndiv(self.trajStep)
        #returns whether we are at end of trajectory or not
        return doneTraj
    
    #after all simulation frames of a single step, query traj obj location and store variables
    #kinDisp : whether or not the ball was moved kinematically or dynamically
    def endSimStep(self):
        #save current location as next step's old location, to verify that this functionality is only component responsible for ball displacement
        #location at beginning of sim step
        self.oldLoc = np.copy(self.curLoc)
        #current location
        self.curLoc = self.trackObj.com()
        #current displacement - how much the ball has moved
        self.curDisp = self.curLoc - self.oldLoc

        #calculate next step's location (changes nextPos)
        self.calcNewPosIndiv(self.trajStep)
       
        if (self.isDyanmic) :
            pass
        else :
            pass
        
    
    #move to some percentage of the full trajectory (a single circle for circular trajs)
    def setTrackedPosition(self, d, fs):
        if d < 0 : 
            d = 0
        elif d > 1 : 
            d = 1        
        self.trajStep = d * self.trajLenMultiplier
        #need to determine next position based on this specified trajStep        
        self.calcNewPosIndiv(self.trajStep)
        #need to manage per-frame skip, if used
        self.startSimStepFS(framesPerStep=fs)
        #move to new trajStep location
        return self.advTrackedPosition()
  
    #instance class-specific calculation for how to move
    #t : desired distance along path from stLoc to endLoc
    @abstractmethod
    def calcNewPosIndiv(self, t): pass

    #move object along
    @abstractmethod
    def incrTrajStepIndiv(self, perFrameIncr):pass

    #set constraint/tracked ball's position to be newPos
    def _setBallPos(self, newPos, newVel):
        q = self.trackObj.q
        dq = self.trackObj.dq

        q[3:6] = newPos
        dq[3:6] = newVel
        
        self.trackObj.set_positions(q) 
        #TODO make sure we want to move ball like this
        self.trackObj.set_velocities(dq)
        #return spherePos        

    #use SPD to determine control to derive new position and control for constraint
    def _setBallPosSPD(self, desPos, desVel):
        #frc = -kp * (q_n + dt*dq_n - desPos) - kd * (dq_n + dt*ddq_n - desVel)
        #ddq_n
        q = self.trackObj.q
        dq = self.trackObj.dq
        qBar = np.zeros(6)
        dqBar = np.zeros(6)
        qBar[-3:len(qBar)]=desPos
        dqBar[-3:len(dqBar)]=desVel
        #sim quantities
        M = self.trackObj.M
        cnstrntFrc = self.trackObj.constraint_forces()
        CorGrav = self.trackObj.coriolis_and_gravity_forces()
        kd_ts = self.Kd_ts
        mKp = self.Kp
        mKd = self.Kd
        dt = self.dt

        #spd formulation
        Mkdts = M + kd_ts
        invM = np.linalg.inv(Mkdts)
        nextPos = q + dq * dt
        p = -mKp.dot(nextPos - qBar)
        d = -mKd.dot(dq)# - dqBar)                        
        ddq = invM.dot(-CorGrav + p + d + cnstrntFrc)
        cntrl = p + d - mKd.dot(ddq * dt)
        
        # if(len(cntrl) > 3):
        #     cntrl[:-3] = 0
        print('cntrl : {}'.format(cntrl))
        print('SPD Cntrl : {} | Curr Location : {} | Desired New Location : {} | Next DispVec : {}'.format(cntrl, q[-3:len(q)], qBar[-3:len(qBar)], self.nextDispVec))
        input()

        self.trackObj.set_forces(cntrl)
        
    
    #############################################################
    #   the following static functions will build an equation that will fit a list of points, and evaulate that equation

    #will return array of deg+1 (rows) x 3 cols of coefficients of polynomial to fit points in ptsAra
    #ptsAra is list of 3d points
    @staticmethod
    def buildEqFromPts(ptsAra, deg):
        numPts = len(ptsAra)
        #take transpose of np matrix of points
        ptsTrans = np.transpose(np.array(ptsAra))
        #ptsTrans is 3 x numpoints
        t = np.linspace(0.0, 1.0, num=numPts)
        coeffsPerDim = []
        for i in range(len(ptsTrans)):
            coeffsPerDim.append(np.polyfit(t, ptsTrans[i], deg))      
        coeffs = np.transpose(np.array(coeffsPerDim))
        return coeffs

    #returns a solution to the polynomial parameterized equation at t given by coeffs
    #where coeffs is deg+1(rows) x 3 array of coefficients for deg polynomial
    @staticmethod
    def solvePolyFromCoeffs(coeffs, t):
        calcVal = np.array([0.0,0.0,0.0])
        numCoeffs = len(coeffs)
        #solve polynomial
        for i in range (numCoeffs - 1):
            calcVal += coeffs[i] * t
        #don't forget constant term       
        calcVal += coeffs[-1]
        return calcVal        

#circular trajectory   
class circleTraj(followTraj):
    def __init__(self, args):
        followTraj.__init__(self, args)
        #planar ellipse tilted ccw by tiltRad
        xRad = args['xRad']
        yRad = args['yRad']
        tiltRad = args['tiltRad']
        self.ballXradCtilt = xRad * np.cos(tiltRad)
        self.ballXradStilt = xRad * np.sin(tiltRad)
        self.ballYradCtilt = yRad * np.cos(tiltRad)
        self.ballYradStilt = yRad * np.sin(tiltRad) 
        #traj length multiplier -> used for setting location within trajectory.
        #circle traj is 2pi
        self.trajLenMultiplier = 2.0 * np.pi
        #make trajIncr negative to match formula
        self.trajIncr *= -1

    #passed endLoc is ignored
    def initTrajIndiv(self, endLoc):
        #base center on current location and vectors of x and y radius
        ctrOffset = np.array([self.ballXradCtilt, self.ballXradStilt, 0])
        self.ballTrajCtr = self.trackObj.com() + ctrOffset
        #start and end at same location
        self.endLoc = np.copy(self.stLoc)
        self.trajStep += self.trajIncr          #incr trajStep for calcNewPosIndiv call which follows in init
        
        if(self.debug):
            ballPos = self.ballTrajCtr - ctrOffset
            print('Tracking Ball location in initTracking : {} | calc pos : {}\n'.format(self.trackObj.com(), ballPos))

    def calcNewPosIndiv(self, t):
        cts = np.cos(t)
        sts = np.sin(t) 
        #circle has no end, always false for end of trajectory
        self.nextPos = self.ballTrajCtr - np.array([(self.ballXradCtilt * cts - self.ballYradStilt * sts), (self.ballYradCtilt * sts + self.ballXradStilt * cts), 0])
        self.nextDispVec = self.nextPos - self.curLoc
        
    def incrTrajStepIndiv(self, perFrameIncr):
        #evolve trajStep, never is finished
        self.trajStep += perFrameIncr
        return False    
    
#linear trajectory        
class linearTraj(followTraj):
    def __init__(self, args):
        followTraj.__init__(self, args)
        #line through origin and relPoint2 of passed length, with origin displaced to self.stLoc 
        self.length = args['length']
        #origDirVec is reasonable approximation of ending traj location w/respect to ANA's com
        #   use as traj target unless another is provided by non-zero assist force
        self.origDirVec = args['rel2ndPt'] / np.linalg.norm(args['rel2ndPt'])

    #dirVec is relative to stloc, and should be unit length
    def initTrajIndiv(self, dirVec):
        #initialize dirVec unless a legal dir vec is overriding it
        self.dirVec = np.copy(self.origDirVec)
        if (dirVec is not None):
            dirMag = np.linalg.norm(dirVec)
            if(dirMag > 0):
                self.dirVec = dirVec/dirMag
                
        #scale dirVec, which should be unit length, by desired length of trajectory
        self.moveVec = self.dirVec * self.length
        self.endLoc = self.stLoc + self.moveVec    
        self.trajStep += self.trajIncr          #incr trajStep for calcNewPosIndiv call which follows in init

    #calculate new position based on distance from start along moveVec
    def calcNewPosIndiv(self, t):
        self.nextPos = self.stLoc + (t * self.moveVec)
        self.nextDispVec = self.nextPos - self.curLoc


    def incrTrajStepIndiv(self, perFrameIncr): 
        self.trajStep += perFrameIncr
        #line is done when we've moved length along dirvec
        return (self.trajStep >= 1.0)

#trajectory that follows a gaussian distribution toward a set end point
#directed wandering with mean direction being from current location to end point
class gaussTraj(followTraj):
    def __init__(self, args):
        followTraj.__init__(self, args)
        #ref to random generator for new trajectory location
        self.np_random = self.env.np_random
        #wander is used to scale stds for sampling
        self.wander = args['wander']
        #traj speed is random every time traj is calculated - make random based on specified trajIncr

    #init end location for trajectory and build appropriate vectors and values used later
    def initTrajIndiv(self, endLoc):
        #random increment for each trajectory
        self.trajIncr = self.np_random.uniform(low = .5*self.trajIncrBase, high = 1.5*self.trajIncrBase)#  .normal(loc=self.trajIncr, scale=1.0)
        self.endLoc = endLoc  
        self.relEndLoc = endLoc - self.stLoc
        length = np.linalg.norm(self.relEndLoc)
        #direction from starting point to ending point
        self.dirStToEnd = self.relEndLoc/length
        #find random value to displace trajectory for first step
        self.trajStep += self.trajIncr      #incr trajStep for calcNewPosIndiv call which follows in init

    #calculate new position and displacement vector based on desired location along vector between start and end loc give by t
    def calcNewPosIndiv(self, t):
        stdScale = self.wander
        #use current location to find current projection on line from stloc to endloc
        curT = self._locAlongLine(self.curLoc)
        #t holds where we want to be on line 
        distToMove = t - curT
        
        #build dir vec based on this difference - if non-negative, want to move toward end, if negative want to move toward beginning
        if (distToMove < 0) :               #move toward beginning if moving backward
            moveToPt = self.stLoc
            d = -distToMove                     #make positive
        else:                               #move toward end target if move is positive
            moveToPt = self.endLoc
            #scale = np.abs(1.0 - curT)*stdScale       #scale std of gaussian by how far away we are - tighter dist closer to dest point
            d = distToMove
        #direction to move
        mvVec = moveToPt-self.curLoc
        distToTarg = np.linalg.norm(mvVec)
        #unit movement vector
        mvDir = mvVec/distToTarg
        #scale is based on distance to target - more std the further we are from target
        scale = distToTarg*stdScale
        #find direction and length of displacement from current location
        sclAra = np.ones(mvDir.shape)*scale*2.0
        smplMvDirRaw = self.np_random.normal(loc=mvDir, scale=sclAra)
        smplMvDist = np.abs(self.np_random.normal(loc=d, scale=scale*.05))
        smplMvDirLen = np.linalg.norm(smplMvDirRaw)
        smplMvDirVec = smplMvDirRaw * smplMvDist/smplMvDirLen        

        self.nextPos = self.curLoc + smplMvDirVec
        self.nextDispVec = smplMvDirVec
        nextT = self._locAlongLine(self.nextPos)
        self.trajIncr = nextT - curT
        # print('next T : {} | cur T : {} | distToMove : {}'.format(t, curT, distToMove))
        # print('scale : {} | stPos : {} | endPos : {} | curPos : {} | nextPos : {} | nextDispVec : {} | nextTrajIncr : {}'.format(scale,self.stLoc,self.endLoc,self.curLoc, self.nextPos, self.nextDispVec, self.trajIncr) )

    #find how far loc is from traj line through stLoc, endLoc
    def _distFromLine(self, loc): 
        t = self. _locAlongLine(loc)
        #point on line
        P = (self.stLoc + t * self.dirStToEnd)
        d = np.linalg.norm(P-loc)
        return d

    #find how far along line loc is - projection to line between stLoc and endLoc as t = [0,1] 
    def _locAlongLine(self,loc):
        relStLoc = loc - self.stLoc
        t = relStLoc.dot(self.dirStToEnd)
        return t
    
    #new tracked position is dependent on ratio of current distance between start and end points current location is projected to        
    def incrTrajStepIndiv(self,perFrameIncr): 
        #increment trajStep, determine if at end of trajectory
        self.trajStep += perFrameIncr
        return (self.trajStep >= 1.0)
    
#parabolic trajectory        
class parabolicTraj(followTraj):
    def __init__(self, args):
        followTraj.__init__(self, args)
    #endloc is ignored
    def initTrajIndiv(self,endLoc):
        self.trajStep += self.trajIncr
        print('parabolicTraj.initTrajPriv : Not Implemented')

    def calcNewPosIndiv(self, t): 
        print('parabolicTraj.calcNewPosIndiv : Not Implemented')
        self.nextDispVec = self.nextPos - self.curLoc

    def incrTrajStepIndiv(self,perFrameIncr):
        print('parabolicTraj.incrTrajStepIndiv : Not Implemented')      
        #evolve trajstep
        self.trajStep += self.trajIncr
       
        return False
    
##trajectory specified by equation
#class eqTraj(followTraj):
#    def _init_ (self, args):
#        followTraj.__init__(self, args)
#        pts = args['trajPoints']
#        numPts = len(pts)
#        t=np.linspace(0,1.0,num=numPts, endpoint=True)
#        x=np.array([p[0] for p in pts ])
#        y=np.array([p[1] for p in pts ])
#        z=np.array([p[2] for p in pts ])
#        xeq = 

        