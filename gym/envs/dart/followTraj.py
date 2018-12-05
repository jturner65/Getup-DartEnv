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
        self.trackBodyNode = self.trackObj.body(0)
        self.trackBodyAntiGrav = np.array([0,9.8,0]) * self.trackObj.mass()
        self.track_nDofs = self.trackObj.ndofs
        self.humanSkelHldr = args['humanSkelHldr']
        self.botSkelHldr = args['botSkelHldr']
        self.env = args['env']
        self.frame_skip = self.env.frame_skip
        #dt should be the time step between updates for the follow traj object
        self.dt = args['dt']
        #type of trajectory
        self.trajType = args['trajType']

        #trajectory creation messages, for debugging
        self.dbgBuildMsg = []
        #whether traj is passive or not
        isPassive = args['isPassive']
        if isPassive :
            self.dbgBuildMsg.append("Traj is passive and does not evolve on its own")
            funcToUse = "setBallNone"
        else :
            #whether traj should evolve dynamically (via setting(via SPD) a control) or kinematically
            isDyanmic = args['isDynamic']
            if isDyanmic :
                self.dbgBuildMsg.append("Traj Is Dynamically evolved")
                funcToUse = args['dynamicFunc']
            else :
                self.dbgBuildMsg.append("Traj Is Kinematically evolved")
                funcToUse = args['kinematicFunc']
        self.advFuncToUse = self.getFuncFromNameStr(funcToUse)
        
        
        if (self.frame_skip == 1):
            self.coeffsFuncToUse = self.getFuncFromNameStr("setCoeffsSmoothAccel")
        else : 
            #coefficients to use to calculate spline trajectory for frameskips
            self.coeffsFuncToUse = self.getFuncFromNameStr(args['trajTargetFunc'])
            #method to derive v1 target velocity
        self.v1FuncToUse = self.getFuncFromNameStr(args['accelManageFunc'])
        self.dbgBuildMsg.append("Set initVel : {} ".format(args['accelManageFunc']))
        self.dbgBuildMsg.append("Set Traj Coeffs : {} ".format(args['trajTargetFunc']))
        self.dbgBuildMsg.append("Advance per step  : {} ".format(funcToUse))
        self.dbgDispDbgMessage()
        #trajectory starting location
        self.stLoc = None
        #traj end location
        self.endLoc = None
        #next location of trajectory
        self.nextLoc = None
        #difference between current and next location of trajectory
        self.nextDispVec = None
        self.trajStep = 0
        #constant traj speed - overridden by gaussian traj
        self.trajIncrBase = self.dt * self.trajSpeed
        #account for smaller timestep used with more frameskips
        self.trajIncrBase *= self.frame_skip
        self.trajIncr = self.trajIncrBase
        self.debug = False
        #traj length multiplier -> used for setting location within trajectory relative to end points for linear this will be 1, for circular will be 2pi.
        self.trajLenMultiplier = 1.0
        self.buildSPDMats()
        #temp debugging variable
        self.tmpVelSum = np.zeros(3)
        #debug message list, value is display string
        self.debugMessageList = []

    def dbgDispDbgMessage(self):
        print("followTraj::__init__ : Functions used : {} | {} | {}".format(self.dbgBuildMsg[0],self.dbgBuildMsg[1],self.dbgBuildMsg[2]) )


    #determine function pointers for functions to be used 
    def getFuncFromNameStr(self, name): 
        #displacement functions
        if "setBallVel" in name : 
            return self._setBallPosVel
        elif "setBallPosKin" in name : 
            return self._setBallPos
        elif "setBallPosSPD" in name : 
            return self._setBallPosSPD
        elif "setBallNone" in name :   #treat as passive construct
            return self._setBallNone

        #trajectory functions
        elif "setCoeffsCubic" in name :
            return self._setCoeffsCubic
        elif "setCoeffsQuartic" in name :
            return self._setCoeffsQuartic
        elif "setCoeffsSmoothAccel" in name:
            return self._setCoeffsSmoothAccel

        #final step velocity determination functions - determines how acceleration is handled
        elif "setV1_Mean" in name :
            return self._setV1_Mean
        elif "setV1_Zero" in name : 
            return self._setV1_Zero
        elif "setV1_ToGoal" in name :
            return self._setV1_ToGoal

        else : 
            print("followTraj::getFuncFromNameStr : unknown advance function name : {}".format(name))
            return None

    #retrieve debug messages set by this trajectory and clear list
    def getCurTrajDbgMsgs(self):
        res = []
        for msg in self.dbgBuildMsg:
            res.append(msg)
        res.append("")
        res.append("---Trajectory debug messages : ")
        for line in self.debugMessageList :
            res.append(line)
        self.debugMessageList=[]
        return res

    #enable traj to print and retain debug messages         
    def trajPrint(self, srcMthd, keys, dbgStrs, prEnd = '\n', printToConsole=False):
        numKeys = len(keys)
        dispStr = "{} : ".format(srcMthd)
        for i  in range(numKeys) : 
            dbgStr =  "{} : {}".format(keys[i],dbgStrs[i])
            self.debugMessageList.append(dbgStr)
            dispStr += dbgStr + " | "
        if (printToConsole):
            print(dispStr, end=prEnd)

    def buildSPDMats(self):
        ndofs = self.track_nDofs
        dt = self.dt
        Kp_const = 1000000
        Kd_const = dt * Kp_const
        self.Kp = Kp_const * np.identity(ndofs)
        #self.Kp[0:3,0:3] = 0
        self.Kd = Kd_const * np.identity(ndofs)
        #self.Kd[0:3,0:3] = 0
        self.Kd_ts = dt * self.Kd

    def setDebug(self, _debug):
        self.debug = _debug
    
    #initialize trajectory with new starting and ending location at beginning of rollout
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
        #set up initial trajectory constants and functional components - traj Incr set here
        self.initTrajIndiv(endLoc)
        #debug message dictionary - keyed by what is displayed, value is display string
        self.debugMessageList = []

    #return traj object to start location of trajectory if one exists
    def _returnToStart(self):
        if (self.stLoc is not None):
            self._setBallPosRaw(self.stLoc, self.curDisp)
        else:
            print('\nfollowTraj::returnToStart : no stLoc set to return to.' )

    @abstractmethod
    def initTrajIndiv(self,dirVec):
        pass

    #return the most recent external(constraint) force responsible for modifying the ball's motion independent of ANA.  This force is 
    #based on the acceleration and the ball's mass
    def getPerStepMoveExtForce(self):
        return self.trackObj.mass()  * self.perStepAccel
            
    #return components of trajectory that would be used for an observation - next step displacement vector or next step position
    def getTrajObs(self, _typ):
        #coeffs[-1] is p0
        disp = np.copy(self.nextDispVec) #self.curLoc - self.coeffs[-1] #can't do this, getting observation before movement has occurred.  
        if ('disp' in _typ.lower()):
            return disp
        else :
            print("followTraj::getTrajObs : Unknown observation type query {} : returning displacement vector".format(_typ))
            return disp
        
    def setVFTrajObs(self, _typ, _val):
        if ('disp' in _typ.lower()):
            self.setNewLocVals(np.copy(_val))
        else :
            print("followTraj::setVFTrajObs : Unknown observation type query {} : setting displacement vector from _val".format(_typ))
            self.setNewLocVals(np.copy(_val))       
    
    #save current state of trajectory
    def saveCurVals(self):
        self.savedLoc = np.copy(self.curLoc)
        self.savedCurDisp = np.copy(self.curDisp)
        self.savedNextLoc = np.copy(self.nextLoc)
        self.savedNextDispLoc = np.copy(self.nextDispVec)
        self.savedOldLoc = np.copy(self.oldLoc)
        self.savedCoeffs = []
        for coeff in self.splineCoeffs:
            self.savedCoeffs.append(np.copy(coeff))
        self.savedTrajStep = self.trajStep
        
    #restore a trajectory to previous saved state-restored between application of current q/qdot and next calced q/qdot
    def restoreSavedVals(self):
        self.oldLoc = self.savedOldLoc
        self.nextLoc = self.savedNextLoc
        self.nextDispVec = self.savedNextDispLoc
        self.curLoc = self.savedLoc
        self.splineCoeffs = []
        for coeff in self.savedCoeffs:
            self.splineCoeffs.append(np.copy(coeff))
        self._setBallPosRaw(self.savedLoc,self.savedCurDisp)  #this will set curLoc and curDisp
        self.trajStep = self.savedTrajStep

    ########################################################################################
    # these are called at the beginning of every timestep.

    #set current location and compare to last curLoc.  should only happen 1 time per ana control step
    def _setStartCurLoc(self):
        #self.endLastStep = np.copy(self.curLoc)
        self.curLoc = self.trackObj.com()
        self.curDisp = self.curLoc - self.oldLoc
        # if (not np.allclose(self.endLastStep,self.curLoc, 1e-10)):
        #     print('followTraj::startSimStepFS : End Last Step  : {}  != current location :  {}'.format(self.endLastStep,self.curLoc))
        # else :
        #     print('followTraj::startSimStepFS : End Last Step  : {}  == current location :  {}'.format(self.endLastStep,self.curLoc))

    #set 1 time per ANA control step if evolving ANA's position naturally
    def setNewPosition(self):
        if(self.debug):
            print("followTraj::setNewPosition : trajStep : {} | avg vel seen : {}".format(self.trajStep,self.tmpVelSum/(1.0+ 1.0*self.frame_skip)))
        #print("Traj Incr : {}".format(self.trajIncr))
        self._setStartCurLoc()
        #calculate next step's location (changes nextPos)
        self._calcNewPos(self.trajStep)
    
    #move to some percentage of the full trajectory
    def setTrackedPosition(self, d):
        self._setStartCurLoc()
        if d < 0 : 
            d = 0
        elif d > 1 : 
            d = 1        
        self.trajStep = d * self.trajLenMultiplier
        #need to determine next position based on this specified trajStep        
        self._calcNewPos(self.trajStep)

    #calculate new position - call individual calculation and then scale displacements by frameskip if necessary
    #this is called at the start of every control application step
    def _calcNewPos(self, t):
        self.calcNewPosIndiv(t)
        #divide expected per-step movement into # frames per step pieces.
        div = (1.0 * self.frame_skip)

        #amount of trajectory increment that should be applied per frame skip
        self.nextFrameIncr =  self.trajIncr/div 
 
        T = (div*self.dt) #total time to go from init to final

        #determine coefficients of spline based on p0, p1, v0, v1, such that acceleration is minimized (not const)
        #use these so that position/velocity obey initial and final constraints
        #init position, final position, init vel, final vel
        p0 = np.copy(self.curLoc)
        p1 = np.copy(self.nextLoc)
        v0 = np.copy(self.trackObj.dq[3:6])
        self.tmpVelSum = np.copy(v0)
        #deterine next velocity model to use
        v1 = self.v1FuncToUse(T, v0)
        # if (self.frame_skip == 1):
        #     #if there is only 1 individual simulation step per control/render frame, use the following eq always for coefficients
        #     self.coeffs = self._setCoeffsSmoothAccel(T, p0, p1, v0, v1)
        # else :   
        self.coeffs = self.coeffsFuncToUse(T, p0, p1, v0, v1)
        # T = (div*self.dt) #total time to go from init to final
        # pos1, vel1 = self.calcSplineValsAtTime(T)
        # print("\tTest @ T={} : pos = {} | vel = {}".format(T, pos1, vel1))

        # #displacement and trajIncr per sim step
        # self.nextSimStepDispVec = self.nextDispVec/div
        # #find acceleration that will smoothly interpolate between current velocity and desired end velocity
        # curVel = self.trackObj.dq[3:6]
        # #velocity over entire frame
        # nextVel = self.nextDispVec/(div*self.dt)
        # #del V over entire frame / per frame
        # self.nextSimStepAccel = (nextVel - curVel)/(div*self.dt)

    #set v1 by using the nextDispVec as the average velocity over the timestep
    def _setV1_Mean(self, T, v0):
        return 2.0*(self.nextDispVec/T) - v0

    #set target end velocity to be 0
    def _setV1_Zero(self, T, v0):
        return np.zeros(3)
    
    #set target end velocity to be magnitude of v1 that would cause self.nextDispVec to be mean displacement, but pointed toward goal
    def _setV1_ToGoal(self, T, v0):
        tmpV1 = self._setV1_Mean(T, v0)
        tmpV1Mag = np.linalg.norm(tmpV1)
        dirVec = self.endLoc - self.curLoc
        dirVecMag = np.linalg.norm(dirVec)
        res = dirVec * (tmpV1Mag/dirVecMag)
        return res

    #this will take the initial velocity and the final velocity and smoothly interpolate between them to determine appropriate accelerations
    def _setCoeffsSmoothAccel(self, T, p0, p1, v0, v1):
        #change in velocity over entire timestep - from equation for displacement for v1, minus v0
        #delVel = 2.0*(self.nextDispVec/T - v0)
        #set acceleration 
        accel = .5*(self.nextDispVec/T - v0)/T
        #use p(x) = at^2 + v0t + p0 
        # v(x) = 2at + v0
        coeffs = []
        coeffs.append(accel)
        coeffs.append(v0)
        coeffs.append(p0)
        if(self.debug):
            #print("followTraj::_setCoeffsSmoothAccel", end=' ')
            keyList = ["Coeffs Type : ","p0", "p1","v0","v1", "Coeffs"]
            msgList = ["SmoothAccel", p0,p1,v0,v1,coeffs]
            self.trajPrint("followTraj::_setCoeffsSmoothAccel",keyList, msgList)
            #print("followTraj::_setCoeffsSmoothAccel :  p0 : {} p1 : {} v0 : {} v1 : {} T : {} coeffs : {}".format(p0,p1,v0,v1,T, coeffs))
        return coeffs

    #build interpolation minimizing acceleration (i.e. minimizing constraint force)
    def _setCoeffsCubic(self, T, p0, p1, v0, v1):
        T2 = T*T
        T3 = T*T2
        #Cubic poly for position : x(t) = at^3 + bt^2 +  v0t + p0 ; t is relative to beginning of traj
        #velocity : 3at^2 + 2bt +  v0
        #accel : 6at  + 2b 
        #jerk : 6a              : to minimize accel, Jounce ought to be > 0 (2nd deriv test); to minimize roc of acceleration, jounce == 0
        #find coefficients to match p1;v1 : use equation for x(t) @ T and for v(t) (dx(t)) @ T
        #augment amd solve, holding c as free (to determine acceleration behavior)
        #|  T3  T2  | p1 - p0 - v0T |  --> | 1   0    Q | :=> 
        #| 3T2 2T   |       v1 - v0 |  --> | 0   1    R | :=> 
        #where Q and R are as below
        p0mp1 = (p0-p1)
        a = ((2.0*p0mp1) + T * (v0 + v1))/T3
        b = ((-3.0*p0mp1) - T *((2*v0) + v1))/T2
        splineCoeffs = []
        splineCoeffs.append(a)  
        splineCoeffs.append(b)  
        splineCoeffs.append(v0)  
        splineCoeffs.append(p0)  
        if(self.debug):
            #print("followTraj::_setCoeffsCubic : ", end=' ')
            keyList = ["Coeffs Type : ","p0", "p1","v0","v1", "Coeffs"]
            msgList = ["Cubic", p0,p1,v0,v1,splineCoeffs]
            self.trajPrint("followTraj::_setCoeffsCubic", keyList, msgList)
            #print("followTraj::_setCoeffsCubic :  p0 : {} p1 : {} v0 : {} v1 : {} T : {} coeffs : {}".format(p0,p1,v0,v1,T, splineCoeffs))
        return splineCoeffs
   
    #build interpolation minimizing acceleration (i.e. minimizing constraint force)
    def _setCoeffsQuartic(self, T, p0, p1, v0, v1):
        T2 = T*T
        T3 = T*T2
        T4 = T2*T2
        #quartic poly for position : x(t) = at^4 + bt^3 + ct^2 + v0t + p0 ; t is relative to beginning of traj
        #velocity : 4at^3 + 3bt^2 + 2ct + v0
        #accel : 12at^2 + 6bt + 2c
        #jerk : 24at + 6b   : 
        #jounce : 24a       : to minimize accel, Jounce ought to be > 0 (2nd deriv test); to minimize roc of acceleration, jounce == 0
        #find coefficients to match p1;v1 : use equation for x(t) @ T and for v(t) (dx(t)) @ T
        #augment amd solve, holding c as free (to determine acceleration behavior)
        #|  T4  T3 T2  | p1 - p0 - v0T |  --> | 1   0  -1/T2  Q | :=> a = Q + c/T2
        #| 4T3 3T2 2T  |       v1 - v0 |  --> | 0   1   2/T   R | :=> b = R - 2c/T
        #|12T2 6T  2   |        accel  |  where accel can be constant or some minimum
        #where Q and R are as below
        p0mp1 = (p0-p1)
        Q = ((3.0*p0mp1) + T * ((2*v0) + v1))/T4
        R = ((-4.0*p0mp1) - T *((3*v0) + v1))/T3
        splineCoeffs = []
        #a = Q + c/T2; we want to minimize accel (cnstrntFrc) 
        # so c = -QT2  (and therefore a == 0)
        #b is dependent on c (acceleration term); R - 2c/T
        #minimize jerk : c = -Q*T2 (so that Jounce == 0)
        #minimize acceleration :    (so that when jerk == 0, a must be > 0)
        c = -Q * T2
        b = R - ((2.0*c)/T) 
        a = Q + (c/T2)
        splineCoeffs.append(a)  
        splineCoeffs.append(b)  
        splineCoeffs.append(c)  
        splineCoeffs.append(v0)  
        splineCoeffs.append(p0)  
        if(self.debug):
            #print("followTraj::_setCoeffsQuartic : ", end=' ')
            keyList = ["Coeffs Type : ","p0", "p1","v0","v1", "Coeffs"]
            msgList = ["Quartic", p0,p1,v0,v1,splineCoeffs]
            self.trajPrint("followTraj::_setCoeffsQuartic", keyList, msgList)
            #print("followTraj::_setCoeffsQuartic :  p0 : {} p1 : {} v0 : {} v1 : {} T : {} coeffs : {}".format(p0,p1,v0,v1,T, splineCoeffs))
        return splineCoeffs

    #calc position and velocity at specific point in time
    def calcSplineValsAtTime(self, t):
        coeffs = self.coeffs
        pt = np.zeros(3)
        vt = np.zeros(3)
        numCs = len(coeffs)
        #((((0 + a)*t + b)*t + c) * t + v0) * t + p0
        #((((0 + 4*a)*t + 3*b)*t + 2*c)*t + v0 
        for i in range(numCs - 2):
            pt = (pt + coeffs[i]) * t
            vt = (vt + ((numCs-1-i)*coeffs[i])) * t
        pt = (pt + coeffs[-2])* t + coeffs[-1]
        vt += coeffs[-2]
        return pt, vt

    #advance per sim step
    #1st/only frame should have fr == 1, subsequent frames (if any) should increment from there
    def advTrackedPosition(self, fr):
        frdt = (self.dt * fr)
        #print("advTrackedPosition : fr == {} dt == {} fr*dt == {}".format(fr, self.dt, frdt))
        #subdivide target location by # of frames to move toward target incrementally
        #nextLocPerFrame = self.curLoc + self.nextSimStepDispVec
        nextLocPerFrame, nextVelPerFrame = self.calcSplineValsAtTime(frdt) 
        #debug with constant values
        #nextVelPerFrame = np.array([0,1.0,0])
        #nextLocPerFrame = self.curLoc + nextVelPerFrame * self.dt
        if (self.debug):
            self.tmpVelSum += nextVelPerFrame
        q = self.trackObj.q
        dq = self.trackObj.dq
        
        nextDisp = nextLocPerFrame - self.curLoc
        perStepAccelFromPos = (nextDisp - self.curDisp )/(self.dt * self.dt)
        self.perStepAccel = (nextVelPerFrame - dq[3:6])/self.dt
        #set in constructor based on if dynamic of kinematic
        
        self.advFuncToUse(q=q, dq=dq, desPos=nextLocPerFrame, desVel=nextVelPerFrame)
        #this is the constraint force resulting in motion - subtract this from penalty term for each step
        if(self.debug):
            #print("followTraj::advTrackedPosition : ", end=" ")
            keyList = ['actual accel seen per ts','derived from postion','next Desired Loc']
            msgList = [self.perStepAccel, perStepAccelFromPos, nextLocPerFrame]
            self.trajPrint("followTraj::advTrackedPosition",keyList, msgList)
            #print("advTrackedPosition : actual accel seen per ts : {} derived from postion : {} next Desired Loc : {} ".format(self.perStepAccel, perStepAccelFromPos,nextLocPerFrame) )    
        #advance trajStep variable, check if finished
        doneTraj = self.incrTrajStepIndiv(self.nextFrameIncr)  
        #returns whether we are at end of trajectory or not
        return doneTraj
    
    #after one simulation step
    #kinDisp : whether or not the ball was moved kinematically or dynamically
    def endSimStep(self):
        #save current location as next step's old location, to verify that this functionality is only component responsible for ball displacement
        #location at beginning of sim step
        self.oldLoc = np.copy(self.curLoc)
        #current location after sim step
        self.curLoc = self.trackObj.com()
        #current displacement - how much the ball has moved
        self.curDisp = self.curLoc - self.oldLoc
        #print('followTraj::endSimStep : current location after sim frame :  {}'.format(self.curLoc))

    #instance class-specific calculation for how to move
    #t : desired distance along path from stLoc to endLoc
    @abstractmethod
    def calcNewPosIndiv(self, t): pass

    #move object along
    @abstractmethod
    def incrTrajStepIndiv(self, perFrameIncr): pass

    #return min and max bounds of displacement for a single iteration
    @abstractmethod
    def getMinMaxBnds(self): pass
    #return a random displacement in line with traj obj params
    @abstractmethod
    def getRandDispVal(self):pass

    #does not evolve constraint object, treats as passive
    def _setBallNone(self, desPos, desVel): pass

    #called as part of the setup of the trajectory
    def _setBallPosRaw(self, desPos, desVel):
        q = self.trackObj.q
        dq = self.trackObj.dq        
        q[3:6] = desPos
        dq[3:6] = desVel
        self.trackObj.set_positions(q) 
        #TODO make sure we want to move ball like this
        self.trackObj.set_velocities(dq)
        
    
    #set constraint/tracked ball's position to be newPos
    def _setBallPos(self, q, dq, desPos, desVel):        
        if(self.debug):
            #print("------------ Set Ball Pos : ", end=" ")
            keyList = ['Evolve Type','cur','cur vel','tar','tar vel']
            msgList = ['Set Pos', q[3:6],dq[3:6],desPos,desVel]
            self.trajPrint("followTraj::_setBallPos", keyList, msgList, prEnd=" | ")
            #print("------------ Set Ball Pos : cur : {} | cur vel : {} | tar : {} | tar vel : {}".format(q[3:6],dq[3:6],desPos,desVel),end=" | ")
        q[3:6] = desPos
        dq[3:6] = desVel
        self.trackObj.set_positions(q) 
        #TODO make sure we want to move ball like this
        self.trackObj.set_velocities(dq)
        #return spherePos        

    #set desired control as based on velocity
    def _setBallPosVel(self, q, dq, desPos, desVel):
        #control is desired velocity for this (linked to world via servo joint) target
        if(self.debug):
            #print("------------ Set Ball Vel cntrol : ", end=" ")
            keyList = ['Evolve Type', 'cur pos','cur vel','des vel','accel coeff']
            msgList = ['Set Vel', self.curLoc,dq[3:6],desVel, self.coeffs[-3]]
            self.trajPrint("followTraj::_setBallPosVel", keyList, msgList, prEnd=" | ")
            #print("------------ Set Ball Vel cntrol : cur pos : {} | vel: {} | des vel : {} | accel coeff : {} ".format(self.curLoc,dq[3:6],desVel, self.coeffs[-3]),end=" | ")
        dq[3:6] = desVel
        self.trackObj.set_commands(dq) 

    #use SPD to determine control to derive new position and control for constraint - used to derive frc/trque from desired displacement
    #this will not work because constraint forces are for "last time" and constraint forces do not evolve smoothly over time (they are like
    # contacts and will exhibit discontinuities).  These constraint forces amount to ANA pulling on ball, and they are impossible to predict
    def _setBallPosSPD(self, q, dq, desPos, desVel):
        #compensate for gravity
        #self.trackBodyNode.add_ext_force(self.trackBodyAntiGrav)
        #frc = -kp * (q_n + dt*dq_n - desPos) - kd * (dq_n + dt*ddq_n - desVel)
        #ddq_n
        qBar = np.copy(q)
        qBar[-3:len(qBar)]=desPos
        dqBar = np.copy(dq)
        #dqBar = np.zeros(6)
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
        nextPos = q + (dq * dt)
        p = -mKp.dot(nextPos - qBar)
        d = -mKd.dot(dq - dqBar)                        
        ddq = invM.dot(-CorGrav + p + d + cnstrntFrc)
        #cntrl = p + d - mKd.dot(ddq * dt)
        cntrl = p + d - kd_ts.dot(ddq)
        cntrl[0:3]=np.zeros(3)
        
        # if(len(cntrl) > 3):
        #     cntrl[:-3] = 0
        if(self.debug):
            #print("------------ SPD Cntrl : ", end=" ")
            keyList = ['Evolve Type', 'Cntrl', 'Curr Location','Desired New Location','Desired vel']
            msgList = ['SPD Cntrl', cntrl, q[-3:len(q)], qBar[-3:len(qBar)], desVel]
            self.trajPrint("followTraj::_setBallPosSPD", keyList, msgList, prEnd=" | ")
            #print("------------SPD Cntrl : {} | Curr Location : {} | Desired New Location : {} | Next DispVec : {}".format(cntrl, q[-3:len(q)], qBar[-3:len(qBar)], desVel),end=" | ")
        #input()
        #add control force
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

    #return min and max bounds of displacement for a single iteration
    def getMinMaxBnds(self): 
        res = [np.array([0,0,0]),np.array([self.trajIncr,self.trajIncr,self.trajIncr])]
        return res

    #return a random displacement in line with traj obj params
    def getRandDispVal(self):
        cts = np.cos(self.trajIncr)
        sts = np.sin(self.trajIncr) 
        return self.ballTrajCtr - np.array([(self.ballXradCtilt * cts - self.ballYradStilt * sts), (self.ballYradCtilt * sts + self.ballXradStilt * cts), 0])


    #passed endLoc is ignored
    def initTrajIndiv(self, endLoc):        
        #base center on current location and vectors of x and y radius
        ctrOffset = np.array([self.ballXradCtilt, self.ballXradStilt, 0])
        self.ballTrajCtr = self.trackObj.com() + ctrOffset
        #start and end at same location
        self.endLoc = np.copy(self.stLoc)
        
        if(self.debug):
            ballPos = self.ballTrajCtr - ctrOffset
            print('Tracking Ball location in initTracking : {} | calc pos : {}\n'.format(self.trackObj.com(), ballPos))

    def calcNewPosIndiv(self, t):
        cts = np.cos(t)
        sts = np.sin(t) 
        #circle has no end, always false for end of trajectory
        self.nextLoc = self.ballTrajCtr - np.array([(self.ballXradCtilt * cts - self.ballYradStilt * sts), (self.ballYradCtilt * sts + self.ballXradStilt * cts), 0])
        self.nextDispVec = self.nextLoc - self.curLoc
        
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

    #return min and max bounds of displacement for a single iteration
    def getMinMaxBnds(self): 
        mvVal = self.nextFrameIncr * self.moveVec
        res = [np.array([0,0,0]),mvVal]
        return res

    #return a random displacement in line with traj obj params
    def getRandDispVal(self):
        return self.nextFrameIncr * self.moveVec

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

    #calculate new position based on distance from start along moveVec
    def calcNewPosIndiv(self, t):
        self.nextLoc = self.stLoc + (t * self.moveVec)
        self.nextDispVec = self.nextLoc - self.curLoc


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

    #return min and max bounds of displacement for a single iteration
    def getMinMaxBnds(self): 
        import math
        #this is maximum trajectory incrBase
        maxDist = 1.5*self.trajIncrBase
        #arbitrary value determined by observation
        #maxD = 3*math.sqrt(3.0 * (maxDist * maxDist))
        maxD = 15*math.sqrt(3.0 * (maxDist * maxDist))

        minVal = np.array([-maxD,-maxD,-maxD])
        maxVal = -1 * minVal
        res = [minVal,maxVal]
        return res

    #return a random displacement in line with traj obj params
    def getRandDispVal(self):
        res, _, _, _ = self._calcNewPosGaussVals(self.trajIncr)

        return res

    #init end location for trajectory and build appropriate vectors and values used later
    def initTrajIndiv(self, endLoc):
        #random increment for each trajectory - need to scale by per frame increment if present
        self.trajIncr = self.np_random.uniform(low = .5*self.trajIncrBase, high = 1.5*self.trajIncrBase)#  .normal(loc=self.trajIncr, scale=1.0)

        self.endLoc = endLoc  
        self.relEndLoc = endLoc - self.stLoc
        length = np.linalg.norm(self.relEndLoc)
        #direction from starting point to ending point
        self.dirStToEnd = self.relEndLoc/length


    def _calcNewPosGaussVals(self, t):
        stdScale = self.wander
        #use current location to find current projection on line from stloc to endloc
        curT = self._locAlongLine(self.curLoc)
        #t holds where we want to be on line between beginning and ending position
        distToMove = t - curT        
        # #build dir vec based on this difference - if non-negative, want to move toward end, if negative want to move toward beginning
        # if (distToMove < 0) :               #move toward beginning if moving backward
        #     moveToPt = self.stLoc
        #     d = -distToMove                     #make positive
        # else:   
                                        #move toward end target if move is positive
        moveToPt = self.endLoc
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
        return smplMvDirVec, curT, distToMove, scale


    #calculate new position and displacement vector based on desired location along vector between start and end loc given by t
    def calcNewPosIndiv(self, t):
        smplMvDirVec, curT, distToMove, scale  = self._calcNewPosGaussVals(t)
        self.nextLoc = self.curLoc + smplMvDirVec
        self.nextDispVec = smplMvDirVec
        nextT = self._locAlongLine(self.nextLoc)
        self.trajIncr = nextT - curT
        if(self.debug):
            #print("------------ gaussTraj::calcNewPosIndiv : ", end=" ")
            keyList = ['Traj Type', 'stPos','endPos','next T','cur T','distToMove','scale','curPos','nextPos','nextDispVec','nextTrajIncr'  ]
            msgList = [self.trajType, self.stLoc,self.endLoc, '{:.5f}'.format(t), '{:.5f}'.format(curT), '{:.5f}'.format(distToMove), '{:.5f}'.format(scale),self.curLoc, self.nextLoc, self.nextDispVec, '{:.5f}'.format(self.trajIncr)]
            self.trajPrint("gaussTraj::calcNewPosIndiv", keyList, msgList, prEnd=" | ")


    #sets new location values based on current location and passed movement vector
    def setNewLocVals(self, smplMvDirVec):
        curT = self._locAlongLine(self.curLoc)
        self.nextLoc = self.curLoc + smplMvDirVec
        self.nextDispVec = smplMvDirVec
        nextT = self._locAlongLine(self.nextLoc)
        self.trajIncr = nextT - curT

    #find how far loc is from traj line through stLoc, endLoc
    def _distFromLine(self, loc): 
        t = self._locAlongLine(loc)
        #point on line
        P = (self.stLoc + t * self.dirStToEnd)
        d = np.linalg.norm(P-loc)
        return d

    #find point on line from st to finish at t (0->1) and return point and point doted with norm of plane (== self.dirStToEnd)
    #this treats line as normal of plane - dot dirStToEnd with point in question 
    def _lineEq(self, t):
        P = (self.stLoc + t * self.dirStToEnd)
        #dVal is d in plane eq
        dVal = np.dot(P, self.dirStToEnd)
        return P, dVal

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

    #return min and max bounds of displacement for a single iteration
    def getMinMaxBnds(self): 
        print('parabolicTraj.getMinMaxBnds : Not Implemented')
        res = [np.array([0,0,0]),np.array([self.trajIncr,self.trajIncr,self.trajIncr])]
        return res

    #return a random displacement in line with traj obj params
    def getRandDispVal(self):
        print('parabolicTraj.getRandDispVal : Not Implemented')
        return self.nextLoc - self.curLoc

    #endloc is ignored
    def initTrajIndiv(self,endLoc):

        print('parabolicTraj.initTrajPriv : Not Implemented')

    def calcNewPosIndiv(self, t): 
        print('parabolicTraj.calcNewPosIndiv : Not Implemented')
        self.nextDispVec = self.nextLoc - self.curLoc

    def incrTrajStepIndiv(self,perFrameIncr):
        print('parabolicTraj.incrTrajStepIndiv : Not Implemented')      
        #evolve trajstep
        self.trajStep += perFrameIncr
       
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

        