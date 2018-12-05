#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 08:48:44 2018

@author: john
"""

#Set Ball Vel cntrol : cur pos : [ 0.47343  0.66267  0.01955] | vel: [ 0.  0.  0.] | des vel : [ 3.92849 -1.17907 -1.05992] | des frame vel : [ 7.85698 -2.35813 -2.11984]
#avg vel seen : [ 1.96425 -0.58953 -0.52996]
#
#!!!!!!! gaussTraj::calcNewPosIndiv :stPos : [ 0.47343  0.66267  0.01955] | endPos : [ 0.96739  1.45324  0.24909] | next T : 0.007969243895825772 | cur T : 0.007969243887048658 | distToMove : 8.777114451907764e-12
#        scale : 0.9529847083526161 | curPos : [ 0.51271  0.65088  0.00895] | nextPos : [ 0.54605  0.69822 -0.03131] | nextDispVec : [ 0.03333  0.04734 -0.04026] | nextTrajIncr : 0.04650545981554231
#
#followTraj::_setCoeffsSmoothAccel :  p0 : [ 0.51271  0.65088  0.00895] p1 : [ 0.54605  0.69822 -0.03131] v0 : [ 3.92849 -1.17907 -1.05992] v1 : [ 0.  0.  0.] T : 0.01 coeffs : [array([ -29.76662,  295.64073, -148.29374]), array([ 3.92849, -1.17907, -1.05992]), array([ 0.51271,  0.65088,  0.00895])]
#Step : 1 : AssistComponent [ 0.03333  0.04734 -0.04026] : Reward : -15.90081 Done : False
#Rwd Comps :
#        action rwd/pen: 0.999998 | obs vals: [actSqMag:0.000010; wt:1.000000; tol:0.000000; var:4.795832] | gdone : False | bdone :False ||
#        height rwd/pen: 0.004052 | obs vals: [htDiffFrStand:-0.892451; heightAboveAvgFoot:0.661561; ratioToStand:0.000405; wt:10.000000; tol:0.000000; var:0.500000] | gdone : False | bdone :False ||
#        lFootMovDist rwd/pen: 10.000000 | obs vals: [lFootMovDist:0.007426] | gdone : False | bdone :False ||
#        rFootMovDist rwd/pen: 10.000000 | obs vals: [rFootMovDist:0.003933] | gdone : False | bdone :False ||
#        comcop rwd/pen: 3.874271 | obs vals: [comCopDotProd:0.440129; wt:20.000000] | gdone : False | bdone :False ||
#        UP_COMVEL rwd/pen: -0.287276 | obs vals: [UP_COMVEL:-0.018603; dir(idx):1.000000; minCOMVel:0.001000; maxCOMVel:2.750000; wt:10.000000] | gdone : False | bdone :False ||
#        X_COMVEL rwd/pen: 0.853312 | obs vals: [X_COMVEL:-0.068099; dir(idx):0.000000; minCOMVel:-0.500000; maxCOMVel:0.900000; wt:1.000000] | gdone : False | bdone :False ||
#        Z_COMVEL rwd/pen: 0.999993 | obs vals: [Z_COMVEL:-0.000815; dir(idx):2.000000; minCOMVel:-0.300000; maxCOMVel:0.300000; wt:1.000000] | gdone : False | bdone :False ||
#        kneeAction rwd/pen: -0.000000 | obs vals: [htsclFact:0.030736; kneeActVal:-0.000000; leftKnee_A:0.000000; rightKnee_A:0.000000; wt:10.000000] | gdone : False | bdone :False ||
#        matchGoalPose rwd/pen: 0.018349 | obs vals: [goalPoseVariance:15.750699; htsclFact:0.030736; wt:10.000000] | gdone : False | bdone :False ||
#        assistFrcPen rwd/pen: -42.363506 | obs vals: [numFrames:1; avgCFrcMag:423.6350566897945; cBodyMG:[ 0.   -9.81  0.  ]; wt:0.1; 
#        Step 0:CnstrFrc : [ 392.84911 -117.90669 -105.99175] ] | gdone : False | bdone :False ||
                                                    
import numpy as np                                                   
                                                    
cnstrntFrc =  np.array([ 392.84911, -117.90669, -105.99175] )
cBodyMG = np.array([ 0.,   -9.81,  0.  ])                                              
vel0 = np.array([0,0,0])
vel1 = np.array([ 3.92849, -1.17907, -1.05992])
dt = .01

accel = (vel1 - vel0)/dt
#the above is exactly the constraint force seen

#with per-frame step of 3 skips
#desAccel per TS : [32.68711 -924.7111   273.54119] | accel coeff : [  24.51533 -693.53332  205.15589] 
#from quartic
da = np.array([32.68711, -924.7111,   273.54119])
ac = np.array([  24.51533, -693.53332,  205.15589])
da/ac
-379.61768 /-284.71326 

#from cubic 4/3
1661.52188 / 1246.14141 
2392.51618 / 1794.38714


#with per frame step of 4 skips 3/2
 35.75456 / 23.83637 
 
#with frame skip of 2, multiplier is 1.0
----------------------- Set Ball Vel cntrol : cur pos : [ 0.43359  0.70587  0.35412] | vel: [ 0.05814 -0.14661 -0.05542] | des vel : [ 0.10336 -0.26064 -0.09852] | 
accel coeff : [ 32.30112 -81.44917 -30.78693]  | advTrackedPosition : actual accel seen per ts : [  45.22156 -114.02883  -43.1017 ] derived from postion : [ -4.30682  10.85989   4.10492] next Desired Loc : [ 0.43364  0.70573  0.35407] 
skelHolders::aggregatePerSimStepRWDQuantities : perStepCnstrntMoveExtFrc : [  45.22156 -114.02883  -43.1017 ]

----------------------- Set Ball Vel cntrol : cur pos : [ 0.43369  0.70561  0.35402] | vel: [ 0.10336 -0.26064 -0.09852] | des vel : [ 0.13566 -0.34209 -0.12931] | 
accel coeff : [ 32.30112 -81.44917 -30.78693]  | advTrackedPosition : actual accel seen per ts : [ 32.30112 -81.44917 -30.78693] 
 derived from postion : [-32.30112  81.44917  30.78693] next Desired Loc : [ 0.43376  0.70543  0.35396] 

nextLoc = np.array([ 0.4746, 0.66008,  0.01097] )
curLoc = np.array( [ 0.47524,  0.65919,  0.01088])
(nextLoc - curLoc)/.001

v1 = np.array([ 0.10336, -0.26064, -0.09852] )
v0 = np.array([ 0.05814, -0.14661, -0.05542])
p1 = np.array([ 0.43369,  0.70561,  0.35402] )
p0 = np.array([ 0.43359,  0.70587,  0.35412])
#delX = (v1 + v0)/2 * dt
#
(0.43369 - 0.43359)/.001