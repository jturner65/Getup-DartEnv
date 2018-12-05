#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 1 14:09:33 2018

@author: john
"""


import numpy as np

from collections import defaultdict
from abc import ABC, abstractmethod

#class holding functionality to handle reward calculation for an agent
class RL_reward():
    #_name : name of reward pertinent to agent it covers
    #_rwdData : holds a list of tuples of (string type, dictionary of ctor args), one for each reward component function, that describe the function's parameters and calculation used to determine reward
    #_sklHldr : skelHolder for agent this reward applies to
    def __init__(self,_name, _rwdData, _sklHldr):
        self.rwdData = _rwdData
        self.sklHldr = _sklHldr
        #list of reward function components : x[0] is type, and x[1] is dict of args
        self.rewardFunctions = { x[1]['name'] : RL_rewardComp.buildRewardFunc(x[0], x[1], self) for x in _rwdData}
        #reward value calculations : dictionary of calcs for each reward value to derive each reward functions check value
        self.checkValCalcs = {}

    #calculate reward-related values that are set every reset
    def calcResetVals(self):
        #initial copcom distance (COM projected on ground to COP on ground) - want this to always get smaller
        #set initial COMCOPDist for COMCOP loc and velocity calcs
        sklHldr = self.sklHldr
        _,COPVal, _= sklHldr.calcAllContactData()
        self.COMCOPDist = sklHldr.calcCOMCOPDist(sklHldr.getRewardCOM(),COPVal)  
       

    #calculate pre-step (before all frame skips) values used for this reward calculation
    def calcPreStepVals(self):
        self.com_b4_step = self.sklHldr.getRewardCOM()




    #calculate the reward based on each reward function's measured value
    def calcTtlReward(self):
        rwrd=0

        #calculate pertinent post-step values

        for _, rwrdFuncObj in self.rewardFunctions.items():
            #chkCalcFuncName : str rep of function in this class to calculate reward
            #chkCalcArgs : dictionary representation of arguments to pass .__dict__[
            chkVal = getattr(self, rwrdFuncObj.chkCalcFuncName)(rwrdFuncObj.chkCalcArgs)        

            rwrd += rwrdFuncObj.calcReward(chkVal)
        return rwrd

    ##############################################
    # calc for data used for rewards - function calls
    #
    
    #TODO

    #calculate height-based value used for reward
    # def calcChkVal_Height(self, args):
    #     chkVal = 0


    #     return chkVal

    # def calcChkVal_COMVel(self, args):
    #     chkVal = 0
    #     com_now, old_com, comVel
    #     #iterate through list of com vel
    #     for idx in args['comvel_idxs']:

    #     #idx 0 == x, idx 1 == y, idx 2 == z


    #     return chkVal


    
    
    #test passed reward function's performance between minVal and maxVal
    #return dictionary of results and moments analysis, if specified
    def testRwrdFunc(self, rwdFunc, minVal=0, maxVal=100, numVals=10000, calcMmnts=False):
        x = np.linspace(minVal, maxVal, numVals)
        y = [rwdFunc.calcReward(xVal) for xVal in x]
        momentInfo = {}
        if(calcMmnts):
            import scipy.stats as stats
            momentInfo['minAdot'] = np.min(y)
            momentInfo['maxAdot'] = np.max(y)
            momentInfo['meanAdot'] = np.mean(y)
            momentInfo['stdAdot'] = np.std(y)
            momentInfo['skewAdot'] = stats.skew(y)
            momentInfo['kurtAdot'] = stats.kurtosis(y)        
        #if calculating moments, find moments info
        valsDict, valsStr = rwdFunc.getFuncVals()        
        res={}
        res['x']=x
        res['y']=y
        res['mmnts']=momentInfo
        res['valsDict']=valsDict
        res['valsStr']=valsStr
        return res

    #process results from reward component check
    #rwdCompsDict : dict holding : current total reward as accumulated value, and a dictionary of dicts of good or bad result completion keyed by rwrd type
    #rwdType : type of this reward
    #rwdComp : reward value
    #rwdValWatched : list of tuples of descriptor and value used by rwd calculation to determine reward
    #succCond, failCond : reward component specific termination conditions (success or failure)
    #debug : whether this is in debug mode or not
    #dct : dictionary of reward types used holding reward components and list of values used to calculate reward, (dict of lists, with first idx == reward and 2nd idx == list of values watched)
    #dbgStructs : dictionary holding lists of values used for debugging
    def procRwdCmpRes(self, rwdStateDict, rwdType, rwdComp, rwdValWatched, succCond, failCond, debug, dbgStructs):
        rwdStateDict['reward'] += rwdComp
        rwdStateDict['isDone']['good'][rwdType]=succCond
        rwdStateDict['isDone']['bad'][rwdType]=failCond
        #checks for success or failure - should be boolean eqs - set to false if no conditions impact done
        rwdStateDict['isGoodDone'] = rwdStateDict['isGoodDone'] or succCond
        rwdStateDict['isBadDone'] = rwdStateDict['isBadDone'] or failCond
        #dct[rwdType] = [rwdComp, rwdValWatched]
        if (debug):
            strVal = '{0} rwd/pen: {1:3f} | obs vals: [{2}] ||'.format(rwdType, rwdComp,'; '.join(['{}:{:3f}'.format(x[0],x[1]) for x in rwdValWatched]))
            csvStrVal = '{0},{1:3f},[{2}]'.format(rwdType, rwdComp,';'.join(['{}:{:3f}'.format(x[0],x[1]) for x in rwdValWatched]))
            dbgStructs['rwdComps'].append(rwdComp)
            dbgStructs['rwdTyps'].append(rwdType)
            dbgStructs['successDone'].append(succCond)
            dbgStructs['failDone'].append(failCond)
            dbgStructs['dbgStrList'][0] += '\t{}\n'.format(strVal)
            dbgStructs['dbgStrList_csv'][0] += '{},'.format(csvStrVal)
            dbgStructs['dbgStrList'].append(strVal)  
            dbgStructs[rwdType] = [rwdComp, rwdValWatched]


#abstract base class for reward components for an RL experiment
#instancing components will be either linear or exponential
class RL_rewardComp(ABC):
    #initialize reward component calculate variables
    #_name : string name desrcibing what reward pertains to - for display
    #_type : string representation of type of reward - for display
    #-------------below must be set in **_rwdDict----------------------------------
    #_wt : weight of this reward, should default to 1.0
    #_bounded : means reward result can only be non-negative
    #_rwrdFuncStr : string function name for reward calculation - function resides in reward owner
    #_rwrdFuncArgsDict : dictionary of named arguments to go to _rwrdFuncStr
    #_rwdOwnr : owning reward RL_reward object
    def __init__(self, _name,_type, _wt=1.0, _bounded=True, _rwrdFuncStr='', _rwrdFuncArgsDict={}, _rwdOwnr=None):
        self.name=_name
        self.type=_type
        self.wt = _wt
        self.bounded = _bounded
        #owning reward, if exists
        self.rwrdOwnr = _rwdOwnr
        #chkCalcFuncName : str rep of function in owning reward class to calculate reward
        self.chkCalcFuncName = _rwrdFuncStr
        #chkCalcArgs : dictionary representation of arguments to pass
        self.chkCalcArgs = _rwrdFuncArgsDict

    #build and return a reward function based on passed arguments
    #rwrdType needs to be : 'exp', 'lin','parabola','tanh'
    #rwrdArgs : dictionary of arguments to use to build reward - needs to be in expected format for reward constructor
    #_rwdOwnr : RL_reward object owning this component function
    @staticmethod
    def buildRewardFunc(rwrdType, rwrdArgs, _rwdOwnr):
        rwrdArgs['_rwdOwnr']=_rwdOwnr
        if('exp' in rwrdType):
            return exp_Rwrd(**rwrdArgs)
        elif('lin'in rwrdType):
            return lin_Rwrd(**rwrdArgs)
        elif('parabola' in rwrdType):
            return parabolic_Rwrd(**rwrdArgs)
        elif('tanH' in rwrdType):
            return tanH_Rwrd(**rwrdArgs)
        elif('wtVec' in rwrdType):
            return wtdVec_Rwrd(**rwrdArgs)
        else :
            print('Unknown Reward Type : {} -> No reward built'.format(rwrdType))
            return None

    #return pertinent values of this reward function, along with a string representation 
    def getFuncVals(self):
        baseDict = {'name':self.name, 'type':self.type, 'wt':self.wt, 'bounded':self.bounded}
        valsDict = self.getValsIndiv(baseDict)
        #build display string
        valsStr = '|'.join(['%s:: %s' % (k, v) for (k, v) in valsDict.items()])
        return valsDict, valsStr

    #returns a string describing the evaluated reward, and the reward itself
    #for debugging
    def getRwrdAndDesc(self, chkVal):
        rwd = self.calcReward(chkVal)
        return '{} rwd w/{} calc={}'.format(self.name, self.type, rwd), rwd

    @abstractmethod
    def calcReward(self, chkVal): pass
    @abstractmethod
    def getValsIndiv(self, vals): pass

#exponential reward function - exponential reward on scalar quantity
class exp_Rwrd(RL_rewardComp):
    #_var : 'variance' of exponential formulation : defaults to 1.0
    def __init__(self, _name, _rwdDict, _tol=0.0,  _var=1.0, _offset=0.0):
        RL_rewardComp.__init__(_name,'exp', **_rwdDict)
        self.tol=_tol
        if(_var == 0):#bad
            self.var=1.0
        else:
            self.var=_var
        self.offset=_offset

    def calcReward(self, chkVal):
        #TODO if bounded need to calculate bound values - should be restricted to 0->wt?
        if (self.bounded):
            print('NOTE : Bound values for exponential reward not supported yet, returning 0 reward')
            return 0
        return self.wt * (np.exp((chkVal-self.tol)/self.var)-self.offset)

    def getValsIndiv(self,vals):
        vals['var']=self.var
        vals['tol']=self.tol
        vals['offset']=self.offset
        return vals

#linear reward function - linear reward on scalar quantity
class lin_Rwrd(RL_rewardComp):
    #min and max are the caps of possible values used to determine reward - 
    # if chkVal<= min, then rwd val is 0.0, if chkVal>= max then rwd val is 1.0 * wt
    # values between min and max give smooth linear rewards
    def __init__(self,_name, _rwdDict,  _min_x=0.0, _max_x=1.0):
        RL_rewardComp.__init__(_name,'lin',  **_rwdDict)
        self.min_x = _min_x
        self.max_x = _max_x
        diff = self.max_x - self.min_x
        if(diff == 0):
            print('lin_Rwrd : Bad reward function formulation : _min_x == _max_x')
            diff = 1.0
        self.wtOvDiff = self.wt/diff

    def calcReward(self, chkVal):
        #if bounding reward to only be positive and within [0,wt] lims
        if self.bounded:
            if(chkVal <= self.min_x): return 0
            if(chkVal >= self.max_x) : return self.wt 
        return self.wtOvDiff * (chkVal-self.min_x)

    def getValsIndiv(self,vals):
        vals['min_x']=self.min_x
        vals['max_x']=self.max_x
        vals['wtOvDiff']=self.wtOvDiff
        return vals

#parabolic reward function - yields a positive value between min and max passed values, with peak of wt at (max + min)/2.0
#if bounded yields 0 outsided min and max, otherwise yields inverted parabola with roots at min and max and peak value of 1 at xMax
class parabolic_Rwrd(RL_rewardComp):
    #min and max are the caps of possible values used to determine reward - 
    # if chkVal<= min, then rwd val is 0.0, if chkVal>= max then rwd val is 1.0
    # values between min and max give smooth linear rewards
    def __init__(self, _name, _rwdDict, _min_x=0.0, _max_x=1.0):
        RL_rewardComp.__init__(_name,'parabola', **_rwdDict)
        self.minRoot = _min_x
        self.maxRoot = _max_x
        self.xMax = (_min_x + _max_x)/2.0
        if(self.minRoot==self.maxRoot):#if min == max then only chkval==min (or max) yields reward, all others are 0
            self.mult=0.0
        else:
            self.mult = self.wt/((self.xMax-self.minRoot) * (self.xMax - self.maxRoot))

    def calcReward(self, chkVal):
        #check if chkval is exactly equal to peak of parabola
        if(chkVal==self.xMax):return self.wt
        #if bounding reward to only be non-negative
        if(self.bounded) and ((chkVal < self.minRoot) or (chkVal > self.maxRoot)):return 0
        return self.mult*((chkVal - self.minRoot) * (chkVal - self.maxRoot))

    def getValsIndiv(self,vals):
        vals['min_root']=self.minRoot
        vals['max_root']=self.maxRoot
        vals['rootMult']=self.mult
        return vals

#use tanH to calculate reward value, reward will always be between 0 and wt
class tanH_Rwrd(RL_rewardComp):
    #bounded is always true for tanh reward
    #_min_x and _max_x are values corresponding to +/-PI in tanh calc (i.e. yield either .01 or .99 rewards) 
    #in other words, these are min and max bounds of values to receive any variable rewards - values higher than max automatically get ~1, values lower than min get ~0 rwrd
    def __init__(self, _name,  _rwdDict, _min_x=0.0, _max_x=1.0):
        RL_rewardComp.__init__(_name,'tanH', **_rwdDict)
        self.min_x = _min_x
        self.max_x = _max_x
        diff = self.max_x - self.min_x
        if(diff == 0):
            print('tanH_Rwrd : Bad reward function formulation : _min_x == _max_x')
            diff=1.0
        #eq to map : -pi + (x-min)/(max-min) * (pi - -pi)-> sclMult == 2pi/(max-min)
        self.sclMult = 2 * np.pi/diff
        #halfweight used since scaling tanh output by half to force from -1->1 to 0->1
        self.halfWt = .5*self.wt
         

    def calcReward(self, chkVal):
        #thetVal is mapping of chkVal between +/- PI given stated min and max values corresponding to -pi and +pi
        thetVal = -np.pi + (self.sclMult) * (chkVal-self.min_x)
        #tanH varies between -1 and 1, want 0 ->1
        return self.halfWt * (np.tanh(thetVal) + 1.0)

    def getValsIndiv(self,vals):
        vals['min_x']=self.min_x
        vals['max_x']=self.max_x
        vals['sclMult']=self.sclMult
        return vals

#a reward function to manage determining a single scalar reward for a vector quantity, using a weight matrix to accentuate certain
#components of the vector.  This class will act as a wrapper for any of the scalar reward functions, which need to be specified in constructor
class wtdVec_Rwrd(RL_rewardComp):
    #uses a specified function to manage reward calculation, but uses weight matrix to scale input vector (xT * W * x) to yield scalar value to pass to reward function
    def __init__(self, _name, _rwdDict, _rwrdFuncType, func_args):
        RL_rewardComp.__init__(_name,'wtdVec',_bounded=func_args['bounded'], **_rwdDict)
        #wtMat is diagonal matrix used as per-component weights
        self.wtMat = _rwdDict['wtMat']
        self.func_args = func_args  #saving this to just verify that child reward function is instanced properly
        self.rewardFunc = RL_rewardComp.buildRewardFunc(_rwrdFuncType, func_args, self.rwrdOwnr)

    def calcReward(self, chkVec):
        val = np.transpose(chkVec).dot(self.wtMat.dot(chkVec))
        return self.rewardFunc(val)

    #returns a string describing the evaluated reward, and the reward itself
    #for debugging - overriding the base class version
    def getRwrdAndDesc(self, chkVec):
        val = np.transpose(chkVec).dot(self.wtMat.dot(chkVec))
        rwrdFuncDesc, rwd = self.rewardFunc.getRwrdAndDesc(val)
        return '{} rwd w/{} using scalar {}'.format(self.name, self.type, rwrdFuncDesc), rwd

    def getValsIndiv(self,vals):
        vals['wtMat']=self.wtMat
        rwrdValsDict = defaultdict(float)
        rwrdValsDict = self.rewardFunc.getValsIndiv(rwrdValsDict)
        for k,v in rwrdValsDict.items():
            vals['rwdFunc_{}'.format(k)] = v
        return vals
        


