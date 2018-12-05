#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:09:33 2018

@author: john
"""
#all skeleton holders used by dart_env_2bot

import numpy as np
from collections import defaultdict
import json



#class holding functionality to handle assistance components used in a RL-driven environment with multiple interactive agents.
#assistive components contribute to an observation, impact the environment, usually are multidimensional, 
# always have : bounds of allowable values, methods of sampling within those bounds, a current value, 
# may have : alternate representations (i.e. frc mult and frc)
class Assist:
    """
        dim : size of assistive component to observation
        assistVals : initial values to use
        assistFuncs : method of deriving each dim of assistive component (i.e. method of sampling and nature of sample space (i.e. uniform, gaussian, list of values, etc)) list of tuples - (idx list, string method)
        assistBnds : bounds on each dim of assistive component - list of lists : idx 0 is min, idx 1 is max, idx 2 is mean, idx 3 is std
        cmpNames : name of each dim of assistive component
        frcIdxs : idxs of frc component of assistance - used to derive actual force (if len is 0 then no force component)
        env : environment that owns this assistive component
    """
    #ids to use for boolean flags
    flagIds = ['hasForce', 'useMultNotForce', 'usePresetAssist' , 'dispCnstrnt','dbgAssistData']

    #load predifined assist values from file filename - use this to actually build an assist to use.
    @classmethod
    def loadAssist(cls, _filenameRaw, _env):
        import os
        _filename = os.path.join(os.path.expanduser(_filenameRaw))
        with open(_filename) as jsonData:
            jsonStr = jsonData.read()
        jsonToDict = json.loads(jsonStr)        
        
        dct = jsonToDict['assistValsDict']
        jsonToDict['assistVals'] = np.array(dct['__ndarray__'], dtype=dct['dtype']).reshape(dct['shape'])
        jsonToDict['env']=_env
        del jsonToDict['assistValsDict']
        obj = cls(**jsonToDict)
        #TODO verify obj's config matches _filename - if not, resave obj
        objFileBaseName = obj._buildFileName()
        obj.savedFileName=_filename
        if(objFileBaseName not in _filename):
            print('Object is different than file name suggests. Resaving object to appropriate file name')
            _, objFileName=obj.buildObjFileDirAndName()
            _=obj.saveAssist(objFileName)
        return obj

    #ctor only used to build initial assist and save it, and then to remake assist when loaded
    def __init__(self, dim, assistVals, assistFuncs, assistBnds, cmpNames, frcIdxs, env, useMultNotForce=True, flags=None):
        #dim is how large force is, _cmpFuncs are 
        self.dim = dim
        #idxs in assist component vector corresponding to forces, if any
        self.frcIdxs = frcIdxs

        #owning environment
        self.env = env
        #random generator
        self.rndGen = env.np_random
        #ANA skel holder
        self.ANASkelHldr = env.skelHldrs[self.env.humanIdx]
        
        #names of components
        self.cmpNames = cmpNames
        #initial values of all assist components
        self.assistVals = assistVals
        #list of tuples - (idx list of idxs, string for method to use) for how each dim of assist component is derived
        self.assistFuncs = assistFuncs
        #initialize bounds ara on allowable assistive components - idx 0 is min, idx 1 is max, idx 2 is mean, idx 3 is std - 4 bounds tracked
        self.assistBnds = [[0 for x in range(self.dim)] for y in range(4)]
        #whether or not this assist object has a particular kind of bound
        self.hasBndType = 0
        #build actual bounds
        self.setAssistBounds(list(range(self.dim)), mins=assistBnds[0], maxs=assistBnds[1], means=assistBnds[2], stds=assistBnds[3])
        #build assistGen = function generators for synthesizing assistance values
        self._buildAssistFuncs()
        
        if(flags is None):
            #use this for flags for various usage rules for this assistive component
            self._initFlags()
            #derive each assistive component - first force if exists
            self.flags['hasForce'] = (len(frcIdxs) > 0)
            self.flags['useMultNotForce'] = useMultNotForce
        else :
            self.flags = flags
        #build direct access to force and force mult components
        if self.flags['hasForce']:            
            self._buildCurFrcVals()


    #initialize boolean flag array
    def _initFlags(self):
        self.flags = defaultdict(int)
        for x in Assist.flagIds :
            self.flags[x]=False


    #process bound modification - idx is bound type index in assistBnds array, dimIdxs are actual dim idxs receiving bound update, bnds is bound array of data to set 
    #if bnds is specifically None then clear out bound and remove bound in hasBndType flag
    #idx 0 is min, idx 1 is max, idx 2 is mean, idx 3 is std
    def _procBnds(self, idx, dimIdxs, bnds):
        if (bnds is None):
            mask = (2**idx) ^ 0xFFFFFFFF
            self.hasBndType &= mask #should clear flag
            self.assistBnds[idx] = []
            return
        if len (bnds) > 0 :
            i = 0
            self.hasBndType |= 2**idx
            for didx in dimIdxs : 
                self.assistBnds[idx][didx] = bnds[i]
                i+=1
        

    #pass only arrays we wish to set - mins/maxs/means or stds only hold len(idx) values
    #make lists of values contiguous for idxs listed in dimIdxs
    #pass array == None to remove a bound type
    def setAssistBounds(self, dimIdxs, mins=[], maxs=[], means=[], stds=[]):
        #bounds on allowable assistive components - idx 0 is min, idx 1 is max, idx 2 is mean, idx 3 is std
        self._procBnds(0, dimIdxs, mins)
        self._procBnds(1, dimIdxs, maxs)
        self._procBnds(2, dimIdxs, means)
        self._procBnds(3, dimIdxs, stds)
        #build assistGen = function generators for synthesizing assistance values
        self._buildAssistFuncs()

    #build list of lambdas that will calculate values based on specified assistFuncs and assistBnds
    #self.assistFuncs holds list of tuples of list of idxs and string of method to use to calculate
    def _buildAssistFuncs(self):
        assistGen = [None]*self.dim #each element is a list of function, and args
        for tup in self.assistFuncs : 
            idxs = tup[0]
            typeStr = tup[1]
            if 'uniform' in typeStr: #using uniform sample between min (idx 0) and max (idx 1) in bounds
                for idx in idxs : #call function with (**<dict name>)
                    assistGen[idx] = {'func':self.rndGen.uniform, 'args':{'low':self.assistBnds[0][idx],'high':self.assistBnds[1][idx]}}
            elif 'gauss' in typeStr:
                for idx in idxs : #call function with (**<dict name>) - assumes mean and std are idxs 2 and 3
                    assistGen[idx] = {'func':self.rndGen.normal, 'args':{'loc':self.assistBnds[2][idx],'scale':self.assistBnds[3][idx]}}             
            #elif 'list' in typeStr :            
            else :
                print('Unknown typeStr : {} for idxs : {} - defaulting to uniform'.format(typeStr, idxs))
                for idx in idxs : #call function with (**<dict name>)
                    assistGen[idx] = {'func':self.rndGen.uniform, 'args':{'low':self.assistBnds[0][idx],'high':self.assistBnds[1][idx]}}
        self.assistGen = assistGen

    #set booleans to denote how to consume this assistive component
    #_flagIds : ids in default dict of flags to set - must be a list
    #_flagVals : values to set these ids to - must be a list of ints of either 0 or 1
    def setAssistFlags(self, _flagIds, _flagVals ):
        self.flags.update(zip(_flagIds, _flagVals))
        
    def _buildCurFrcVals(self):
        frcMult = self.assistVals[self.frcIdxs]
        self.frcMult = frcMult
        self.frcVals = self.getForceFromMult(frcMult)
        norm=np.linalg.norm(frcMult)
        if (norm == 0) :
            self.frcDir = np.array([0,0,0]) 
        else : 
            self.frcDir = (frcMult / norm)

    #return appropriate force from passed frc multiplier
    def getForceFromMult(self, frcMult):
        return self.ANASkelHldr.mg * frcMult

    #return appropriate force multiplier from passed force
    def getMultFromFrc(self, frc):
        return frc/self.ANASkelHldr.mg  

    def getCurrAssistDim(self):
        return self.dim

    #get current assist vector component of observation
    def getCurrAssist(self):   
        #defaults to frcComp using frcMults
        assistComp = np.copy(self.assistVals)
        #use actual force instead of multiplier for observation
        if self.flags['hasForce'] and not self.flags['useMultNotForce'] :
            assistComp[self.frcIdxs] = self.frcVals
        return assistComp

    #these assume there is a force value
    #get current assist force
    def getCurrAssistFrc(self):
        return self.frcMult if self.flags['useMultNotForce'] else self.frcVals
    #set current force value
    def setCurrAssistFrc(self, _assistFrc):
        self.assistVals[self.frcIdxs] = _assistFrc
        self._buildCurFrcVals()

    #set desired current assistive components
    def setCurrAssist(self, _assist, _idxs=[]):
        if len(_idxs) > 0 :
            self.assistVals[_idxs]=_assist[_idxs]
        else : 
            self.assistVals = _assist
        #rebuild the force values if they exist
        if self.flags['hasForce']:
            self._buildCurFrcVals()

    #return the names of each dof of the assist vector
    def getAssistObsNames(self):
        names = list(self.cmpNames)
        if self.flags['hasForce'] and not self.flags['useMultNotForce'] :  
            rpls = ['frc x', 'frc y', 'frc z'] 
            for (idx, rpl) in zip(self.frcIdxs, rpls):
                names[idx]=rpl
        return names

    #return a new assist component on reset
    def resetAssist(self):
        assistVals = [x['func'](**x['args']) for x in self.assistGen]
        return assistVals

    ########################################
    # json and file io to save this assist object

    def _buildDictFromNPAra(self, npAra):
        return dict(__ndarray__= npAra.tolist(),dtype=str(npAra.dtype),shape=npAra.shape) 
    
    def toJSON(self):
        d = {}# dim, assistVals, assistFuncs, assistBnds, cmpNames, frcIdxs, env):       
        d['dim']=self.dim
        d['assistFuncs']= self.assistFuncs
        d['cmpNames']=self.cmpNames
        d['assistValsDict']= self._buildDictFromNPAra(self.assistVals)
        d['assistBnds'] = self.assistBnds
        d['frcIdxs']=self.frcIdxs
        d['flags']=self.flags        
        return json.dumps(d, sort_keys=True, indent=4)
    
    def compThisObjToThat(self, obj):
        for k,v in self.__dict__.items() : 
            objV = obj.__dict__[k]
            print('{} : self.{}(={}) == obj.{}(={})'.format( (v==objV), k, v, k, objV))


    def _buildFileName(self):
        #strRes = 'assist_d_{}_cmpNames_{}_initVals_{}_useFrcObs_{}'.format(self.dim, self.cmpNames, self.assistVals.tolist(), ('Y' if self.flags['hasForce'] and not self.flags['useMultNotForce'] else 'N'))
        bndNameAra = ['min','max','mean','std']
        bndsStrAra = []
        for i in range(len(self.assistBnds)) :
            if (self.hasBndType | 2**i == self.hasBndType) :
                bnd = self.assistBnds[i]
                bndsStrAra.append( ''.join([bndNameAra[i],'_'.join(['{:.3f}'.format(x) for x in bnd])]))

        strRes = 'assist_d_{}_cmpNames_{}_bndVals_{}_useFrcObs_{}'.format(self.dim, self.cmpNames, '_'.join(bndsStrAra), ('Y' if self.flags['hasForce'] and not self.flags['useMultNotForce'] else 'N'))
        strRes = strRes.replace('[','').replace(']','').replace(',','_').replace(' ','').replace('\'','')   
        return strRes

    def dispDebugInfo(self):
        print(self.toJSON())
    
    #build a destination directory and a file name for this assist object
    #nameProto : some custom component
    def buildObjFileDirAndName(self, rootDir='~/env_expData/assist/'):
        import os
        baseName = self._buildFileName()
        objDir = os.path.join(os.path.expanduser(rootDir) + baseName)
        if not os.path.exists(objDir):
            os.makedirs(objDir)
        objFileName = os.path.join(objDir, baseName +'_'+self.env.datetimeStr +'.json')         
        return objDir, objFileName

    #save the current assist object
    #does not save class variables - if flagIds changes, shouldn't have an impact though, since flags is a defaultdict
    def saveAssist(self, filename):            
        jsonStr = self.toJSON()
        with open(filename, "w") as jsonFile:
            print(jsonStr, file=jsonFile)
        self.savedFileName = filename
        return jsonStr

