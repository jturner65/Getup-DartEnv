#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:09:33 2018

@author: john
"""

import numpy as np

from collections import defaultdict
#for algorithm
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.algos.npo import NPO
#for baseline
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
#for Environment
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
#for policy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

#for reward
from rewardClass import RL_reward


#a class to hold the particulars of an experiment - aggregation of code from dart_env_2bot.py, skelHolders.py, and others
#  including the reward function formulation being used, in the form of multiple reward calculation objects.  
#this is intended to provide not only a flexible and customizable interface for experiments, 
#but also to provide self-documentation and verification for experiments and 
#individual reward components owned by an environment - duplicates will be made for each instanced environment for 
# multithreaded calcs, so should not attempt to preserve any inter-thread data
# 
# The environment will own an experiment, and the experiment will be defined by what is loaded from a config xml file
# The experiment will manage loading/building skeleton holders, all experimental data, the simulation stepping, the reward calculation, etc. 
class RL_Experiment(): 
    #an exeperiment object will either be determined by passed arguments to constructor
    #or a call to loadExperiment, which will load passed file name and use data in file to build 
    #experiment
    def __init__(self, expData, expFileName, loadExp=False, env=None, pol=None, bl=None):
        self.expFileName = expFileName
        #expData is dictionary of passed experimental arguments contains all data to describe experiement, including configuration of reward functions for each agent, held as list of dictionaries
        #if load, then ignore expdata and load experiment found at passed expFileName
        if((loadExp) and (expFileName is not None)):
            self.loadExperiment(expFileName, env, pol, bl)            
        #otherwise build new experiment using expData
        else:
            self._buildExperiment(expData, env, pol, bl)
            #save built experiment
            self.saveExperiment(expFileName)

    #build experiment with self.expData
    def _buildExperiment(self, expData, env=None, pol=None, bl=None, algo=None):
        self.expData = expData    
        #either use trpo or ppo
        self.useTRPO = expData['useTRPO']

        #build environment if none exists
        if(env is None):
            self.env = RL_Experiment._buildEnv(expData['envName'], expData['isGymEnv'])
        else :
            self.env = env

        #build policy if none exists
        if (pol is None):
            self.policy = RL_Experiment._buildPolicy(self.env, expData['polNetArch'])
        else :
            self.policy = pol
        
        #build baseline if none exists
        if (bl is None):
            self.bl = RL_Experiment._buildBaseline(self.env, expData['blArch'], expData['blType'])
        else :
            self.bl = bl
        
        if (algo is None):

            #build dictionary describing algorithm in use using pre-built environment, policy and baseline structures
            #algDefData either loaded from file or specified from expDict
            #defined in batch_polopt.py, shown with ctor defaults
            algoDict = {}
            algoDict['env'] = self.env                          #     :param env: Environment
            algoDict['policy'] = self.policy                    #     :param policy: Policy
            algoDict['baseline'] = self.bl                      #     :param baseline: Baseline

            #get sub-dictionary of algorithm-specific data
            algDefData = self.expData['algDefData']             #   specifically algorithm-related data, defined below

            for k,v in algDefData.items():
                algoDict[k] = v

            self.algo = RL_Experiment._buildRLAlg(self.useTRPO, algoDict, optimizerArgs=None)
        else :
            self.algo = algo

        #expand expData as needed - hold # of training iterations, # of samples/batch size, experimental parameters such as discount, name of environment, name of skel file, type and nature of reward functions, policy and baseline configurations, etc.
        # TODO 

        #build list of init states for all skels used in experiment ?

        #build list of reward functions used by each agent in experiment
        self.agentRwrds = [RL_reward(**x) for x in expData['rwrdDictsPerAgent']]


        
    #load experiment, including building environment, policy and baseline
    def loadExperiment(self, expFileName, env=None, pol=None, bl=None):
        #build an experiment from data held in an experiment file, loaded into self.expData
        print('TODO loading experimental data not yet supported')        
        expData={}
        #expData = -loaded data from expFileName-

        #attempt to load environment, policy, baseline if already built, otherwise set as none or passed values

        self._buildExperiment(expData, env, pol, bl)
        
    def saveExperiment(self, expFileName):
        #save current experimental configurations to file
        print('TODO saving experimental configuration not yet supported') 




    #reset experimental data to start state for all skeletons/skelholders particiapting in this experiment
    def resetExperiment(self):

        pass

    #step experiment - corresponds to stepping environment from policy
    #include policy as part of experiment?  i.e. manage consumption of policy via experiment object?
    def stepExperiment(self):

        pass 

    #################################################
    #   static methods
    #################################################

    #build an environment
    @staticmethod
    def _buildEnv(envName, isGymEnv=True):
        if(isGymEnv):
            env = normalize(GymEnv(envName,record_video=False, record_log=False))
        else :
            env = normalize(envName)       
        return env 

    #build gaussian MLP policy using passed env and polDict arguments
    @staticmethod
    def _buildPolicy(env, polNetArch):
        pol = GaussianMLPPolicy(
            env_spec=env.spec,
            #polNetArch must be tuple - if only 1 value should be followed by comma i.e. (8,)
            hidden_sizes=polNetArch
        )   
        return pol  

    #build baseline
    #env : environment
    #mlpArch : tuple of architecture, if only 1 layer, needs trailing comma
    @staticmethod
    def _buildBaseline(env, blArch, blType='MLP'):
        if ('linear' in blType):
            bl = LinearFeatureBaseline(env_spec=env.spec)
        elif ('MLP' in blType):
            #use regressor_args as dict to define regressor arguments like layers 
            regArgs = dict()
            regArgs['hidden_sizes'] = blArch
            #only used if adaptive_std == True
            regArgs['std_hidden_sizes']= blArch
            #defaults to normalizing so set to false
            regArgs['normalize_inputs'] = False
            regArgs['normalize_outputs'] = False
            #regArgs['adaptive_std'] = True
            #regArgs['learn_std']= False  #ignored if adaptive_std == true - sets global value which is required for all thread instances
            bl = GaussianMLPBaseline(env_spec=env.spec, regressor_args=regArgs)
        else:
            print('unknown baseline type : ' + blType)
            bl = None
        return bl

    @staticmethod
    #build RL algorithm using passed dictionary of values - set to either TRPO or PPO based on what alg we are using
    def _buildRLAlg(useCG, algDict, optimizerArgs=None):
        #RL Algorithm
        if optimizerArgs is None:
                optimizerArgs = dict()

        if useCG :
        #either use CG optimizer == TRPO
            optimizer = ConjugateGradientOptimizer(**optimizerArgs)
        #or use BFGS optimzier == penalized policy optimization TODO can this be an avenue to PPO? does it not require also liklihood truncation?
        else:
            optimizer = PenaltyLbfgsOptimizer(**optimizerArgs)
        #NPO is expecting in ctor : 
        #self.optimizer = optimizer - need to specify this or else defaults to PenaltyLbfgsOptimizer
        #self.step_size = step_size : defaults to 0.01
        #truncate_local_is_ratio means to truncate distribution likelihood ration, which is defined as
        #  lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        # if truncation is not none : lr = TT.minimum(self.truncate_local_is_ratio, lr)        
        #self.truncate_local_is_ratio = truncate_local_is_ratio
           
        algo = NPO(optimizer=optimizer, **algDict)
        return algo

    #call externally 
    #build dictionary of boolean values for algorithm definition
    @staticmethod
    def buildAlgDefBools(plt=False, psPlt=False, ctrAdv=True, posAdv=False, strPath=False, whlPath=True ):
        res = {'plot':plt,'pause_for_plot': psPlt, 'center_adv':ctrAdv,'positive_adv':posAdv,'store_paths':strPath, 'whole_paths':whlPath }
        return res

    #build alg def data dictionary 
    #this dictionary will hold configuration of algorithm definition
    #argument defaults are defaults in ctor of BatchPolopt in batch_polopt.py 
    @staticmethod
    def buildAlgDefDataDict(step_size=0.01,truncate_local_is_ratio=None, scope=None, n_itr=500, start_itr=0, batch_size=5000, max_path_length=500, discount=0.99, gae_lambda=1.0, dictBools=None):
        algDefData = {'step_size':step_size,'truncate_local_is_ratio':truncate_local_is_ratio,      \
                    'scope':scope,'n_itr':n_itr, 'start_itr':start_itr, 'batch_size':batch_size,    \
                    'max_path_length':max_path_length,'discount':discount,'gae_lambda':gae_lambda }

        if dictBools==None :#if no dict passed,then use default
            dictBools = RL_Experiment.buildAlgDefBools()
        for k,v in dictBools.items():
            algDefData[k]=v
        return algDefData  
        #batch_polopt args : 
        # algDefData['scope']=None                      #     :param scope: (dflt : None) Scope for identifying the algorithm. Must be specified if running multiple algorithms simultaneously, each using different environments and policies
        # algDefData['n_itr']=500,                      #     :param n_itr: (dflt : 500) Number of iterations.
        # algDefData['start_itr']=0,                    #     :param start_itr: (dflt : 0) Starting iteration.
        # algDefData['batch_size']=5000,                #     :param batch_size: (dflt : 5000) Number of samples per iteration.
        # algDefData['max_path_length']=500,            #     :param max_path_length: (dflt : 500) Maximum length of a single rollout.
        # algDefData['discount']=0.99,                  #     :param discount: (dflt : .99) Discount.
        # algDefData['gae_lambda']=1,                   #     :param gae_lambda: (dflt : 1) Lambda used for generalized advantage estimation.
        # algDefData['plot']=False,                     #     :param plot: (dflt : False) Plot evaluation run after each iteration.
        # algDefData['pause_for_plot']=False,           #     :param pause_for_plot: (dflt : False) Whether to pause before contiuing when plotting.
        # algDefData['center_adv']=True,                #     :param center_adv: (dflt : True) Whether to rescale the advantages so that they have mean 0 and standard deviation 1.
        # algDefData['positive_adv']=False,             #     :param positive_adv: (dflt : False) Whether to shift the advantages so that they are always positive. When used in conjunction with center_adv the advantages will be standardized before shifting.
        # algDefData['store_paths']=False,              #     :param store_paths: (dflt : False) Whether to save all paths data to the snapshot.
        # algDefData['whole_paths']=True,               #     :param whole_paths: (dflt : True) Whether entire paths are to be used for sampler, or if paths should be truncated
        # algDefData['sampler_cls']=None,               #       :param sampler_cls : appears to be class of sampler to use, if none defaults to BatchSampler
        # algDefData['sampler_args']=None,              #       :param sampler_args : appears to be arguments to send to sampler ctor.  makes empty dict if none
