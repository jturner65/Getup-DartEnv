from gym.envs.dart.dart_env import DartEnv
# ^^^^^ so that user gets the correct error
# message if pydart is not installed correctly
#added by JT
from gym.envs.dart.dart_env_2bot import DartEnv2Bot, skelHolder


from gym.envs.dart.cart_pole import DartCartPoleEnv
from gym.envs.dart.hopper import DartHopperEnv
from gym.envs.dart.cartpole_swingup import DartCartPoleSwingUpEnv
from gym.envs.dart.reacher import DartReacherEnv
from gym.envs.dart.cart_pole_img import DartCartPoleImgEnv
from gym.envs.dart.walker2d import DartWalker2dEnv
from gym.envs.dart.walker3d import DartWalker3dEnv
from gym.envs.dart.inverted_double_pendulum import DartDoubleInvertedPendulumEnv
from gym.envs.dart.dog import DartDogEnv
from gym.envs.dart.reacher2d import DartReacher2dEnv
#added by JT
from gym.envs.dart.standUp3d import DartStandUp3dEnv
from gym.envs.dart.standUp3d_FullBody import DartStandUpFullBody3dEnv
from gym.envs.dart.standUp2d import DartStandUp2dEnv
from gym.envs.dart.standUp3dTorque import DartStandUp3dTrqueEnv
from gym.envs.dart.standUp3dDemo import DartStandUp3dDemo
#main standup w/assist environment
from gym.envs.dart.standUp3d_2Bot import DartStandUp3dAssistEnv
#standup 2bot trained with constraint
from gym.envs.dart.standUp3d_2Bot_Cnstrnt import DartStandUp3dAssistEnvCnstrnt

#environment to test GAE paper results in dart
from gym.envs.dart.standUp3d_GAE import DartStandUp3dGAE
#environment that will consume Akanksha's policy
from gym.envs.dart.standUp3d_AKCnstrnt import DartStandUp3dAssistEnv_AKCnstrnt
#Akanksha's env
from gym.envs.dart.kimaStandUp import DartKimaStandUpEnv
#from gym.envs.dart.human_walker import DartHumanWalkerEnv

