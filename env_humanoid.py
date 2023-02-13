#!/usr/bin/env python3
import time
import numpy as np
from os import getcwd

import mujoco
from mocap import MocapDM

import gym
from gym.envs.mujoco import MujocoEnv

from pyquaternion import Quaternion

BODY_JOINTS = [
    "chest", "neck", "right_shoulder", "right_elbow", 
    "left_shoulder", "left_elbow", "right_hip", "right_knee", 
    "right_ankle", "left_hip", "left_knee", "left_ankle"
]
BODY_JOINTS_IN_DP_ORDER = [
    "chest", "neck", "right_hip", "right_knee",
    "right_ankle", "right_shoulder", "right_elbow", "left_hip", 
    "left_knee", "left_ankle", "left_shoulder", "left_elbow"
]
DOF_DEF = {
    "chest": 3, "neck": 3, "right_shoulder": 3, "right_elbow": 1, 
    "left_shoulder": 3, "left_elbow": 1, "right_hip": 3, "right_knee": 1, 
    "right_ankle": 3, "left_hip": 3, "left_knee": 1, "left_ankle": 3
}
PARAMS_KP_KD = {
    "chest": [1000, 100], "neck": [100, 10], "right_shoulder": [400, 40], "right_elbow": [300, 30],
    "left_shoulder": [400, 40], "left_elbow": [300, 30], "right_hip": [500, 50],
    "right_knee": [500, 50], "right_ankle": [400, 40], "left_hip": [500, 50],
    "left_knee": [500, 50], "left_ankle": [400, 40]
}
JOINT_UP = np.array([
    1.2, 1.20, 1.2, 1.0, 1.0,  1.0, 0.5, 0.7,  1.50, 2.8, 3.14, 0.7, 1.50, 2.8,
    1.2, 1.57, 1.0, 0.0, 1.0, 1.57, 1.0, 1.2,  1.57, 1.0,  0.0, 1.0, 1.57, 1.0
])
JOINT_DOWN = np.array([
    -1.2, -1.20, -1.2, -1.0, -1.0, -1.0, -3.14, -3.14, -1.50,  0.0, -0.5, -3.14, -1.5,  0.0,
    -1.2, -2.57, -1.0, -2.7, -1.0, -1.0,  -1.0,  -1.2, -2.57, -1.0, -2.7, -1.0 , -1.0, -1.0
])

SIM_FPS, CTRL_FPS = 100, 20 
class DMEnv(MujocoEnv, gym.utils.EzPickle):
    metadata = {
        "render_modes": [ "human", "rgb_array", "depth_array" ],
        "render_fps": CTRL_FPS,
    }

    def __init__(self, env_config):
        self.randStart = env_config["randStart"]
        self.mocap = MocapDM()
        self.xml_file = getcwd() + "/robots/pf_humanoid.xml"
        self.motion_file = getcwd() + "/motions/humanoid3d_walk.txt"
        self._load_mocap(self.motion_file)

        self.errWeights = np.array([
            0.8, 0.8, 0.8, 0.1, 0.1, 0.1,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ])
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(75,), dtype=np.float32)

        self.idx_curr = self.idx_init = -1
        skips = SIM_FPS//CTRL_FPS
        if env_config["render"]:
            MujocoEnv.__init__(self, self.xml_file, skips, obs_space, "human") # for evaluation
        else:
            MujocoEnv.__init__(self, self.xml_file, skips, obs_space) # for train in ray
        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        root = self.data.qpos.flat[2:7].copy() # to exclude root-x & root-y
        config = self.data.qpos.flat[7:].copy() / 3.14   # 28, joint angle
        
        root_v = self.data.qvel.flat[0:6].copy()  # 6, root velocity
        config_v = np.clip(self.data.qvel.flat[6:].copy(), -15, 15) / 15.0 # 28, joint angle vel
        
        contact_force = self.data.sensordata.flat.copy()    # 8, contact force sensor under feet
        contact_force= np.clip(contact_force, 0.0, 50.0) / 50.0 # remove impulse force

        obs = np.concatenate( [root, config, root_v, config_v, contact_force] )
        return obs

    def step(self, action):
        self.do_simulation(action, None)    # in mujoco2.3.0, second param is not in use
        self.idx_curr += 1
        self.idx_curr %= self.mocap_data_len

        # if self.idx_curr==self.mocap_data_len-1:
        #     done, reward = False, self._calc_config_pos_reward()
        if self._early_termination():
            done, reward = True, -5
        else:
            done = False
            reward = 0.8*self._calc_config_pos_reward() + 0.1*self._calc_config_vel_reward() + \
                0.1*self._calc_root_pos_reward()
            # reward = self._calc_config_pos_reward() * self._calc_config_vel_reward() * \
            #     self._calc_root_pos_reward()
            # done, reward = False, 0
        return self._get_obs(), reward, done, {}

    def reset_model(self):
        # print(self.model.nbody) # 1+13, plane + BODY_JOINTS
        # print(self.model.njnt) # 1+28, one floating base & 28 joints
        if self.randStart:
            self.idx_curr = self.idx_init = np.random.randint(0, self.mocap_data_len-1)
        else:
            self.idx_curr = self.idx_init = 0

        # self.total_errs = 0
        qpos = self.mocap.data_config[self.idx_init] + np.random.uniform(-0.02, 0.02,self.model.nq)
        qvel = self.mocap.data_vel[self.idx_init] + np.random.uniform(-0.02, 0.02, self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    # override by pf
    def reset( self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_simulation()
        return self.reset_model()

    def _early_termination(self):
        rootZ = self.data.qpos[2].flat.copy()
        rootQuat = self.data.qpos[3:7].flat.copy()
        currConfig = self.data.qpos[7:].flat.copy()
        targetConfig = self.mocap.data_config[self.idx_curr][7:]

        rootZAxis = Quaternion(rootQuat[0], rootQuat[1], rootQuat[2], rootQuat[3])
        rootZAxis = rootZAxis.rotate(np.array([.0, .0, 1.0]))
        absErrs = np.abs(currConfig-targetConfig) * self.errWeights
        # walk reference, z range from 0.846 to 0.875
        if rootZ<0.7 or rootZ>0.95 or np.median(absErrs)>0.2 or np.max(absErrs)>0.4 or \
           rootZAxis[2]<0.98:
            return True
        else:
            return False

    # def _compute_torque(self, target_pos, target_vel):
    #     dt = self.model.opt.timestep
    #     qpos = self.data.qpos[7:].flat.copy()
    #     qvel = self.data.qvel[6:].flat.copy()
    #     qpos_err = (qpos + qvel*dt) - target_pos
    #     qvel_err = qvel - target_vel
    #     self.total_errs += qpos_err
    #     torque = -self.kps * qpos_err - self.kds * qvel_err - self.kis*self.total_errs
    #     return torque

    def _load_mocap(self, filepath):
        self.mocap.load_mocap(filepath)
        self.mocap_dt = self.mocap.dt
        self.mocap_data_len = len(self.mocap.data)  # 39
        assert len(self.mocap.data) != 0

    def _calc_config_pos_reward(self):
        target_config = self.mocap.data_config[self.idx_curr][7:] # to exclude root joint, 28
        curr_config = self.data.qpos[7:].flat.copy()    # exclude root, 28

        err_configs = np.sum(np.square(curr_config - target_config)*self.errWeights)
        return np.exp(-2*err_configs)

    def _calc_config_vel_reward(self):
        target_vel = self.mocap.data_vel[self.idx_curr][6:] # to exclude root joint
        curr_vel = self.data.qvel[6:].flat.copy()    # exclude root

        err_configs = np.sum(np.square(curr_vel - target_vel)*self.errWeights)
        return np.exp(-0.1*err_configs)

    def _calc_root_orient_reward(self):
        target_root = self.mocap.data_config[self.idx_curr][3:7]
        current_root = self.data.qpos[3:7].flat.copy()  # exclude displacement
        # quaternion formation (w,x,y,z)
        q_0 = Quaternion(current_root[0], current_root[1], current_root[2], current_root[3])
        q_1 = Quaternion(target_root[0], target_root[1], target_root[2], target_root[3])

        q_diff =  q_0.conjugate * q_1
        err_root_orient = abs(q_diff.angle)

        return np.exp(-err_root_orient)

    def _calc_root_pos_reward(self):
        target_root = self.mocap.data_config[self.idx_curr][0:3]
        current_root = self.data.qpos[0:3].flat.copy()
        err = target_root-current_root
        err_root_pos = np.matmul(err, err)
        return np.exp(-10.0*err_root_pos)

    def _mass_center(self, model, data):
        mass = np.expand_dims(model.body_mass, axis=1)  # list of mass
        xpos = data.xipos   # CoMs of Links
        return (np.sum(mass * xpos, axis=0) / np.sum(mass)).copy()



# # TEST
# if __name__ == "__main__":
#     env = DMEnv(randStart=False, render=True)
#     while True:
#         # env.reset_model()
#         for i in range(env.mocap_data_len):
#             # qpos = env.mocap.data_config[i]
#             # qvel = env.mocap.data_vel[i]
#             # env.set_state(qpos, qvel)
#             # mujoco.mj_forward(env.model, env.data)

#             # quat = env.data.qpos[3:7].flat.copy()
#             # quat = Quaternion(quat[0], quat[1], quat[2], quat[3])   #wxyz
#             # axis = quat.rotate(np.array([0,0,1.0]))
#             # print(f"{i}: {axis}")
#             env.render()

# if __name__ == "__main__":
#     env = DMEnv({"randStart":False, "render":True})
#     env.reset_model()
#     # print(env.model.body_mass)
#     # exit(0)
#     # print(env.observation_space)
#     # print(env.action_space)

#     rewards, errs = [], []
#     np.random.seed(int(time.time_ns())%100)
#     while True:
#         obs = env.reset()

#         pos = np.zeros(28)
#         for i in range(50):
#             obs, reward, done, info = env.step(pos)
#             env.render()

#         pos = env.action_space.sample()
#         for i in range(50):
#             obs, reward, done, info = env.step(pos)
#             errs.append( (env.data.qpos[7:].flat.copy() - pos).tolist() )
#             env.render()

#         pos = env.action_space.sample()
#         for i in range(50):
#             obs, reward, done, info = env.step(pos)
#             errs.append( (env.data.qpos[7:].flat.copy() - pos).tolist() )
#             env.render()
#         break
#     env.close()

#     import matplotlib.pyplot as plt
#     errs = np.transpose(errs)
#     errs = errs[-14:]
#     for i,item in enumerate(errs):
#         plt.plot(item, c=[0,0,1.0])
#     plt.show()