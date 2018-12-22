import os
import numpy as np
from gym.envs.robotics.robot_env import RobotEnv
from gym.envs.robotics import utils
from gym.envs.robotics.fetch_env import goal_distance


class PegInsertionEnv(RobotEnv):
    def __init__(self):
        initial_qpos = initial_qpos = {
            'r_shoulder_pan_joint': 0.1,
            'r_shoulder_lift_joint': 0.1,
            'r_upper_arm_roll_joint': -1.54,
            'r_elbow_flex_joint': -1.7,
            'r_forearm_roll_joint': 1.54,
            'r_wrist_flex_joint': -0.2,
            'r_wrist_roll_joint': 0,
        }
        RobotEnv.__init__(
            self,
            model_path=os.path.join(os.path.dirname(__file__), 'assets/pr2_arm3d.xml'),
            n_substeps=5,
            n_actions=7,
            initial_qpos=initial_qpos
        )
        self.distance_threshold = 0.05

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        return -goal_distance(achieved_goal, goal)

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _set_action(self, action):
        assert action.shape == (7, )
        action = np.clip(action, -1, +1)
        utils.ctrl_set_action(self.sim, action)

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.sim.data.qpos,
                self.sim.data.qvel,
                self.sim.data.get_site_xpos('leg_bottom'),
                self.sim.data.get_site_xpos('leg_top'),
                #self.sim.data.get_site_xvelp('leg_bottom'),
                #self.sim.data.get_site_xvelp('leg_top'),
                #np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )
        achieved_goal = np.concatenate([
            self.sim.data.get_site_xpos('leg_bottom'),
            self.sim.data.get_site_xpos('leg_top')])
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        lookat = [0.0, 0.0, -0.2]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = -90.
        self.viewer.cam.elevation = -30.

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        return np.asarray([0.0, 0.3, -0.5, 0.0, 0.3, -0.2])

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()
