import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
#         if np.sqrt(sum(self.sim.pose[:3]**2)) != 0:
#             normalized_sim_pose = self.sim.pose[:3] / np.sqrt(sum(self.sim.pose[:3]**2))
#         else:
#             normalized_sim_pose = self.sim.pose[:3]
#         if np.sqrt(sum(self.target_pos**2)) != 0:
#             normalized_target_pos = self.target_pos / np.sqrt(sum(self.target_pos**2))
#         else:
#             normalized_target_pos = self.target_pos
#         reward = 1 - (np.sqrt((normalized_sim_pose[0] - normalized_target_pos[0])**2 + 
#                           (normalized_sim_pose[1] - normalized_target_pos[1])**2 + 
#                           (normalized_sim_pose[2] - normalized_target_pos[2])**2))

        distance_from_target_penalty = (abs(self.sim.pose[0] - self.target_pos[0]) + 
                                        abs(self.sim.pose[1] - self.target_pos[1]))
    
        reward_for_vertical_movement = self.sim.pose[2] + self.sim.v[2]
        
        reaching_target_penalty = abs(self.sim.pose[2] - self.target_pos[2])
        
        reward = 1 - 0.01*distance_from_target_penalty + 0.2*reward_for_vertical_movement - 0.05*reaching_target_penalty
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        rotor_speeds = [np.mean(rotor_speeds)]*self.action_size
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        reward = np.tanh(reward)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state