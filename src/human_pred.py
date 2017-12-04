#!/usr/bin/env python2.7
from __future__ import division
import os
import sys

# Get the path of this file, go up two directories, and add that to our 
# Python path so that we can import the pedestrian_prediction module.
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

import numpy as np

from pedestrian_prediction.pp.mdp import GridWorldMDP
from pedestrian_prediction.pp.inference import hardmax as inf
from pedestrian_prediction.pp.plot import plot_heat_maps

from human_msgs import OccupancyGridTime, ProbabilityGrid

A = GridWorldMDP.Actions

class HumanPredMap(object):
	"""
	This class models and predicts human motions in a 2D planar environment.
	It stores:
		- human's tracked trajectory 
		- occupancy grid of states human is likely to go to
		- moving obstacle representing human future motion
	"""

	def __init__(self, height, width):
    self.height = height
    self.width = width
    self.resolution = 10

		# rationality coefficient in P(u_H | x, u_R; theta) = e^{1/beta*Q(x,u_R,u_H,theta)}
		self.beta = 0.5

		# grid world representing the experimental environment
		self.gridworld = GridWorldMDP(height, width, {}, default_reward=-4)

		# tracks the human's state over time
		self.human_traj = None

    # number of steps that we do forward prediction for
    self.fwd_pred_tsteps = 3

		# stores occupancy grid list, with probability of human at a certain state
    # at each time in the future
		self.occupancy_grids = np.zeros((self.fwd_pred_tsteps,height,width))

		# stores 4D obstacle computed from the occupancy grid
		self.moving_obstacle = None
	
		# list of known goals where the human might want to go
		# x = row, y = column
		self.goal_pos = [height-1, width-1] #width//2]				
		self.goal = self.gridworld.coor_to_state(height-1, width-1)#width-1)


	def update_human_traj(self, newstate):
		"""
		Given a new sensor measurement of where the human is, update the tracked
		trajectory of the human's movements.
		"""
		if self.human_traj is None:
			self.human_traj = np.array([newstate])
		else:
			self.human_traj = np.append(self.human_traj, np.array([newstate]), 0)
			#print "human_traj: ", self.human_traj

	def infer_occupancies(self, time):
		"""
		Using the current trajectory data, recompute a new occupancy grid
		for where the human might be
		"""
		if self.human_traj is None:
			print "Can't infer occupancies -- human hasn't appeared yet!"
			return 

		# approximate the current timestep
		currT = int(np.round(time))
		#print "	--> currT: ", currT

		print "human traj latest: " + str(self.human_traj[-1])
		init_state = self.gridworld.coor_to_state(
                      int(self.human_traj[-1][0]*self.resolution), 
                      int(self.human_traj[-1][1]*self.resolution))
		print "init state: ", init_state

    for t in range(1,self.fwd_pred_tsteps):
		  # A numpy.ndarray with dimensions (g.rows x g.cols).
		  # `state_prob` holds the exact state probabilities for
		  # a beta-irrational, softmax-action-choice-over-hardmax-values agent.
		  result_grid = inf.state.infer_from_start(self.gridworld, init_state, self.goal, T=t, beta=self.beta, all_steps=False) 
		  state_prob = result_grid.reshape(self.gridworld.rows, self.gridworld.cols)


      # TODO i need to update the beta after the inference...?
		  self.occupancy_grids[t-1] = state_prob

if __name__ == '__main__':
	human_map = HumanPredMap(10, 10)
	human_map.update_human_traj([0,0])
	human_map.infer_occupancies(0.7)

