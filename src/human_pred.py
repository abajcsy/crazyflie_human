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
		# rationality coefficient in P(u_H | x, u_R; theta) = e^{1/beta*Q(x,u_R,u_H,theta)}
		self.beta = 1

		# grid world representing the experimental environment
		self.gridworld = GridWorldMDP(height, width, {}, default_reward=-1)

		# tracks the human's state over time
		self.human_traj = None

		# stores occupancy grid, with probability of human at a certain state
		self.occupancy_grid = np.zeros((height,width))

		# stores 4D obstacle computed from the occupancy grid
		self.moving_obstacle = None
	
		# list of known goals where the human might want to go
		self.goal_pos = [height-1, width-1]
		self.goal = self.gridworld.coor_to_state(height-1, width-1)

		# number of timesteps
		self.T = 12

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
		print "	--> currT: ", currT

		#print "human traj latest: " + str(self.human_traj[-1])
		init_state = self.gridworld.coor_to_state(self.human_traj[0][0], self.human_traj[0][1])

		# A numpy.ndarray with dimensions (T x g.rows x g.cols).
		# `state_prob[t]` holds the exact state probabilities for
		# a beta-irrational, softmax-action-choice-over-hardmax-values agent.
		state_prob = inf.state.infer_from_start(self.gridworld, init_state, self.goal,
					T=currT, beta=self.beta, all_steps=False).reshape(self.gridworld.rows, self.gridworld.cols) #.reshape(self.T+1, self.gridworld.rows, self.gridworld.cols)

		self.occupancy_grid = state_prob
		#print "occu grid:", self.occupancy_grid

if __name__ == '__main__':
	human_map = HumanPredMap(10, 10)
	human_map.update_human_traj([0,0])
	human_map.infer_occupancies(0.7)

