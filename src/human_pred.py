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

from crazyflie_human.msg import OccupancyGridTime, ProbabilityGrid

A = GridWorldMDP.Actions

class HumanPredMap(object):
	"""
	This class models and predicts human motions in a 2D planar environment.
	It stores:
		- human's tracked trajectory 
		- occupancy grid of states human is likely to go to
		- moving obstacle representing human future motion
	"""

	def __init__(self, height, width, res, fwd_tsteps, fwd_deltat, goals, init_beta):

		self.height = height
		self.width = width
		self.resolution = res

		# grid world representing the experimental environment
		self.gridworld = GridWorldMDP(int(self.height), int(self.width), {}, default_reward=-4)

		# number of steps that we do forward prediction for
		self.fwd_tsteps = fwd_tsteps

		# forward timestep for prediction
		self.delta_t = fwd_deltat

		# TODO we need to have a goal set
		# list of known goals where the human might want to go, x = row, y = column
		self.goals = goals

		# TODO THIS IS TEMPORARY, FOR TESTING JUST DO 1 GOAL
		self.goal_pos = self.real_to_sim_coord(self.goals[0])
		self.goal = int(self.gridworld.coor_to_state(self.goal_pos[0],self.goal_pos[1]))

		# rationality coefficient in P(u_H | x, u_R; theta) = e^{1/beta*Q(x,u_R,u_H,theta)}
		self.beta = init_beta

		# tracks the human's state over time
		self.human_traj = None

		# stores 2D array of size (fwd_tsteps) x (height x width) of probabilities
		self.occupancy_grids = None
	
	def update_human_traj(self, newstate):
		"""
		Given a new sensor measurement of where the human is, update the tracked
		trajectory of the human's movements.
		"""
		if self.human_traj is None:
			self.human_traj = np.array([newstate])
		else:
			self.human_traj = np.append(self.human_traj, np.array([newstate]), 0)

	# TODO we need to have beta updated over time, and have a beta for each goal
	def infer_occupancies(self):
		"""
		Using the current trajectory data, recompute a new occupancy grid
		for where the human might be
		"""
		if self.human_traj is None:
			print "Can't infer occupancies -- human hasn't appeared yet!"
			return 

		# TODO update beta here

		corrected = self.real_to_sim_coord(self.human_traj[-1])
		print "(real) human traj latest: " + str(self.human_traj[-1])
		print "(sim) corrected human traj: ", corrected

		init_state = self.gridworld.coor_to_state(int(corrected[0]), int(corrected[1]))

		# returns all state probabilities for
		# timesteps 0,1,...,T in a 2D array. (with dimension (T+1) x (height x width)
		self.occupancy_grids = inf.state.infer_from_start(self.gridworld, init_state, self.goal, T=self.fwd_tsteps, beta=self.beta, all_steps=True)

		#print self.occupancy_grids	

	def interpolate_grid(self, tstep_future):
		"""
		Interpolates the grid at the current time
		"""
		if self.occupancy_grids is None:
			print "Occupancy grids are not created yet!"
			return None

		if tstep_future < 0:
			print "Can't interpolate for negative time!"
			return None

		return self.occupancy_grids[int(tstep_future)]
		
	def real_to_sim_coord(self, real_coord):
		"""
		Takes [x,y] coordinate in the ROS real frame, and returns a shifted
		value in the simulation frame
		"""
		return [int(real_coord[0]+self.width/2), int(real_coord[1]+self.height/2)]

if __name__ == '__main__':
	human_map = HumanPredMap(10, 10)
	human_map.update_human_traj([0,0])
	human_map.infer_occupancies(0.7)

