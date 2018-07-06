#!/usr/bin/env python2.7
from __future__ import division
import rospy
import sys, select, os
import numpy as np
import time
import copy
import math

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Pose2D, Vector3
from nav_msgs.msg import OccupancyGrid
from crazyflie_human.msg import OccupancyGridTime, ProbabilityGrid

class MultiHumanPrediction(object):
	"""
	This class:
		- listens to multiple human occupancy grids
		- aggregates all occupancy grids (one for each human) into a 
		single occupancy grid by noisy-ORing the probabilities
	"""

	def __init__(self):

		# create ROS node
		rospy.init_node('multi_human_prediction', anonymous=True)

		# load all the prediction params and setup subscriber/publishers
		self.load_parameters()
		self.register_callbacks()

		rate = rospy.Rate(10) 

		while not rospy.is_shutdown():
			# shudown upon ENTER
			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break

			# update the final occupancy grid by merging all human grids
			self.update_noisyOR_grid()
			rate.sleep()


	def load_parameters(self):
		"""
		Loads all the important paramters of the human sim
		"""
		# --- simulation params ---# 
		
		# simulation forward prediction parameters
		self.fwd_tsteps = rospy.get_param("pred/fwd_tsteps")

		# --- real-world params ---# 

		# get the speed of the human (meters/sec)
		self.human_vel = rospy.get_param("pred/human_vel")

		# resolution (m/cell)
		self.res = rospy.get_param("pred/resolution")

		# compute the timestep (seconds/cell)
		self.deltat = self.res/self.human_vel

		# number of humans in your space
		self.num_humans = rospy.get_param("pred/num_humans")

		# stores the occu_grid_time for each human
		self.all_occu_grids = [None]*self.num_humans
		self.noisyOR_occu_grid = None

		# measurements of gridworld
		self.sim_height = int(rospy.get_param("pred/sim_height"))
		self.sim_width = int(rospy.get_param("pred/sim_width"))


		# TODO This is for debugging.
		print "----- Running multi-prediction for: -----"
		print " - num humans: ", self.num_humans
		print "-----------------------------------"

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		self.human_subs = [None]*self.num_humans
		for human_num in range(self.num_humans):
			# subscribe to the info of the human walking around the space
			self.human_subs[human_num] = rospy.Subscriber('/occupancy_grid_time'+str(human_num+1), 
				OccupancyGridTime, self.human_grid_callback, queue_size=1)

		# occupancy grid publisher & small publishers for visualizing the start/goal
		self.occu_pub = rospy.Publisher('/occupancy_grid_time', OccupancyGridTime, queue_size=1)

	def human_grid_callback(self, msg):
		"""
		Takes a human grid callback and stores it 
		"""

		#print "got human grid callback: ", msg.object_num
		#print "timestamp: ", msg.gridarray[0].header.stamp

		# human num takes values 1 --> NUM_HUMAN
		# but if no human_num is provided, make sure to index right
		if msg.object_num == 0:
			self.all_occu_grids[0] = msg.gridarray
		else: 
			self.all_occu_grids[msg.object_num-1] = msg.gridarray

	def update_noisyOR_grid(self):
		"""
		Update final gird with noisyOR of all the human grids
		"""

		# sanity check before we have gotten any messages
		if self.all_occu_grids is None:
			print "[multi_human_prediction]: all_occu_grids are None"
			return

		curr_time = rospy.Time.now()
		all_grids_copy = copy.deepcopy(self.all_occu_grids)
		grid_len = self.sim_height*self.sim_width

		noisyNOR_grid = np.array([[1.0]*grid_len]*self.fwd_tsteps)
		noisyOR_grid = np.array([[0.0]*grid_len]*self.fwd_tsteps)

		s = rospy.Time().now()

		for occu_grid in all_grids_copy:
			if occu_grid is not None:
				grid_time = occu_grid[0].header.stamp

				# shift old data to align
				if grid_time.to_sec() < (curr_time.to_sec() - self.deltat):
					d = int(math.floor((curr_time.to_sec() - grid_time.to_sec())/self.deltat))
					future_times = int(self.fwd_tsteps)
					for k in range(future_times-d-1):
						occu_grid[k].data = occu_grid[k+d].data
					for k in range(future_times-d, future_times-1):
						occu_grid[k].data = [0.0]*grid_len

				occu_grid_data = np.array([[0.0]*grid_len]*self.fwd_tsteps)
				for k in range(self.fwd_tsteps):
					occu_grid_data[k] = occu_grid[k].data

				# accumulate the noisy-NOR
				noisyNOR_grid = noisyNOR_grid*(1 - np.array(occu_grid_data))
			else:
				print "[multi_human_prediction]: occu_grid is None"

		# compute noisy-OR
		noisyOR_grid = 1 - noisyNOR_grid

		e = rospy.Time().now()
		#print "TOTAL TIME TO UPDATE GRID: ", (e.to_sec()-s.to_sec())

		# convert to ROS message and publish over topic
		self.noisyOR_occu_grid = self.noisyOR_to_message(noisyOR_grid, curr_time)
		self.occu_pub.publish(self.noisyOR_occu_grid)


	def noisyOR_to_message(self, noisyOR_grid, curr_time):
		"""
		Converts noisyOR grid into OccupancyGridTime structure to ROS msg
		"""
		timed_grid = OccupancyGridTime()
		timed_grid.gridarray = [None]*self.fwd_tsteps
		timed_grid.object_num = 0

		for t in range(self.fwd_tsteps):
			grid_msg = ProbabilityGrid()

			# Set up the header.
			grid_msg.header.stamp = curr_time + rospy.Duration(t*self.deltat)
			grid_msg.header.frame_id = "/world"

			# .info is a nav_msgs/MapMetaData message. 
			grid_msg.resolution = self.res
			grid_msg.width = self.sim_width
			grid_msg.height = self.sim_height

			# Rotated maps are not supported... 
			origin_x=0.0 
			origin_y=0.0 
			grid_msg.origin = Pose(Point(origin_x, origin_y, 0), Quaternion(0, 0, 0, 1))

			# convert to list of doubles from 0-1
			grid_msg.data = list(noisyOR_grid[t])
			timed_grid.gridarray[t] = grid_msg
 
		return timed_grid

 
if __name__ == '__main__':

	multi_human = MultiHumanPrediction()
