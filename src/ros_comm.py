#!/usr/bin/env python
import rospy
import sys, select, os
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Pose2D
from visualization_msgs.msg import Marker, MarkerArray
from crazyflie_human.msg import OccupancyGridTime, ProbabilityGrid
import os
import numpy as np
import time

from human_pred import HumanPredMap
from sim_human import SimHuman


class SimWorld(object):

	def __init__(self):

		# create ROS node
		rospy.init_node('human_prediction', anonymous=True)

		# load all the prediction params and setup subscriber/publishers
		self.load_parameters()
		self.register_callbacks()

		rate = rospy.Rate(100) # 100hz

		while not rospy.is_shutdown():
			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				break

			# plot start/goal location TODO this is only for 1 goal right now
			self.start_pub.publish(self.state_to_marker(xy=self.start, color="G"))
			self.goal_pub.publish(self.state_to_marker(xy=self.goals[0], color="R"))

			rate.sleep()

	def load_parameters(self):
		"""
		Loads all the important paramters of the human sim
		"""
		low = rospy.get_param("state/lower")
		up = rospy.get_param("state/upper")
		self.height = up[1] - low[1] 
		self.width = up[0] - low[0]
		self.res = rospy.get_param("pred/resolution")
		self.fwd_tsteps = rospy.get_param("pred/fwd_tsteps")
		self.fwd_deltat = rospy.get_param("pred/fwd_deltat")
		self.start = rospy.get_param("pred/start")
		self.goals = rospy.get_param("pred/goals")
		self.beta = rospy.get_param("pred/init_beta")

		self.human_height = rospy.get_param("pred/human_height")
		self.prob_thresh = rospy.get_param("pred/prob_thresh")

		# create human prediction map
		self.human_map = HumanPredMap(self.height, self.width, self.res, self.fwd_tsteps, self.fwd_deltat, self.goals, self.beta)

		# set start time to None until get first human state message
		self.start_t = None
		

	#TODO THESE TOPICS SHOULD BE FROM THE YAML/LAUNCH FILE
	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		# subscribe to the info of the human walking around the space
		self.human_subscriber = rospy.Subscriber('/human_pose', PoseStamped, self.human_state_callback, queue_size=1)

		# occupancy grid publisher & small publishers for visualizing the start/goal
		self.occupancy_pub = rospy.Publisher('/occupancy_grid_time', OccupancyGridTime, queue_size=1)
		self.goal_pub = rospy.Publisher('/goal_marker', Marker, queue_size=10)
		self.start_pub = rospy.Publisher('/start_marker', Marker, queue_size=10)
		self.grid_vis_pub = rospy.Publisher('/grid_vis_marker', Marker, queue_size=10)
		
	def grid_to_message(self):
		"""
		Converts OccupancyGridTime structure to ROS msg
		"""
		timed_grid = OccupancyGridTime()
		timed_grid.gridarray = [None]*self.fwd_tsteps
	
		curr_time = rospy.Time.now()

		# set the start time of the experiment to the time of first occugrid msg
		if self.start_t is None:
			self.start_t = curr_time.secs

		for t in range(self.fwd_tsteps):
			grid_msg = ProbabilityGrid()

		  # Set up the header.
			grid_msg.header.stamp = curr_time + rospy.Duration(t*self.fwd_deltat)
			grid_msg.header.frame_id = "map"

			# .info is a nav_msgs/MapMetaData message. 
			grid_msg.resolution = self.res
			grid_msg.width = self.width
			grid_msg.height = self.height

			# Rotated maps are not supported... 
			origin_x=0.0 
			origin_y=0.0 
			grid_msg.origin = Pose(Point(origin_x, origin_y, 0), Quaternion(0, 0, 0, 1))

			# convert to list of doubles from 0-1
			grid_msg.data = list(self.human_map.occupancy_grids[t])

			timed_grid.gridarray[t] = grid_msg
 
		return timed_grid

	def human_state_callback(self, msg):
		"""
		Grabs the human's state from the mocap publisher
		"""

		# update the map with where the human is at the current time
		self.human_map.update_human_traj([msg.pose.position.x, msg.pose.position.y])
		
		# infer the new human occupancy map from the current state
		self.human_map.infer_occupancies() 

		# publish occupancy grid list
		self.occupancy_pub.publish(self.grid_to_message())

		# TODO THIS IS DEBUG
		self.visualize_occugrid(1)
		self.visualize_occugrid(2)
		self.visualize_occugrid(3)
		self.visualize_occugrid(4)

	def state_to_marker(self, xy=[0,0], color="R"):
		"""
		Converts xy position to marker type to vizualize human
		"""
		marker = Marker()
		marker.header.frame_id = "/world"

		marker.type = marker.SPHERE
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0
		marker.scale.x = self.res
		marker.scale.y = self.res
		marker.scale.z = self.res
		marker.color.a = 1.0
		if color is "R":
			marker.color.r = 1.0
		elif color is "G":
			marker.color.g = 1.0
		else:
			marker.color.b = 1.0

		marker.pose.position.x = xy[0]
		marker.pose.position.y = xy[1]

		return marker

	def visualize_occugrid(self, time):
		"""
		Visualizes occupancy grid at time
		"""
		marker = Marker()
		marker.header.frame_id = "/world"
		marker.header.stamp = rospy.Time.now()
		marker.id = 0

		marker.type = marker.CUBE
		marker.action = marker.ADD

		marker.scale.x = self.width
		marker.scale.y = self.height
		marker.scale.z = self.width/2.0
		marker.color.a = 0.3
		marker.color.r = 0.3
		marker.color.g = 0.7
		marker.color.b = 0.7

		marker.pose.orientation.w = 1
		marker.pose.position.z = 0
		marker.pose.position.x = -10.0+0.5*marker.scale.x 
		marker.pose.position.y = -10.0+0.5*marker.scale.y
		marker.pose.position.z = 0.0+0.5*marker.scale.z

		self.grid_vis_pub.publish(marker)

		if self.human_map.occupancy_grids is not None:
			grid = self.human_map.interpolate_grid(time)
			
			for i in range(len(grid)):
				# only visualize if greater than prob thresh
				if grid[i] > self.prob_thresh:
					row = i/self.width
					col = i%self.height

					marker = Marker()
					marker.header.frame_id = "/world"
					marker.header.stamp = rospy.Time.now()
					marker.id = i+1

					marker.type = marker.CUBE
					marker.action = marker.ADD

					marker.scale.x = self.res
					marker.scale.y = self.res
					marker.scale.z = self.res*self.human_height
					marker.color.a = 0.4
					marker.color.r = 1
					marker.color.g = 1 - grid[i]
					marker.color.b = 0

					marker.pose.orientation.w = 1
					marker.pose.position.z = 0
					marker.pose.position.x = (row-self.width/2.0)-self.res/2
					marker.pose.position.y = (col-self.height/2.0)-self.res/2
					marker.pose.position.z = self.res*2

					self.grid_vis_pub.publish(marker)

if __name__ == '__main__':

	SimWorld()


