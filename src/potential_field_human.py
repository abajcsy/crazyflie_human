#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker
import time
import sys

class PotentialFieldHuman(object):
	"""
	This class simulates a human pedestrian with an
	Attractive-Repulsive Potential Field Method.

	References: 
		- Goodrich, M. A. "Potential Fields Tutorial". https://bit.ly/2LXAeIM
	"""

	def __init__(self):

		rospy.init_node('potential_field_human', anonymous=True)

		# load all the prediction params and setup subscriber/publishers
		self.load_parameters()
		self.register_callbacks()

		rate = rospy.Rate(100)

		while not rospy.is_shutdown():
			# get current time and update the human's pose
			t = rospy.Time.now().secs - self.start_T
			self.update_pose(t)
			self.state_pub.publish(self.human_pose)
			rate.sleep()

	def load_parameters(self):
		"""
		Loads all the important paramters of the human sim.
		"""

		# store which human occu grid we are computing
		self.human_number = str(rospy.get_param("human_number"))

		# --- real-world params ---# 

		low = rospy.get_param("state/lower")
		up = rospy.get_param("state/upper")

		# get real-world measurements of experimental space
		self.real_height = up[1] - low[1] 
		self.real_width = up[0] - low[0] 
		self.real_lower = low
		self.real_upper = up

		# (real-world) start and goal locations 
		self.real_start = rospy.get_param("pred/human"+self.human_number+"_real_start")
		self.real_goals = rospy.get_param("pred/human"+self.human_number+"_real_goals")

		# ======== NOTE ======== #
		# For now, assume that there is only one goal that
		# the human is being "attracted" to with the potential
		# field.
		# ======== NOTE ======== #

		# color to use to represent this human
		self.color = rospy.get_param("pred/human"+self.human_number+"_color")

		# ======== EDIT ======== #

		# trajectory info
		self.start_T = rospy.Time.now().secs
		self.final_T = 60.0
		self.step_time = self.final_T/(len(self.real_goals)+1) 
		self.waypt_times = [i*self.step_time for i in range(len(self.real_goals)+2)] # include start and end

		# ======== EDIT ======== #

		self.human_pose = None

		# --- simulation params ---# 

		# resolution (m/cell)
		self.res = rospy.get_param("pred/resolution")

		self.sim_start = self.sim_to_real_coord(self.real_start) 
		self.sim_goals = [self.real_to_sim_coord(g) for g in self.real_goals]
		self.human_height = rospy.get_param("pred/human_height")

		self.prev_pose = self.real_start

		# ======== EDIT ======== #
		# You will probably need some variables to 
		# store state information about the other agents
		# in the environment.(E.g. if this class is 
		# simulating human1, it needs to know about 
		# human2, robot2, robot3, ...)
		# ======== EDIT ======== #

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		self.state_pub = rospy.Publisher('/human_pose'+self.human_number, PoseStamped, queue_size=10)

		# ======== EDIT ======== #
		# You will probably need some subscribers to 
		# the state information about the other agents
		# in the environment. (E.g. if this class is 
		# simulating human1, it needs to know about 
		# human2, robot2, robot3, ...)
		# ======== EDIT ======== #

	def pose_to_marker(self, color=[1.0, 0.0, 0.0]):
		"""
		Converts pose to marker type to vizualize human
		"""
		marker = Marker()
		marker.header.frame_id = "/world"

		marker.type = marker.CUBE
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0.1
		marker.scale.x = self.res
		marker.scale.y = self.res
		marker.scale.z = self.human_height
		marker.color.a = 1.0		
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]


		if self.human_pose is not None:
			marker.pose.position.x = self.human_pose.pose.position.x 
			marker.pose.position.y = self.human_pose.pose.position.y
			marker.pose.position.z = marker.scale.z/2.0
		else:
			marker.pose.position.x = 0
			marker.pose.position.y = 0
			marker.pose.position.z = 2

		return marker


	def update_pose(self, curr_time):
		"""
		Gets the next position of the human that is moving to
		a goal, and reacting to obstacles following a potential 
		field model. 
		"""

		# ======== EDIT ======== #

		raise NotImplementedError

		# ======== EDIT ======== #

	def sim_to_real_coord(self, sim_coord):
		"""
		Takes [x,y] coordinate in simulation frame and returns a rotated and 
		shifted	value in the ROS coordinates
		"""
		return [sim_coord[0]*self.res + self.real_lower[0], 
						self.real_upper[1] - sim_coord[1]*self.res]

	def real_to_sim_coord(self, real_coord):
		"""
		Takes [x,y] coordinate in the ROS real frame, and returns a rotated and 
		shifted	value in the simulation frame
		"""
		return [int(round((real_coord[0] - self.real_lower[0])/self.res)),
						int(round((self.real_upper[1] - real_coord[1])/self.res))]

if __name__ == '__main__':
	human = PotentialFieldHuman()
