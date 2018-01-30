#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker
import time
import sys

class SimHuman(object):	
	"""
	This class simulates mocap data of a human moving around a space.
	Publishes the human's (x,y,theta) data to ROS topic /human_pose
	"""

	def __init__(self):

		rospy.init_node('sim_human', anonymous=True)

		# load all the prediction params and setup subscriber/publishers
		self.load_parameters()
		self.register_callbacks()

		rate = rospy.Rate(100)

		while not rospy.is_shutdown():
			t = rospy.Time.now().secs - self.start_T
			self.update_pose(t)
			self.state_pub.publish(self.human_pose)
			#self.marker_pub.publish(self.pose_to_marker())
			rate.sleep()

	def load_parameters(self):
		"""
		Loads all the important paramters of the human sim
		"""

		# --- real-world params ---# 

		low = rospy.get_param("state/lower")
		up = rospy.get_param("state/upper")

		# get real-world measurements of experimental space
		self.real_height = up[1] - low[1] 
		self.real_width = up[0] - low[0] 
		self.real_lower = low
		self.real_upper = up

		# start and goal locations 
		self.real_start = rospy.get_param("pred/real_start")
		self.real_goal = rospy.get_param("pred/real_goals")[0]

		self.start_T = rospy.Time.now().secs
		self.final_T = 30.0
		self.human_pose = None

		# --- simulation params ---# 

		# resolution (m/cell)
		self.res = rospy.get_param("pred/resolution")

		self.sim_start = self.sim_to_real_coord(self.real_start) #rospy.get_param("pred/sim_start")
		self.sim_goals = [[23,23]] #rospy.get_param("pred/sim_goals")  

		self.human_height = rospy.get_param("pred/human_height")

		self.prev_pose = self.real_start

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		self.state_pub = rospy.Publisher('/human_pose', PoseStamped, queue_size=10)
		self.marker_pub = rospy.Publisher('/human_marker', Marker, queue_size=10)

	def pose_to_marker(self):
		"""
		Converts pose to marker type to vizualize human
		"""
		marker = Marker()
		marker.header.frame_id = "/world"

		marker.type = marker.CUBE
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0.1
		marker.scale.x = 1
		marker.scale.y = 1
		marker.scale.z = self.human_height
		marker.color.a = 1.0
		marker.color.r = 1.0

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
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		if curr_time >= self.final_T:
			target_pos = np.array(self.real_goal)
		else:
			prev = np.array(self.real_start)
			next = np.array(self.real_goal)		
			ti = 0
			tf = self.final_T
			target_pos = (next - prev)*((curr_time-ti)/(tf - ti)) + prev	
			# add in gaussian noise
			noisy_pos = self.prev_pose + np.random.normal(0,self.res, (2,))
			self.prev_pose = target_pos #(target_pos + noisy_pos)*0.5	

		self.human_pose = PoseStamped()
		self.human_pose.header.frame_id="/frame_id_1"
		self.human_pose.header.stamp = rospy.Time.now()
		# set the current timestamp
		# self.human_pose.header.stamp.secs = curr_time
		self.human_pose.pose.position.x = target_pos[0]
		self.human_pose.pose.position.y = target_pos[1]
		self.human_pose.pose.position.z = 0.0

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

	human = SimHuman()
