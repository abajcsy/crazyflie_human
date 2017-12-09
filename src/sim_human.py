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

		rate = rospy.Rate(100) # 40hz

		while not rospy.is_shutdown():
			t = rospy.Time.now().secs - self.start_T
			self.update_pose(t)
			self.state_pub.publish(self.human_pose)
			self.marker_pub.publish(self.pose_to_marker())
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
		self.human_height = rospy.get_param("pred/human_height")
		self.start = rospy.get_param("pred/start")
		self.goals = rospy.get_param("pred/goals")
		# TODO this only works for one goal right now!
		self.goal = self.goals[0]
		self.start_T = rospy.Time.now().secs
		self.final_T = 20.0
		self.human_pose = None

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
		marker.pose.position.z = 0.2
		marker.scale.x = self.res
		marker.scale.y = self.res
		marker.scale.z = self.res*self.human_height
		marker.color.a = 1.0
		marker.color.r = 1.0

		if self.human_pose is not None:
			marker.pose.position.x = self.human_pose.pose.position.x 
			marker.pose.position.y = self.human_pose.pose.position.y
			marker.pose.position.z = self.res*2
		else:
			marker.pose.position.x = 0
			marker.pose.position.y = 0
			marker.pose.position.z = 0

		return marker

	def update_pose(self, curr_time):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		if curr_time >= self.final_T:
			target_pos = np.array(self.goal)
		else:
			prev = np.array(self.start)
			next = np.array(self.goal)
			ti = 0
			tf = self.final_T
			target_pos = (next - prev)*((curr_time-ti)/(tf - ti)) + prev		
		
		self.human_pose = PoseStamped()
		self.human_pose.header.frame_id="/frame_id_1"
		self.human_pose.header.stamp = rospy.Time.now()
		# set the current timestamp
		# self.human_pose.header.stamp.secs = curr_time
		self.human_pose.pose.position.x = target_pos[0]/self.res
		self.human_pose.pose.position.y = target_pos[1]/self.res
		self.human_pose.pose.position.z = 0.0

if __name__ == '__main__':

	human = SimHuman()
