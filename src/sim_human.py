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

	def __init__(self, height, width):
		self.height = height		
		self.width = width
    self.resolution = 10.0

		self.start = np.array([0,0])
		self.goal = np.array([height-1,width-1])

		self.final_T = 12.0

		self.human_pose = None

		try:
			state_pub = rospy.Publisher('human_pose', PoseStamped, queue_size=10)
			marker_pub = rospy.Publisher('human_marker', Marker, queue_size=10)

			rospy.init_node('sim_human', anonymous=True)

			rate = rospy.Rate(100) # 40hz

			self.start_T = time.time()
			while not rospy.is_shutdown():
				t = time.time() - self.start_T
				self.update_pose(t)
				state_pub.publish(self.human_pose)
				marker_pub.publish(self.pose_to_marker())
				rate.sleep()

		except rospy.ROSInterruptException:
			pass

	def pose_to_marker(self):
		"""
		Converts pose to marker type to vizualize human
		"""
		marker = Marker()
		marker.header.frame_id = "/root"

		marker.type = marker.CUBE
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0.2
		marker.scale.x = 0.1
		marker.scale.y = 0.1
		marker.scale.z = 0.4
		marker.color.a = 1.0
		marker.color.r = 1.0

		if self.human_pose is not None:
			marker.pose.position.x = self.human_pose.pose.position.x 
			marker.pose.position.y = self.human_pose.pose.position.y
		else:
			marker.pose.position.x = 0
			marker.pose.position.y = 0

		return marker

	def update_pose(self, curr_time):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""
		if curr_time >= self.final_T:
			target_pos = self.goal
		else:
			prev = self.start
			next = self.goal
			ti = 0
			tf = self.final_T
			target_pos = (next - prev)*((curr_time-ti)/(tf - ti)) + prev		
		
		self.human_pose = PoseStamped()
		self.human_pose.header.frame_id="/frame_id_1"
		self.human_pose.header.stamp = rospy.Time.now()
		# set the current timestamp
		# self.human_pose.header.stamp.secs = curr_time
		self.human_pose.pose.position.x = target_pos[0]/self.resolution
		self.human_pose.pose.position.y = target_pos[1]/self.resolution
		self.human_pose.pose.position.z = 0.0

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print "ERROR: Not enough arguments. Specify height, width of world"
	else:	
		height = int(sys.argv[1])
		width = int(sys.argv[2])

		human = SimHuman(height, width)
