#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
import tf_conversions
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker
import time
import sys

class DubinsCar(object):
	"""
	This class simulates mocap data of a dubins car 
	driving around an environment. 
	"""	

	def __init__(self):

		rospy.init_node('dubins_car', anonymous=True)

		# load all the prediction params and setup subscriber/publishers
		self.load_parameters()
		self.register_callbacks()

		rate = rospy.Rate(100)

		while not rospy.is_shutdown():
			t = rospy.Time.now().secs - self.start_T
			self.update_pose(t)
			self.state_pub.publish(self.car_pose)
			self.marker_pub.publish(self.pose_to_marker(color=self.color))
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

		# (real-world) start and goal locations 
		self.real_start = rospy.get_param("pred/car_real_start")
		self.real_goals = rospy.get_param("pred/car_real_goals")

		self.car_height = rospy.get_param("pred/car_height")

		# color to use to represent this human
		self.color = rospy.get_param("pred/car_color")

		# compute the timestep (seconds/cell)
		self.car_vel = rospy.get_param("pred/car_vel")
		self.res = rospy.get_param("pred/resolution")
		self.deltat = self.res/self.car_vel

		# compute timetamps for waypts
		self.start_T = rospy.Time.now().secs
		self.final_T = 60.0
		self.step_time = self.final_T/(len(self.real_goals)) 
		self.waypt_times = [i*self.step_time for i in range(len(self.real_goals)+1)] # include start
		self.car_pose = None

		# --- simulation params ---# 

		# resolution (m/cell)
		self.res = rospy.get_param("pred/resolution")

		self.curr_state = self.real_start

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		self.state_pub = rospy.Publisher('/car_pose', PoseStamped, queue_size=10)
		self.marker_pub = rospy.Publisher('/car_marker', Marker, queue_size=10)

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
		marker.scale.z = self.res*self.car_height
		marker.color.a = 1.0		
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]

		if self.car_pose is not None:
			marker.pose.position.x = self.car_pose.pose.position.x 
			marker.pose.position.y = self.car_pose.pose.position.y
			marker.pose.position.z = marker.scale.z/2.0
			marker.pose.orientation.x = self.car_pose.pose.orientation.x
			marker.pose.orientation.y = self.car_pose.pose.orientation.y
			marker.pose.orientation.z = self.car_pose.pose.orientation.z
			marker.pose.orientation.w = self.car_pose.pose.orientation.w
		else:
			marker.pose.position.x = 0
			marker.pose.position.y = 0
			marker.pose.position.z = marker.scale.z/2.0

		return marker

	def update_pose(self, curr_time):
		"""
		Gets the next desired position along trajectory
		by interpolating between waypoints given the current t.
		"""

		# get the next target position along the trajectory
		target_pos = self.interpolate(curr_time)

		# get the next state based on dynamics and target position
		print "before dyn: ", self.curr_state
		self.curr_state = self.dynamics(self.curr_state, target_pos)
		print "after dyn: ", self.curr_state

		# create message to carry next state
		self.car_pose = PoseStamped()
		self.car_pose.header.frame_id="/frame_id_1"
		self.car_pose.header.stamp = rospy.Time.now()
		self.car_pose.pose.position.x = self.curr_state[0]
		self.car_pose.pose.position.y = self.curr_state[1]
		self.car_pose.pose.position.z = 0.0
		self.car_pose.pose.orientation = Quaternion(*tf_conversions.transformations.quaternion_from_euler(0,0,self.curr_state[2]))

	def interpolate(self, curr_time):
		# robot moves from the start to all the intermediate goals, then waits
		waypts = [self.real_start] + self.real_goals
		if curr_time >= self.final_T:
			target_pos = np.array(self.real_start)
		else:
			curr_waypt_idx = int(curr_time/self.step_time)
			prev = np.array(waypts[curr_waypt_idx])
			next = np.array(waypts[curr_waypt_idx+1])		
			ti = self.waypt_times[curr_waypt_idx]
			tf = self.waypt_times[curr_waypt_idx+1]
			target_pos = (next - prev)*((curr_time-ti)/(tf - ti)) + prev		
		return target_pos

	def dynamics(self, curr, ref):
		"""
		Computes PID control to track the reference given the current state 
		measurement. Pushes the control through the dynamics and returns
		the next state.
		"""
		x = curr[0]
		y = curr[1]
		phi = curr[2]
		kP = np.array([1, 1, 1])
		pos_error = np.array(curr) - np.array(ref)
		u = np.dot(kP,pos_error)
		if np.abs(u-0.0) > 1e-08:
			x_next = (self.car_vel/u)*(np.sin(phi + self.deltat/u) - np.sin(phi))
			y_next = (self.car_vel/u)*(-np.cos(phi + self.deltat/u) + np.cos(phi))
			phi_next = u*self.deltat
			return [x_next, y_next, phi_next]
		else:
			return curr

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
	car = DubinsCar()
