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

		self.prev_t = rospy.Time.now()

		while not rospy.is_shutdown():
			t_since_start = rospy.Time.now().secs - self.start_T
			curr_t = rospy.Time.now()
			time_diff = (curr_t - self.prev_t).to_sec()

			# induce fake time dilation to make predictions catch up!
			TIME_DILATION = 1.0
			time_diff = time_diff/TIME_DILATION

			# only use measurements of the car every deltat timesteps
			if time_diff >= self.deltat:
				self.update_pose(t_since_start/TIME_DILATION)
				self.prev_t += rospy.Duration.from_sec(self.deltat) 
			
			if self.car_pose is not None:
				self.state_pub.publish(self.car_pose)
				self.marker_pub.publish(self.pose_to_marker(color=[0.0,0.0,1.0]))

			rate.sleep()

	def load_parameters(self):
		"""
		Loads all the important paramters of the human sim
		"""

		# --- real-world params ---# 

		# store which car occu grid we are computing
		self.car_number = str(rospy.get_param("car_number"))

		# are we simulating the accurate example, unmodeled goal, or 
		# unmodeled obstacle?
		self.example = rospy.get_param("example")

		low = rospy.get_param("state/lower")
		up = rospy.get_param("state/upper")

		# get real-world measurements of experimental space
		self.real_height = up[1] - low[1] 
		self.real_width = up[0] - low[0] 
		self.real_lower = low
		self.real_upper = up

		# (real-world) start and goal locations 
		self.real_start = rospy.get_param("pred/car"+self.car_number+"_real_start_"+self.example)
		self.real_goals = rospy.get_param("pred/car"+self.car_number+"_real_goals_"+self.example)
		if self.example == "goal":
			# if the example is to simulate going to an unmodeled goal, 
			# set the unmodeled goal here (so prediction is unaware)
			self.real_goals = [[1.0, 11.5, 3.1415927]]
		self.car_height = rospy.get_param("pred/car_height")

		# color to use to represent this car
		self.color = rospy.get_param("pred/car"+self.car_number+"_color")

		# compute the timestep (seconds/cell)
		self.car_vel = rospy.get_param("pred/car_vel")

		# compute timetamps for waypts
		self.start_T = rospy.Time.now().secs

		# store PoseStamped message for publishing
		self.car_pose = None

		# --- simulation params ---# 
		# grid world height
		self.sim_height = int(rospy.get_param("pred/sim_height"))

		# resolution (m/cell) -- NOTE: ASSUMES SQUARE GRID
		self.res = self.real_height/self.sim_height 
		self.deltat = self.res/self.car_vel

		self.curr_state = self.real_start

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		self.state_pub = rospy.Publisher("/car_pose"+self.car_number, PoseStamped, queue_size=10)
		self.marker_pub = rospy.Publisher("/car_marker"+self.car_number, Marker, queue_size=10)

	def pose_to_marker(self, color=[1.0, 0.0, 0.0]):
		"""
		Converts pose to marker type to vizualize human
		"""
		marker = Marker()
		marker.header.frame_id = "/world"

		marker.type = marker.ARROW
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0.1
		marker.scale.x = self.res
		marker.scale.y = self.res/2.0
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
		tol = 0.1
		(gx, gy, gtheta) = self.real_goals[0]
		(x,y,theta) = self.curr_state

		# TODO: THIS IS WRONG -- CAR CAN'T STOP INSTANTANEOUSLY. ALSO ONLY WORKS 
		# 		FOR ONE GOAL. 
		
		# if we are close enough to the goal, stay there
		if np.abs(x - gx) < tol and np.abs(y-gy) < tol and np.abs(theta - gtheta) < tol:
			 self.curr_state = self.real_goals[0]
		else:	 
			# get the next target position along the trajectory
			self.curr_state = self.interpolate(curr_time)
			#print "curr state: ", self.curr_state

		# create message to carry next state
		self.car_pose = PoseStamped()
		self.car_pose.header.frame_id="/frame_id_1"
		self.car_pose.header.stamp = rospy.Time.now()
		self.car_pose.pose.position.x = self.curr_state[0]
		self.car_pose.pose.position.y = self.curr_state[1]
		self.car_pose.pose.position.z = 0.0
		self.car_pose.pose.orientation = Quaternion(*tf_conversions.transformations.quaternion_from_euler(0,0,self.curr_state[2]))

	def interpolate(self, curr_time):
		"""
		Get the current position of the car 
		"""

		u = 0.5

		if self.example == "accurate":
			# drive straight towards goal
			return self.dynamics(0.0)
		elif self.example == "obstacle":
			# swerve around a pothole on side of road
			if curr_time >= 0.0 and curr_time < 2.0:
				return self.dynamics(0.0)
			elif curr_time >= 2.0 and curr_time < 4.0:
				# turn left
				return self.dynamics(u)
			elif curr_time >= 4.0 and curr_time < 6.0:
				# turn right
				return self.dynamics(-u)
			elif curr_time >= 6.0 and curr_time < 8.0:
				return self.dynamics(-u)
			elif curr_time >= 8.0 and curr_time < 10.0:
				return self.dynamics(u)
			else:
				return self.dynamics(0.0)
		elif self.example == "goal":
			# take a turn off the road
			if curr_time >= 0.0 and curr_time < 6.0:
				return self.dynamics(0.0)
			elif curr_time >= 6.0 and curr_time < 9.0:
				return self.dynamics(-u)
			else:
				return self.dynamics(0.0)

	def dynamics(self, u):
		"""
		Pushes a control input through the dynamics and returns
		the next state.
		"""
		x = self.curr_state[0]
		y = self.curr_state[1]
		phi = self.curr_state[2]

		# check if control is non-zero
		if np.abs(u-0.0) > 1e-08:
			x_next = x + (self.car_vel/u)*(np.sin(phi + u*self.deltat) - np.sin(phi))
			y_next = y - (self.car_vel/u)*(np.cos(phi + u*self.deltat) - np.cos(phi))
			phi_next = phi + u*self.deltat
			return [x_next, y_next, phi_next]
		else:
			x_next = x + self.car_vel*self.deltat*np.cos(phi)
			y_next = y + self.car_vel*self.deltat*np.sin(phi)
			phi_next = phi + u*self.deltat
			return [x_next, y_next, phi_next]

if __name__ == '__main__':
	car = DubinsCar()
