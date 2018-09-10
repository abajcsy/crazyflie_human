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

		# potential field parameters
		self.goal_field_spread = rospy.get_param("pred/goal_s")
		self.obstacle_field_spread = rospy.get_param("pred/obstacle_s")
		self.goal_radius = rospy.get_param("pred/goal_r")
		self.obstacle_radius = rospy.get_param("pred/obstacle_r")
		self.alpha = rospy.get_param("pred/alpha_pot_field")
		self.beta = rospy.get_param("pred/beta_pot_field")


		# ======== NOTE ======== #
		# For now, assume that there is only one goal that the human is being 
		# "attracted" to with the potential field.
		# ======== NOTE ======== #

		# color to use to represent this human
		self.color = rospy.get_param("pred/human"+self.human_number+"_color")

		# ======== EDIT BEGIN ======== #
		# Store information you need about generating potential field motions.
		# E.g. you can store the start time
		self.start_T = rospy.Time.now().secs
		# ======== EDIT END ======== #

		# PoseStamped ROS message
		self.human_pose = None

		# --- simulation params ---# 

		# resolution (m/cell)
		self.res = rospy.get_param("pred/resolution")

		# store the human's height (visualization) and the previous pose
		self.human_height = rospy.get_param("pred/human_height")
		self.prev_pose = self.real_start

		# get the total number of humans in the environment
		self.total_number_of_humans = rospy.get_param("pred/total_number_of_humans")

		# ======== EDIT BEGIN ======== #

		# You will probably need some variables to 
		# store state information about the other agents
		# in the environment.(E.g. if this class is 
		# simulating human1, it needs to know about 
		# human2, robot2, robot3, ...)
		#
		# Dictionary mapping from human pose topic to current pose
		self.other_human_poses = {}

		# ======== EDIT END ======== #

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		self.state_pub = rospy.Publisher('/human_pose'+self.human_number, PoseStamped, queue_size=10)

		# Create a subscriber for each other human in the environment.
		for ii in range(1, self.total_number_of_humans+1):
			if ii is not int(self.human_number):
				topic = "/human_pose"+str(ii)

				# Lambda function allows us to call a callback
				# with more arguments than normally intended. 
				def curried_callback(t):
					return lambda m: self.human_pose_callback(t, m)

				rospy.Subscriber(topic, PoseStamped, curried_callback(topic), queue_size=1)

		# ======== NOTE ======== #
		# For now, just focus on humans, but eventually 
		# we will need the same kind of code block as above
		# to subscribe to the state information about the robots too. 
		# ======== NOTE ======== #

	def human_pose_callback(self, topic, msg):
		"""
		This callback stores the most recent human pose for
		any given human. 
		"""

		# Store the PoseStamped message of the other human in the dictionary.
		self.other_human_poses[topic] = msg

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
		field model. Sets self.human_pose to be a PoseStamped 
		with the next position.
		"""

		# ======== EDIT BEGIN ======== #
		x_grad = 0
		y_grad = 0
		x_goal_grad = 0
		y_goal_grad = 0
		x_obs_grad = 0
		y_obs_grad = 0

		for goal in self.real_goals:
			dist_to_goal = np.sqrt((goal[0] - self.prev_pose[0])**2 + (goal[1] - self.prev_pose[1])**2)
			theta = np.arctan2(goal[1] - self.prev_pose[1], goal[0] - self.prev_pose[0])
			if dist_to_goal < self.goal_radius:
				x_goal_grad += 0
				y_goal_grad += 0
			elif dist_to_goal >= self.goal_radius and dist_to_goal <= self.goal_radius + self.goal_field_spread:
				x_goal_grad += self.alpha * (dist_to_goal - self.goal_radius) * np.cos(theta)
				y_goal_grad += self.alpha * (dist_to_goal - self.goal_radius) * np.sin(theta)
			else:
				#print "theta: ", theta
				#print "cos(theta): ", np.cos(theta)
				#print "x grad update: ", self.alpha * self.goal_field_spread * np.cos(theta)
				x_goal_grad += self.alpha * self.goal_field_spread * np.cos(theta)
				y_goal_grad += self.alpha * self.goal_field_spread * np.sin(theta)
				#print "x_goal_grad: ", x_goal_grad

		#print "other humans: ", list(self.other_human_poses.keys())
		for p in list(self.other_human_poses.values()):
			obs_x = p.pose.position.x
			obs_y = p.pose.position.y 
			dist_to_obs = np.sqrt((obs_x - self.prev_pose[0])**2 + (obs_y - self.prev_pose[1])**2)
			theta = np.arctan2(obs_y - self.prev_pose[1], obs_x - self.prev_pose[0])
			if dist_to_obs < self.obstacle_radius:
				x_obs_grad += -100000 * np.sign(np.cos(theta))
				y_obs_grad += -100000 * np.sign(np.sin(theta))
			elif dist_to_obs >= self.obstacle_radius and \
				dist_to_obs <= self.obstacle_radius + self.obstacle_field_spread:
				x_obs_grad += -self.beta * (self.obstacle_field_spread + self.obstacle_radius - dist_to_obs) * \
					np.cos(theta)
				y_obs_grad += -self.beta * (self.obstacle_field_spread + self.obstacle_radius - dist_to_obs) * \
					np.sin(theta)
			else:
				x_obs_grad += 0
				y_obs_grad += 0


		x_grad = x_goal_grad + x_obs_grad
		y_grad = y_goal_grad + y_obs_grad
		#print "x_obs_grad: ", x_obs_grad
		#print "x_grad: ", x_grad
		self.prev_pose = [self.prev_pose[0] + 0.01 * x_grad, self.prev_pose[1] + 0.01 * y_grad]

		self.human_pose = PoseStamped()
		self.human_pose.header.frame_id="/frame_id_1"
		self.human_pose.header.stamp = rospy.Time.now()
		self.human_pose.pose.position.x = self.prev_pose[0] 
		self.human_pose.pose.position.y = self.prev_pose[1]
		self.human_pose.pose.position.z = 0.0

		#print self.prev_pose

		# ======== EDIT END ======== #

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
