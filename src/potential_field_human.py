#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
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

		#make a marker array for all the goals
		marker_array = MarkerArray()
		for g in self.real_goals:
			marker = self.state_to_marker(xy=g, color=[0, 0, 1])#self.color)
			marker_array.markers.append(marker)

		# Re-number the marker IDs
		id = 0
		for m in marker_array.markers:
			m.id = id
			id += 1

		rate = rospy.Rate(100)

		while not rospy.is_shutdown():
			# get current time and update the human's pose
			t = rospy.Time.now().secs - self.start_T
			self.update_pose(t)
			self.state_pub.publish(self.human_pose)
			self.marker_pub.publish(self.pose_to_marker(color=self.color))
			self.goal_pub.publish(marker_array)


			# publish markers for goal spread and radius
			#goal_spread_marker = self.radius_to_sphere_marker(self.real_goals[0], self.goal_field_spread+self.goal_radius)
			#self.goal_spread_pub.publish(goal_spread_marker)
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

		# color to use to represent this human
		self.color = rospy.get_param("pred/human"+self.human_number+"_color")

		# PoseStamped ROS message
		self.human_pose = None

		# --- simulation params ---# 

		# potential field parameters
		self.start_T = rospy.Time.now().secs
		self.dt = rospy.get_param("sim/dt")
		self.goal_field_spread = rospy.get_param("sim/goal_s")
		self.obstacle_field_spread = rospy.get_param("sim/obstacle_s")
		self.goal_radius = rospy.get_param("sim/goal_r")
		self.obstacle_radius = rospy.get_param("sim/obstacle_r")
		self.alpha = rospy.get_param("sim/alpha_pot_field")
		self.beta = rospy.get_param("sim/beta_pot_field")

		# resolution (m/cell)
		self.res_x = rospy.get_param("pred/resolution_x")
		self.res_y = rospy.get_param("pred/resolution_y")

		# store the human's height (visualization) and the previous pose
		self.human_height = rospy.get_param("pred/human_height")
		self.prev_pose = self.real_start

		# get the total number of humans in the environment
		self.total_number_of_humans = rospy.get_param("pred/total_number_of_humans")

		# Dictionary mapping from human pose topic to current pose
		self.other_human_poses = {}
		# Dictionary mapping from robot pose topic to current pose
		self.other_robot_poses = {}

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		self.goal_pub = rospy.Publisher('/goal_markers'+self.human_number, MarkerArray, queue_size=10)
		self.state_pub = rospy.Publisher('/human_pose'+self.human_number, PoseStamped, queue_size=10)
		self.marker_pub = rospy.Publisher('/human_marker'+self.human_number, Marker, queue_size=10)

		# Create a subscriber for each other human in the environment.
		for ii in range(1, self.total_number_of_humans+1):
			if ii is not int(self.human_number):
				topic = "/human_pose"+str(ii)

				# Lambda function allows us to call a callback
				# with more arguments than normally intended. 
				def curried_callback(t):
					return lambda m: self.human_pose_callback(t, m)

				rospy.Subscriber(topic, PoseStamped, curried_callback(topic), queue_size=1)

		# Visualize the spread and the radius of the goal.
		# self.goal_spread_pub = rospy.Publisher('/goal_spread'+self.human_number, 
		#	Marker, queue_size=10)

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

	def state_to_marker(self, xy=[0,0], color=[1.0,0.0,0.0]):
		"""
		Converts xy position to marker type to vizualize human
		"""
		marker = Marker()
		marker.header.frame_id = "/world"
		marker.header.stamp = rospy.Time().now()

		marker.type = marker.SPHERE
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0
		marker.scale.x = self.res_x
		marker.scale.y = self.res_y
		marker.scale.z = self.res_x
		marker.color.a = 1.0
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]

		marker.pose.position.x = xy[0]
		marker.pose.position.y = xy[1]

		return marker


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
		marker.scale.x = self.res_x
		marker.scale.y = self.res_y
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

	def radius_to_sphere_marker(self, xy, radius, color=[1.0, 0.0, 0.0]):
		"""
		Converts radius to SPHERE marker type placed at xy position.
		"""
		marker = Marker()
		marker.header.frame_id = "/world"

		marker.type = marker.SPHERE
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0.1
		marker.scale.x = radius
		marker.scale.y = radius
		marker.scale.z = radius
		marker.color.a = 0.3		
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]

		marker.pose.position.x = xy[0]
		marker.pose.position.y = xy[1]
		marker.pose.position.z = 0.0

		return marker

	def update_pose(self, curr_time):
		"""
		Gets the next position of the human that is moving to
		a goal, and reacting to obstacles following a potential 
		field model. Sets self.human_pose to be a PoseStamped 
		with the next position.
		"""

		# Store the goal and obstacle gradients in x, y
		x_goal_grad = 0.0
		y_goal_grad = 0.0
		x_obs_grad = 0.0
		y_obs_grad = 0.0

		# Compute goal-gradient based on the distance from human to goal.
		for goal in self.real_goals:
			dist_to_goal = np.linalg.norm(np.array(goal) - np.array(self.prev_pose)) 
			theta = np.arctan2(goal[1] - self.prev_pose[1], goal[0] - self.prev_pose[0])
			if dist_to_goal < self.goal_radius:
				# Stop if inside of goal.
				x_goal_grad += 0.0
				y_goal_grad += 0.0
			elif dist_to_goal >= self.goal_radius and \
				dist_to_goal <= self.goal_radius + self.goal_field_spread:
				# Within the spread of the potential field but outside the 
				# goal's radius, the gradient is proportional to the distance 
				# between the agent and the goal.
				x_goal_grad += self.alpha * (dist_to_goal - self.goal_radius) * np.cos(theta)
				y_goal_grad += self.alpha * (dist_to_goal - self.goal_radius) * np.sin(theta)
			else:
				# Outside the potential field spread, gradient is maximal.
				x_goal_grad += self.alpha * self.goal_field_spread * np.cos(theta)
				y_goal_grad += self.alpha * self.goal_field_spread * np.sin(theta)

		# Compute obstacle-gradient based on the distance from human to other humans.
		for curr_pose in list(self.other_human_poses.values()):
			obs_xy = [curr_pose.pose.position.x, curr_pose.pose.position.y]
			dist_to_obs = np.linalg.norm(np.array(obs_xy) - np.array(self.prev_pose)) 
			theta = np.arctan2(obs_xy[1] - self.prev_pose[1], obs_xy[0] - self.prev_pose[0])
			if dist_to_obs < self.obstacle_radius:
				# Within the obstacle, the potential field is "infinitely" repulsive.
				x_obs_grad += -100000 * np.sign(np.cos(theta))
				y_obs_grad += -100000 * np.sign(np.sin(theta))
			elif dist_to_obs >= self.obstacle_radius and \
				dist_to_obs <= self.obstacle_radius + self.obstacle_field_spread:
				# Within the spread of the potential field but outside the
				# radius of the obstacle, the gradient grows from zero (when agent
				# is at the edge of the spread) to Beta (when agent is at the 
				# edge of the obstacle).
				x_obs_grad += -self.beta * (self.obstacle_field_spread + self.obstacle_radius - dist_to_obs) * \
					np.cos(theta)
				y_obs_grad += -self.beta * (self.obstacle_field_spread + self.obstacle_radius - dist_to_obs) * \
					np.sin(theta)
			else:
				# Outside the obstacle spread, it has no effect.
				x_obs_grad += 0.0
				y_obs_grad += 0.0

		# Combine the potential fields.
		x_grad = x_goal_grad + x_obs_grad
		y_grad = y_goal_grad + y_obs_grad

		# Compute the next location for agent given the gradient.
		self.prev_pose = [self.prev_pose[0] + self.dt * x_grad, self.prev_pose[1] + self.dt * y_grad]

		# Construct a new Pose message with the updated state.
		self.human_pose = PoseStamped()
		self.human_pose.header.frame_id="/frame_id_1"
		self.human_pose.header.stamp = rospy.Time.now()
		self.human_pose.pose.position.x = self.prev_pose[0] 
		self.human_pose.pose.position.y = self.prev_pose[1]
		self.human_pose.pose.position.z = 0.0

	def sim_to_real_coord(self, sim_coord):
		"""
		Takes [x,y] coordinate in simulation frame and returns a rotated and 
		shifted	value in the ROS coordinates
		"""
		return [sim_coord[0]*self.res_x + self.real_lower[0], 
						self.real_upper[1] - sim_coord[1]*self.res_y]

	def real_to_sim_coord(self, real_coord):
		"""
		Takes [x,y] coordinate in the ROS real frame, and returns a rotated and 
		shifted	value in the simulation frame
		"""
		return [int(round((real_coord[0] - self.real_lower[0])/self.res_x)),
						int(round((self.real_upper[1] - real_coord[1])/self.res_y))]

if __name__ == '__main__':
	human = PotentialFieldHuman()
