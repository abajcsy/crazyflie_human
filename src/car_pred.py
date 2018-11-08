#!/usr/bin/env python2.7
from __future__ import division
import rospy
import sys, select, os
import numpy as np
import time
import pickle
import tf
import tf_conversions

from std_msgs.msg import String, Float32, ColorRGBA
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Pose2D, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from crazyflie_human.msg import OccupancyGridTime, ProbabilityGrid

# Get the path of this file, go up two directories, and add that to our 
# Python path so that we can import the pedestrian_prediction module.
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from pp.mdp.car import CarMDP
from pp.inference import hardmax as inf

Actions = CarMDP.Actions

class CarPrediction(object):
	"""
	This class models and predicts dubins-car motion.
	It stores:
		- cars's tracked trajectory 
		- occupancy grid of states car is likely to go to
		- moving obstacle representing car's future motion
	"""

	def __init__(self):

		# create ROS node
		rospy.init_node('car_prediction', anonymous=True)

		# load all the prediction params and setup subscriber/publishers
		self.load_parameters()
		self.register_callbacks()

		# make a marker array for all the goals
		marker_array = MarkerArray()
		for g in self.real_goals:
			marker = self.state_to_marker(xytheta=g, color=self.color)
			marker_array.markers.append(marker)

		# Re-number the marker IDs
		id = 0
		for m in marker_array.markers:
			m.id = id
			id += 1

		# make marker array for all obstacles
		obstacle_marker_array = MarkerArray()
		for box in self.real_obstacles:
			marker = self.box_to_marker(box[0], box[1])
			obstacle_marker_array.markers.append(marker)

		# Re-number the marker IDs
		id = 0
		for m in obstacle_marker_array.markers:
			m.id = id
			id += 1

		rate = rospy.Rate(100) 

		while not rospy.is_shutdown():
			# plot start/goal markers for visualization
			self.goal_pub.publish(marker_array)
			self.obstacle_pub.publish(obstacle_marker_array)

			rate.sleep()

	def load_parameters(self):
		"""
		Loads all the important paramters of the car sim
		"""
		# --- simulation params ---# 
		self.car_number = str(rospy.get_param("car_number"))
		
		# are we simulating the accurate example, unmodeled goal, or 
		# unmodeled obstacle?
		self.example = rospy.get_param("example")

		# measurements of gridworld
		self.sim_height = int(rospy.get_param("pred/sim_height_"+self.example))
		self.sim_width = int(rospy.get_param("pred/sim_width_"+self.example))
		self.sim_theta = int(rospy.get_param("pred/sim_theta_"+self.example))

		# simulation forward prediction parameters
		self.fwd_tsteps = rospy.get_param("pred/fwd_tsteps")

		self.car_height = rospy.get_param("pred/car_height")
		self.prob_thresh = rospy.get_param("pred/prob_thresh")	

		# hidden state volatility in HMM
		self.epsilon_dest = rospy.get_param("pred/epsilon_dest")
		self.epsilon_beta = rospy.get_param("pred/epsilon_beta")

		# stores 2D array of size: 
		# (fwd_tsteps) x (height x width x num_theta) of probabilities
		self.occupancy_grids = None

		# stores list of beta values for each goal
		self.beta_model = rospy.get_param("beta")
		print "beta_model", self.beta_model
		if self.beta_model == "irrational":
			self.betas = rospy.get_param("pred/beta_irrational")
		elif self.beta_model == "rational":
			self.betas = rospy.get_param("pred/beta_rational")
		elif self.beta_model == "adaptive":
			self.betas = rospy.get_param("pred/beta_adaptive")
		else:
			rospy.signal_shutdown("Beta model type is not valid!")
		
		# stores dest x beta array with posterior prob of each beta
		self.dest_beta_prob = None

		# --- real-world params ---# 

		low = rospy.get_param("state/lower_"+self.example)
		up = rospy.get_param("state/upper_"+self.example)

		# get real-world measurements of experimental space
		self.real_width = up[0] - low[0]
		self.real_height = up[1] - low[1]  
		self.real_theta = up[2] - low[2]

		# store the lower and upper measurements
		self.real_lower = low
		self.real_upper = up

		# (real-world) start and goal locations 
		self.real_start = rospy.get_param("pred/car"+self.car_number+"_real_start_"+self.example)
		self.real_goals = rospy.get_param("pred/car"+self.car_number+"_real_goals_"+self.example)

		# get list of obstacles
		self.real_obstacles = rospy.get_param("pred/car"+self.car_number+"_real_obstacles_"+self.example)

		# color to use to represent this car
		self.color = rospy.get_param("pred/car"+self.car_number+"_color")

		# tracks the car's state over time
		self.real_car_traj = None
		self.sim_car_traj = None

		# store the previous time to compute deltat
		self.prev_t = None
		self.prev_pos = None

		# get the speed of the car (meters/sec)
		self.car_vel = rospy.get_param("pred/car_vel")

		# resolution (m/cell) 
		self.res_x = self.real_width/self.sim_width
		self.res_y = self.real_height/self.sim_height

		# compute the timestep (seconds/cell)
		self.deltat_x = self.res_x/self.car_vel
		self.deltat_y = self.res_y/self.car_vel
		self.dt = max(self.deltat_x, self.deltat_y)

		# --- gridworld creation --- #

		# grid world representing the experimental environment
		self.gridworld = CarMDP(self.sim_width, self.sim_height, \
			self.sim_theta, self.real_goals, self.real_lower, dt=self.dt, \
			vel=self.car_vel, res_x=self.res_x, res_y=self.res_y, \
			allow_wait=True, obstacle_list=self.real_obstacles)

		# (simulation) start and goal locations
		self.sim_start = self.gridworld.real_to_coor(self.real_start[0], self.real_start[1], self.real_start[2])
		self.sim_goals = [self.gridworld.real_to_coor(x,y,t) for (x,y,t) in self.real_goals]

		# TODO This is for debugging.
		print "----- Running prediction for one dubins car : -----"
		print "	- car num: ", self.car_number
		print "	- car vel: ", self.car_vel
		print "	- beta model: ", self.beta_model
		print "	- prob thresh: ", self.prob_thresh
		print "	- dt: ", self.dt
		print "	- resolution x: ", self.res_x
		print "	- resolution y: ", self.res_y
		print "---------------------------------------------------"

	#TODO THESE TOPICS SHOULD BE FROM THE YAML/LAUNCH FILE
	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		# subscribe to the info of the car walking around the space
		self.car_sub = rospy.Subscriber('/car_pose'+self.car_number, PoseStamped, 
											self.car_state_callback, queue_size=1)

		# occupancy grid publisher & small publishers for visualizing the goals
		self.occu_pub = rospy.Publisher('/occupancy_grid_time'+self.car_number, 
			OccupancyGridTime, queue_size=1)
		self.beta_pub = rospy.Publisher('/beta_topic'+self.car_number, 
			Float32, queue_size=1)
		self.goal_pub = rospy.Publisher('/goal_markers'+self.car_number, MarkerArray, queue_size=10)
		self.grid_vis_pub = rospy.Publisher('/occu_grid_marker'+self.car_number, 
			Marker, queue_size=10)
		self.obstacle_pub = rospy.Publisher('/obstacle_markers'+self.car_number, MarkerArray, queue_size=10)
		#self.car_marker_pub = rospy.Publisher('/car_marker'+self.car_number, 
		#	Marker, queue_size=10)

	# ---- Inference Functionality ---- #

	def car_state_callback(self, msg):
		"""
		Grabs the car's state from the mocap publisher
		"""

		curr_time = rospy.Time.now()
		quat = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
		euler = tf.transformations.euler_from_quaternion(quat)
		xythetapose = self.make_valid_state([msg.pose.position.x, msg.pose.position.y, euler[2]])

		# if this is the first car state message, just record the time and pose
		if self.prev_t is None:
			self.prev_t = curr_time
			self.prev_pos = xythetapose	
			self.update_car_traj(xythetapose)
			return 
	
		time_diff = (curr_time - self.prev_t).to_sec()

		# only use measurements of the car every deltat timesteps
		if time_diff >= self.dt:

			self.prev_t += rospy.Duration.from_sec(self.dt) 

			# update the map with where the car is at the current time
			self.update_car_traj(xythetapose)

			s = rospy.Time().now()

			# infer the new car occupancy map from the current state
			self.infer_occupancies() 
	
			# update car pose marker
			#self.car_marker_pub.publish(self.pose_to_marker(xythetapose, color=self.color))

			# publish occupancy grid list
			if self.occupancy_grids is not None:
				self.occu_pub.publish(self.grid_to_message())
				self.visualize_occugrid(10)

			self.prev_pos = xythetapose	
			
	def make_valid_state(self, xypose):
		"""
		Takes car state measurement, checks if its inside of the world grid, and 
		creates a valid [x,y,theta] position. 
		If car is in valid [x,y,theta] grid location, returns original xypose
		Else if car is NOT valid, then clips the car's pose to a valid location
		"""

		valid_xypose = [np.clip(xypose[0], self.real_lower[0], self.real_upper[0]),
							np.clip(xypose[1], self.real_lower[1], self.real_upper[1]),
							xypose[2]]		

		return valid_xypose

	def update_car_traj(self, newstate):
		"""
		Given a new sensor measurement of where the car is, update the tracked
		trajectory of the car's movements.
		"""

		sim_newstate = self.gridworld.real_to_coor(newstate[0],newstate[1],newstate[2])

		if self.real_car_traj is None:
			self.real_car_traj = np.array([newstate])
			self.sim_car_traj = np.array([sim_newstate])
		else:
			self.real_car_traj = np.append(self.real_car_traj, 
												np.array([newstate]), 0)

			# if the new measured state does not map to the same state in sim, add it
			# to the simulated trajectory. We need this for the inference to work
			#in_same_state = (sim_newstate == self.sim_car_traj[-1]).all()
			#if not in_same_state:
			self.sim_car_traj = np.append(self.sim_car_traj, 
											np.array([sim_newstate]), 0)

	def infer_occupancies(self):
		"""
		Using the current trajectory data, recompute a new occupancy grid
		for where the car might be.
		"""
		if self.real_car_traj is None or self.sim_car_traj is None:
			print "Can't infer occupancies -- car hasn't appeared yet!"
			return 

		# Convert the car trajectory points from real-world to 3D grid values. 
		#traj = [self.gridworld.real_to_coor(x[0], x[1], x[2]) for x in self.real_car_traj]

  		real_next = self.real_car_traj[-1]
  		real_prev = self.real_car_traj[-2]

  		action = self.gridworld.real_to_action(real_prev, real_next)
  		state = self.gridworld.real_to_state(real_prev[0], real_prev[1], real_prev[2])

  		print "s,a: ", (state, action)

		# The line below feeds in the last car (s,a) pair and previous posterior
		# and does a recursive Bayesian update.
		(self.occupancy_grids, self.beta_occu, self.dest_beta_prob) = inf.state.infer_joint(self.gridworld, 
			self.sim_goals, self.betas, T=self.fwd_tsteps, use_gridless=False, priors=self.dest_beta_prob,
			traj=[(state, action)], epsilon_dest=self.epsilon_dest, epsilon_beta=self.epsilon_beta, verbose_return=True)

		print "----P(dest, beta):----"
		print self.dest_beta_prob
		print "----------------------"

	# ---- Utility Functions ---- #

	def interpolate_grid(self, future_time):
		"""
		Interpolates the grid at some future time
		"""
		if self.occupancy_grids is None:
			print "Occupancy grids are not created yet!"
			return None

		if future_time < 0:
			print "Can't interpolate for negative time!"
			return None

		if future_time > self.fwd_tsteps:
			print "Can't interpolate more than", self.fwd_tsteps, "steps into future!"
			print "future_time =", future_time
			return None

		in_idx = -1
		for i in range(self.fwd_tsteps+1):
			if np.isclose(i, future_time, rtol=1e-05, atol=1e-08):
				in_idx = i
				break
	
		if in_idx != -1:
			# if interpolating exactly at the timestep
			return self.occupancy_grids[in_idx]
		else:
			prev_t = int(future_time)
			next_t = int(future_time)+1

			low_grid = self.occupancy_grids[prev_t]
			high_grid = self.occupancy_grids[next_t]

			interpolated_grid = np.zeros((self.sim_height*self.sim_width*self.sim_theta))

			for i in range(self.sim_height*self.sim_width*self.sim_theta):
				prev = low_grid[i]
				next = high_grid[i]
				curr = prev + (next - prev) *((future_time - prev_t) / (next_t - prev_t))
				interpolated_grid[i] = curr

			return interpolated_grid

	# ---- Visualization ---- #

	def visualize_occugrid(self, time):
		"""
		Visualizes occupancy grid for all grids in time
		"""
		if self.grid_vis_pub.get_num_connections() == 0:
			rospy.loginfo_throttle(1.0, "visualize_occugrid: I'm lonely.")
			return

		if self.occupancy_grids is not None:
			marker = Marker()
			marker.header.frame_id = "/world"
			marker.header.stamp = rospy.Time.now()
			marker.id = 0
			marker.ns = "visualize"

			marker.type = marker.CUBE_LIST
			marker.action = marker.ADD

			marker.scale.x = self.res_x
			marker.scale.y = self.res_y
			marker.scale.z = self.car_height

			for t in range(time):
				grid = self.interpolate_grid(t)

				# TODO: SUM OVER ALL THETAS WHEN VISUALIZING THE PROBABILITY 
				# 		FOR A PARTICULAR (x,y)
				if grid is not None:
					grid_3D = self.convert_occugrid_1Dto3D(grid)
					for x in range(self.sim_width):
						for y in range(self.sim_height):

							# sum over all theta's to get P(x,y)
							probability = np.sum(grid_3D[x,y,:])

							# get the real-world coordinate
							real_coord = self.gridworld.coor_to_real(x,y,0)

							color = ColorRGBA()
							color.a = np.sqrt((1 - (time-1)/self.fwd_tsteps)*probability)
							color.r = np.sqrt(probability)
							color.g = np.minimum(np.absolute(np.log(probability)),5.0)/5.0
							color.b = 0.9*np.minimum(np.absolute(np.log(probability)),5.0)/5.0 + 0.5*np.sqrt(probability)
							if probability < self.prob_thresh:
								color.a = 0.0
							marker.colors.append(color)

							pt = Vector3()
							pt.x = real_coord[0]
							pt.y = real_coord[1]
							pt.z = self.car_height/2.0
							marker.points.append(pt)
					
			self.grid_vis_pub.publish(marker)
			
	def convert_occugrid_1Dto3D(self, occugrid):
		"""
		Given flattened 1D occupancy grid at a particular time, return the
		3D occupancy grid.
		"""
		occugrid_3D = np.zeros((self.sim_width, self.sim_height, self.sim_theta))

		for i in range(len(occugrid)):
			(x,y,theta) = self.gridworld.state_to_coor(i)
			occugrid_3D[x][y][theta] = occugrid[i]

		return occugrid_3D

	# ---- ROS Message Conversion ---- #

	# ---------------------------------------------------------------------#
	# TODO: THIS IS WRONG. NEED TO MODIFY THE OCCUPANCY GRID TIME MESSAGE
	# 		STRUCTURE TO ACCEPT ANOTHER DIMENSION.
	# ---------------------------------------------------------------------#
	def grid_to_message(self):
		"""
		Converts OccupancyGridTime structure to ROS msg
		"""
		timed_grid = OccupancyGridTime()
		timed_grid.gridarray = [None]*self.fwd_tsteps
		timed_grid.object_num = int(self.car_number) 

		curr_time = rospy.Time.now()

		for t in range(self.fwd_tsteps):
			grid_msg = ProbabilityGrid()

			# Set up the header.
			grid_msg.header.stamp = curr_time + rospy.Duration(t*self.dt)
			grid_msg.header.frame_id = "/world"

			# .info is a nav_msgs/MapMetaData message. 
			grid_msg.resolution = self.res_x
			grid_msg.width = self.sim_width
			grid_msg.height = self.sim_height

			# Rotated maps are not supported... 
			origin_x=0.0 
			origin_y=0.0 
			grid_msg.origin = Pose(Point(origin_x, origin_y, 0), Quaternion(0, 0, 0, 1))

			# Assert that all occupancies are in [0, 1].
			assert self.occupancy_grids[t].max() <= 1.0 +1e-8 and self.occupancy_grids[t].min() >= 0.0 - 1e-8
			assert abs(self.occupancy_grids[t].sum() - 1.0) < 1e-8

			# convert to list of doubles from 0-1
			grid_msg.data = list(self.occupancy_grids[t])

			timed_grid.gridarray[t] = grid_msg
 
		return timed_grid

	def state_to_marker(self, xytheta=[0,0,0], color=[1.0,0.0,0.0]):
		"""
		Converts xy position to marker type to vizualize car
		"""
		marker = Marker()
		marker.header.frame_id = "/world"
		marker.header.stamp = rospy.Time().now()

		marker.type = marker.ARROW
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0
		marker.scale.x = self.res_x
		marker.scale.y = self.res_y/2.0
		marker.scale.z = self.res_x
		marker.color.a = 1.0
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]

		marker.pose.position.x = xytheta[0]
		marker.pose.position.y = xytheta[1]
		marker.pose.orientation = Quaternion(*tf_conversions.transformations.quaternion_from_euler(0,0,xytheta[2]))

		return marker

	def box_to_marker(self, xylow, xyupp):
		"""
		Converts lowerxy and upperxy definition of box into 3D marker.
		"""
		xlen = xyupp[0] - xylow[0]
		ylen = xyupp[1] - xylow[1] 

		marker = Marker()
		marker.header.frame_id = "/world"
		marker.header.stamp = rospy.Time().now()

		marker.type = marker.CUBE
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0
		marker.scale.x = xlen
		marker.scale.y = ylen
		marker.scale.z = self.res_x
		marker.color.a = 0.5
		marker.color.r = 0.0
		marker.color.g = 0.0
		marker.color.b = 0.0

		marker.pose.position.x = xylow[0] + xlen/2.0
		marker.pose.position.y = xylow[1] + ylen/2.0
		return marker

	def pose_to_marker(self, xypose, color=[1.0,0.0,0.0]):
		"""
		Converts pose to marker type to vizualize car
		"""
		marker = Marker()
		marker.header.frame_id = "/world"

		marker.type = marker.CUBE
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0.1
		marker.scale.x = self.res_x
		marker.scale.y = self.res_y
		marker.scale.z = self.car_height*self.res_y 
		marker.color.a = 1.0
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]

		if xypose is not None:
			marker.pose.position.x = xypose[0] 
			marker.pose.position.y = xypose[1]
			marker.pose.position.z = marker.scale.z/2.0
		else:
			marker.pose.position.x = 0
			marker.pose.position.y = 0
			marker.pose.position.z = 2

		return marker

if __name__ == '__main__':
	car = CarPrediction()
