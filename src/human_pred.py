#!/usr/bin/env python2.7
from __future__ import division
import rospy
import sys, select, os
import numpy as np
import time
import pickle

from std_msgs.msg import String, Float32, ColorRGBA
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Pose2D, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from crazyflie_human.msg import OccupancyGridTime, ProbabilityGrid

# Get the path of this file, go up two directories, and add that to our 
# Python path so that we can import the pedestrian_prediction module.
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from pp.mdp import GridWorldMDP
from pp.mdp.expanded import GridWorldExpanded
from pp.inference import hardmax as inf

Actions = GridWorldMDP.Actions

class HumanPrediction(object):
	"""
	This class models and predicts human motions in a 2D planar environment.
	It stores:
		- human's tracked trajectory 
		- occupancy grid of states human is likely to go to
		- moving obstacle representing human future motion
	"""

	def __init__(self):

		# create ROS node
		rospy.init_node('human_prediction', anonymous=True)

		# load all the prediction params and setup subscriber/publishers
		self.load_parameters()
		self.register_callbacks()

		# make a marker array for all the goals
		marker_array = MarkerArray()
		for g in self.real_goals:
			marker = self.state_to_marker(xy=g, color=self.color)
			marker_array.markers.append(marker)

		# Re-number the marker IDs
		id = 0
		for m in marker_array.markers:
			m.id = id
			id += 1

		rate = rospy.Rate(100) 

		while not rospy.is_shutdown():
			# plot start/goal markers for visualization
			self.goal_pub.publish(marker_array)

			rate.sleep()

	def load_parameters(self):
		"""
		Loads all the important paramters of the human sim
		"""
		# --- simulation params ---# 
		self.human_number = str(rospy.get_param("human_number"))
		
		# measurements of gridworld
		self.sim_height = int(rospy.get_param("pred/sim_height"))
		self.sim_width = int(rospy.get_param("pred/sim_width"))

		# resolution (m/cell)
		self.res = rospy.get_param("pred/resolution")

		# simulation forward prediction parameters
		self.fwd_tsteps = rospy.get_param("pred/fwd_tsteps")

		self.human_height = rospy.get_param("pred/human_height")
		self.prob_thresh = rospy.get_param("pred/prob_thresh")	

		# hidden state volatility in HMM
		self.epsilon_dest = rospy.get_param("pred/epsilon_dest")
		self.epsilon_beta = rospy.get_param("pred/epsilon_beta")

		# stores 2D array of size (fwd_tsteps) x (height x width) of probabilities
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

		# grid world representing the experimental environment
		self.gridworld = GridWorldExpanded(self.sim_height, self.sim_width)

		# --- real-world params ---# 

		low = rospy.get_param("state/lower")
		up = rospy.get_param("state/upper")

		# get real-world measurements of experimental space
		self.real_height = up[1] - low[1] 
		self.real_width = up[0] - low[0] 
		# store the lower and upper measurements
		self.real_lower = low
		self.real_upper = up

		# (real-world) start and goal locations 
		self.real_start = rospy.get_param("pred/human"+self.human_number+"_real_start")
		self.real_goals = rospy.get_param("pred/human"+self.human_number+"_real_goals")

		# (simulation) start and goal locations
		self.sim_start = self.real_to_sim_coord(self.real_start)
		self.sim_goals = [self.real_to_sim_coord(g) for g in self.real_goals]

		# color to use to represent this human
		self.color = rospy.get_param("pred/human"+self.human_number+"_color")

		# tracks the human's state over time
		self.real_human_traj = None
		self.sim_human_traj = None

		# store the previous time to compute deltat
		self.prev_t = None
		self.prev_pos = None

		# get the speed of the human (meters/sec)
		self.human_vel = rospy.get_param("pred/human_vel")

		# compute the timestep (seconds/cell)
		self.deltat = self.res/self.human_vel

		# TODO This is for debugging.
		print "----- Running prediction for one human : -----"
		print "	- human: ", self.human_number
		#print "	- experiment: ", self.exp
		print "	- beta model: ", self.beta_model
		print "	- prob thresh: ", self.prob_thresh
		print "----------------------------------------------"

	#TODO THESE TOPICS SHOULD BE FROM THE YAML/LAUNCH FILE
	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		# subscribe to the info of the human walking around the space
		self.human_sub = rospy.Subscriber('/human_pose'+self.human_number, PoseStamped, 
											self.human_state_callback, queue_size=1)

		# occupancy grid publisher & small publishers for visualizing the goals
		self.occu_pub = rospy.Publisher('/occupancy_grid_time'+self.human_number, 
			OccupancyGridTime, queue_size=1)
		self.beta_pub = rospy.Publisher('/beta_topic'+self.human_number, 
			Float32, queue_size=1)
		self.goal_pub = rospy.Publisher('/goal_markers'+self.human_number, MarkerArray, queue_size=10)
		self.grid_vis_pub = rospy.Publisher('/occu_grid_marker'+self.human_number, 
			Marker, queue_size=10)
		self.human_marker_pub = rospy.Publisher('/human_marker'+self.human_number, 
			Marker, queue_size=10)

	# ---- Inference Functionality ---- #

	def human_state_callback(self, msg):
		"""
		Grabs the human's state from the mocap publisher
		"""
		curr_time = rospy.Time.now()
		xypose = self.make_valid_state([msg.pose.position.x, msg.pose.position.y])

		# if this is the first human state message, just record the time and pose
		if self.prev_t is None:
			self.prev_t = curr_time
			self.prev_pos = xypose	
			self.update_human_traj(xypose)
			return 
	
		time_diff = (curr_time - self.prev_t).to_sec()

		# only use measurements of the human every deltat timesteps
		if time_diff >= self.deltat:

			self.prev_t += rospy.Duration.from_sec(self.deltat) 

			# get the human's current state and make sure its always a valid location
			#xypose = self.make_valid_state([msg.pose.position.x, msg.pose.position.y])

			# update the map with where the human is at the current time
			self.update_human_traj(xypose)

			s = rospy.Time().now()

			# infer the new human occupancy map from the current state
			self.infer_occupancies() 
	
			# update human pose marker
			self.human_marker_pub.publish(self.pose_to_marker(xypose, color=self.color))

			# publish occupancy grid list
			if self.occupancy_grids is not None:
				self.occu_pub.publish(self.grid_to_message())
				self.visualize_occugrid(3)

			# adjust the deltat based on the observed measurements
			if self.prev_pos is not None:
				self.human_vel = np.linalg.norm((np.array(xypose) - np.array(self.prev_pos)))/time_diff
				self.deltat = np.minimum(np.maximum(self.res/self.human_vel,0.05),0.2)

			self.prev_pos = xypose	
			

	def make_valid_state(self, xypose):
		"""
		Takes human state measurement, checks if its inside of the world grid, and 
		creates a valid [x,y] position. 
		If human is in valid [x,y] grid location, returns original xypose
		Else if human is NOT valid, then clips the human's pose to a valid location
		"""

		valid_xypose = [np.clip(xypose[0], self.real_lower[0], self.real_upper[0]),
							np.clip(xypose[1], self.real_lower[1], self.real_upper[1])]		

		return valid_xypose

	def update_human_traj(self, newstate):
		"""
		Given a new sensor measurement of where the human is, update the tracked
		trajectory of the human's movements.
		"""

		sim_newstate = self.real_to_sim_coord(newstate)

		if self.real_human_traj is None:
			self.real_human_traj = np.array([newstate])
			self.sim_human_traj = np.array([sim_newstate])
		else:
			self.real_human_traj = np.append(self.real_human_traj, 
												np.array([newstate]), 0)

			# if the new measured state does not map to the same state in sim, add it
			# to the simulated trajectory. We need this for the inference to work
			#in_same_state = (sim_newstate == self.sim_human_traj[-1]).all()
			#if not in_same_state:
			self.sim_human_traj = np.append(self.sim_human_traj, 
											np.array([sim_newstate]), 0)


	def infer_occupancies(self):
		"""
		Using the current trajectory data, recompute a new occupancy grid
		for where the human might be.
		"""
		if self.real_human_traj is None or self.sim_human_traj is None:
			print "Can't infer occupancies -- human hasn't appeared yet!"
			return 

		# Get list of goals in 2D 
		dest_list = [self.gridworld.coor_to_state(g[0], g[1]) for g in self.sim_goals]

		# Convert the human trajectory points from real-world to 2D grid values. 
		traj = [self.real_to_sim_coord(x, round_vals=False) for x in self.real_human_traj] 
  
  		# OPTION 1: The line below feeds in the entire human traj history so far
  		# 			and does a single bulk Bayesian inference step.
		# (self.occupancy_grids, self.beta_occu, self.dest_beta_prob) = inf.state.infer_joint(self.gridworld, 
		# 	dest_list, self.betas, T=self.fwd_tsteps, use_gridless=True, traj=traj, verbose_return=True)

		# OPTION 2: The line below feeds in the last human (s,a) pair and previous posterior
		# 			and does a recursive Bayesian update.
		(self.occupancy_grids, self.beta_occu, self.dest_beta_prob) = inf.state.infer_joint(self.gridworld, 
			dest_list, self.betas, T=self.fwd_tsteps, use_gridless=True, priors=self.dest_beta_prob,
			traj=traj[-2:], epsilon_dest=self.epsilon_dest, epsilon_beta=self.epsilon_beta, verbose_return=True)

	# ---- Utility Functions ---- #

	def traj_to_state_action(self):
		"""
		Converts the measured state-based sim_human_traj into (state, action) traj.
		"""		
		prev = self.sim_human_traj[0]		
		states = np.array([prev])
		actions = None
		for i in range(1,len(self.sim_human_traj)):
			next = self.sim_human_traj[i]
			# dont consider duplicates of the same measurement
			if not np.array_equal(prev, next):
				states = np.append(states, [next], 0)
				curr_action = self.motion_to_action(prev, next)

				if actions is None:
					actions = np.array([curr_action])
				else:
					actions = np.append(actions, curr_action)			
			prev = next

		grid_states = [self.gridworld.coor_to_state(s[0], s[1]) for s in states]

		sa_traj = []
		if actions is not None:
			for i in range(len(actions)):			
				sa_traj.append((grid_states[i], actions[i]))

		return sa_traj

	def motion_to_action(self, prev_pos, next_pos):
		"""
		Takes two measured positions of the human (previous and next) 
		and returns the gridworld action that the human took.
		"""
		xdiff = next_pos[0] - prev_pos[0]
		ydiff = next_pos[1] - prev_pos[1]
		if xdiff < 0:
			if ydiff < 0:
					action = Actions.DOWN_LEFT
			elif ydiff == 0:	
					action = Actions.LEFT		
			else:
					action = Actions.UP_LEFT
		elif xdiff == 0:
			if ydiff < 0:
					action = Actions.DOWN
			elif ydiff == 0:
					action = Actions.ABSORB		# note this should only happen at goal
			else:
					action = Actions.UP
		else:
			if ydiff < 0:
				action = Actions.DOWN_RIGHT
			elif ydiff == 0:
				action = Actions.RIGHT
			else:
				action = Actions.UP_RIGHT	

		return action

	def state_to_coor(self, state):
		"""
		Goes from 1D array value ot [x,y] in simulation
		"""
		return [state/self.sim_width, state%self.sim_width]

	def sim_to_real_coord(self, sim_coord):
		"""
		Takes [x,y] coordinate in simulation frame and returns a rotated and 
		shifted	value in the ROS coordinates
		"""
		return [sim_coord[0]*self.res + self.real_lower[0], 
				self.real_upper[1] - sim_coord[1]*self.res]

	def real_to_sim_coord(self, real_coord, round_vals=True):
		"""
		Takes [x,y] coordinate in the ROS real frame, and returns a rotated and 
		shifted	value in the simulation frame
		-- round_vals: 
					True - gives an [i,j] integer-valued grid cell entry.
					False - gives a floating point value on the grid cell
		"""
		if round_vals:
			x = round((real_coord[0] - self.real_lower[0])/self.res)
			y = round((self.real_upper[1] - real_coord[1])/self.res)
		else:
			x = (real_coord[0] - self.real_lower[0])/self.res
			y = (self.real_upper[1] - real_coord[1])/self.res

		i_coord = np.minimum(self.sim_height-1, np.maximum(0.0,x));
		j_coord = np.minimum(self.sim_width-1, np.maximum(0.0,y));

		if round_vals:
			return [int(i_coord), int(j_coord)]
		else:
			return [i_coord, j_coord]

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

			interpolated_grid = np.zeros((self.sim_height*self.sim_width))

			for i in range(self.sim_height*self.sim_width):
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

			marker.scale.x = self.res
			marker.scale.y = self.res
			marker.scale.z = self.human_height
			for t in range(time):
				grid = self.interpolate_grid(t)

				if grid is not None:
					for i in range(len(grid)):
						(row, col) = self.state_to_coor(i)
						real_coord = self.sim_to_real_coord([row, col])

						color = ColorRGBA()
						color.a = np.sqrt((1 - (time-1)/self.fwd_tsteps)*grid[i])
						color.r = np.sqrt(grid[i])
						color.g = np.minimum(np.absolute(np.log(grid[i])),5.0)/5.0
						color.b = 0.9*np.minimum(np.absolute(np.log(grid[i])),5.0)/5.0 + 0.5*np.sqrt(grid[i])
						if grid[i] < self.prob_thresh:
							color.a = 0.0
						marker.colors.append(color)

						pt = Vector3()
						pt.x = real_coord[0]
						pt.y = real_coord[1]
						pt.z = self.human_height/2.0
						marker.points.append(pt)
					
			self.grid_vis_pub.publish(marker)
			
	# ---- ROS Message Conversion ---- #

	def grid_to_message(self):
		"""
		Converts OccupancyGridTime structure to ROS msg
		"""
		timed_grid = OccupancyGridTime()
		timed_grid.gridarray = [None]*self.fwd_tsteps
		timed_grid.object_num = int(self.human_number) 

		curr_time = rospy.Time.now()

		for t in range(self.fwd_tsteps):
			grid_msg = ProbabilityGrid()

			# Set up the header.
			grid_msg.header.stamp = curr_time + rospy.Duration(t*self.deltat)
			grid_msg.header.frame_id = "/world"

			# .info is a nav_msgs/MapMetaData message. 
			grid_msg.resolution = self.res
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
		marker.scale.x = self.res
		marker.scale.y = self.res
		marker.scale.z = self.res
		marker.color.a = 1.0
		marker.color.r = color[0]
		marker.color.g = color[1]
		marker.color.b = color[2]

		marker.pose.position.x = xy[0]
		marker.pose.position.y = xy[1]

		return marker

	def pose_to_marker(self, xypose, color=[1.0,0.0,0.0]):
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
		marker.scale.z = self.human_height + 0.001
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
	human = HumanPrediction()
