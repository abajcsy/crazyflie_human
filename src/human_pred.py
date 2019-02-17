#!/usr/bin/env python2.7
from __future__ import division
import rospy
import rospkg
import sys, select, os
import numpy as np
import time
import pickle

from std_msgs.msg import String, Float32, ColorRGBA
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Pose2D, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from crazyflie_human.msg import OccupancyGridTime, ProbabilityGrid
from fastrack_msgs.msg import Trajectory
from crazyflie_msgs.msg import PositionVelocityYawStateStamped
# Get the path of this file, go up two directories, and add that to our 
# Python path so that we can import the pedestrian_prediction module.
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from pedestrian_prediction.pp.mdp import GridWorldMDP
from pedestrian_prediction.pp.mdp.expanded import GridWorldExpanded
from pedestrian_prediction.pp.inference import hardmax as inf

Actions = GridWorldMDP.Actions

class HumanPrediction(object):
	"""
	This class models and predicts human motions in a 2D planar environment.
	It stores:
		- human's tracked trajectory 
		- occupancy grid of states human is likely to go to
		- moving obstacle representing human future motionr
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
			# self.goal_pub.publish(marker_array)
			rate.sleep()

		# When ROS shuts down, save metrics.
		if hasattr(self, "metrics_file_name"):
			rospack = rospkg.RosPack()
	        path = rospack.get_path('data_logger') + "/data/"
	        pickle.dump(self.pred_compute_times, open(path + self.metrics_file_name + ".py", "wb"))


	def load_parameters(self):
		"""
		Loads all the important paramters of the human sim
		"""

		self.human_number = str(rospy.get_param("human_number"))

		# --- real-world params ---# 

		low = rospy.get_param("state/lower"+self.human_number)
		up = rospy.get_param("state/upper"+self.human_number)

		# get real-world measurements of experimental space
		self.real_height = up[1] - low[1] 
		self.real_width = up[0] - low[0] 
		# store the lower and upper measurements
		self.real_lower = low
		self.real_upper = up

		# (real-world) start and goal locations 
		self.real_start = rospy.get_param("pred/human"+self.human_number+"_real_start")
		self.real_goals = rospy.get_param("pred/human"+self.human_number+"_real_goals")

		# --- simulation params ---# 

		# measurements of gridworld 
		self.sim_height = int(rospy.get_param("pred/sim_height"+self.human_number))
		self.sim_width = int(rospy.get_param("pred/sim_width"+self.human_number))

		# resolution (real meters)/(sim dim) (m/cell)
		self.res_x = self.real_width/self.sim_width
		self.res_y = self.real_height/self.sim_height

		# (simulation) start and goal locations
		self.sim_start = self.real_to_sim_coord(self.real_start)
		self.sim_goals = [self.real_to_sim_coord(g) for g in self.real_goals]

		# simulation forward prediction parameters
		self.fwd_tsteps = rospy.get_param("pred/fwd_tsteps")

		self.human_height = rospy.get_param("pred/human_height")
		self.prob_thresh = rospy.get_param("pred/prob_thresh")	

		# hidden state volatility in HMM
		self.epsilon_dest = rospy.get_param("pred/epsilon_dest")
		self.epsilon_beta = rospy.get_param("pred/epsilon_beta")

		# stores 2D array of size (fwd_tsteps) x (height x width) of probabilities
		self.occupancy_grids = None

		# model type; initially start with conservative FRT
		self.model_type = "FRT"
		self.model_list = rospy.get_param("pred/model_list")

		# metrics file name
		if rospy.has_param("pred/metrics_file_name"):
			self.metrics_file_name = rospy.get_param("pred/metrics_file_name")

		# stores list of beta values for each goal
		self.beta_model = rospy.get_param("beta")
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
		self.gridworld = GridWorldExpanded(self.sim_width, self.sim_height)

		# color to use to represent this human
		self.color = rospy.get_param("pred/human"+self.human_number+"_color")

		# tracks the human's state over time
		self.real_human_traj = None
		self.sim_human_traj = None

		# store the previous time to compute deltat
		self.prev_t = None
		self.prev_pos = None
		self.robot_pos = None

		# get the speed of the human (meters/sec)
		self.human_vel = rospy.get_param("pred/human_vel")

		# compute the timestep (seconds/cell)
		self.deltat = self.res_x/self.human_vel

		# Benchmarks: prediction compute time, trajectory time, distance to human
		self.pred_compute_times = None
		self.traj_time = None
		self.dist_to_human = None

		# robot prefixes
		self.robot_prefix = rospy.get_param("sim/robot_prefix")
		self.robot_traj_states = None
		self.robot_traj_times = None

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
		self.traj_sub = rospy.Subscriber('/traj'+self.robot_prefix, Trajectory, 
											self.trajectory_callback, queue_size=1)
		self.robot_sub = rospy.Subscriber('/state/position_velocity_yaw'+self.robot_prefix,
											PositionVelocityYawStateStamped,
											self.robot_state_callback, queue_size=1)

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

	# ---- Model Switching Functionality ---- #
	def trajectory_callback(self, msg):
		"""
		Grabs the most recent trajectory and compute metrics
		"""
		self.robot_traj_states = msg.states
		self.robot_traj_times = msg.times

	def robot_state_callback(self, msg):
	    self.robot_pos = [msg.state.x, msg.state.y]
    
    # Distance between human and robot.
	def HR_distance(self):
		if self.prev_pos is None or self.robot_pos is None:
			return float("inf")
		return np.linalg.norm(np.array(self.prev_pos) - np.array(self.robot_pos))

	def traj_length(self):
		if self.robot_traj_states is None:
			return float("inf")
		print "MODEL:", self.model_type, "; traj: ", self.robot_traj_states
		return sum([np.linalg.norm(np.array(self.robot_traj_states[i+1].x[:2]) - np.array(s.x[:2])) for i, s in enumerate(self.robot_traj_states[:-1])])

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

			# update the map with where the human is at the current time
			self.update_human_traj(xypose)

			# First compute trajectory in FREESPACE
			self.model_type = "FREESPACE"
			self.infer_occupancies() 
			if self.occupancy_grids is not None:
				self.occu_pub.publish(self.grid_to_message())
				time.sleep(0.1)
			freespace_length = self.traj_length()

			# Compute trajectory with FRT and compare to FREESPACE
			self.model_type = "FRT"
			self.infer_occupancies() 
			if self.occupancy_grids is not None:
				self.occu_pub.publish(self.grid_to_message())
				time.sleep(0.1)
			frt_length = self.traj_length()

			# If FRT not good enough, compute trajectory with model_list
			if freespace_length is not float("inf") and frt_length is not float("inf"):
				print "freespace_length: ", freespace_length, "; frt_length: ", frt_length
				if abs(freespace_length - frt_length) > 1.0:
					self.model_type = self.model_list[0]
					self.infer_occupancies() 
					if self.occupancy_grids is not None:
						self.occu_pub.publish(self.grid_to_message())
						time.sleep(0.1)

			# adjust the deltat based on the observed measurements
			if self.prev_pos is not None:
				self.human_vel = np.linalg.norm((np.array(xypose) - np.array(self.prev_pos)))/time_diff
				self.deltat = np.minimum(np.maximum(self.res_x/self.human_vel,0.05),0.2)

			if self.occupancy_grids is not None:
				self.visualize_occugrid(2)

			self.prev_pos = xypose	
			
	def compute_metrics(self, compute_time):
		if self.pred_compute_times is None:
			self.pred_compute_times = [compute_time]
		else:
			self.pred_compute_times.append(compute_time)
		if self.robot_traj_times is not None:
			self.traj_time = (self.robot_traj_times[-1] - self.robot_traj_times[0])
		self.dist_to_human = self.HR_distance()
		
		average_compute_time = sum(self.pred_compute_times)/len(self.pred_compute_times)

		print "Human ", self.human_number, ", Model: ",self.model_type, ", Avg Compute Time: ", average_compute_time
		print "Human  ", self.human_number, ", Model: ",self.model_type, ", Distance from Robot: ", self.dist_to_human
		print "Human ", self.human_number, ", Model: ",self.model_type, ", Replanned Trajectory Time: ", self.traj_time, "\n"		

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

		# METRICS
		s = rospy.Time().now()

		if self.real_human_traj is None or self.sim_human_traj is None:
			print "Can't infer occupancies -- human hasn't appeared yet!"
			return 

		# Get list of goals in 2D 
		dest_list = [self.gridworld.coor_to_state(g[0], g[1]) for g in self.sim_goals]

		# Convert the human trajectory points from real-world to 2D grid values. 
		traj = [self.real_to_sim_coord(x, round_vals=True) for x in self.real_human_traj] 
  
  		if self.model_type == "BOLTZMANN":
			(self.occupancy_grids, self.beta_occu, self.dest_beta_prob) = inf.state.infer_joint(self.gridworld, 
				dest_list, self.betas, T=self.fwd_tsteps, use_gridless=True, priors=self.dest_beta_prob,
				traj=traj[-2:], epsilon_dest=self.epsilon_dest, epsilon_beta=self.epsilon_beta, verbose_return=True)
		elif self.model_type == "FREESPACE":
			# We are planning in free space, so the grid is empty everywhere
			self.occupancy_grids = np.zeros((self.fwd_tsteps+1, self.gridworld.S))
		elif self.model_type == "FRT":
			# Planning with Forward reachable sets.
			# Get all humans, draw a circle of radius T around them, occupy the grid around there
			self.occupancy_grids = np.zeros((self.fwd_tsteps+1, self.gridworld.S))
			H_coords = self.real_to_sim_coord(self.prev_pos)
			for T in range(self.fwd_tsteps + 1):
				for i in range(H_coords[0] - T, H_coords[0] + T + 1):
					for j in range(H_coords[1] - T, H_coords[1] + T + 1):
						if 0 <= i < self.gridworld.rows and 0 <= j < self.gridworld.cols:
							self.occupancy_grids[T, self.gridworld.coor_to_state(i,j)] = 1

		# METRICS
		e = rospy.Time().now()
		compute_time = (e-s).to_sec()
		self.compute_metrics(compute_time)

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

	def sim_to_real_coord(self, sim_coord):
		"""
		Takes [x,y] coordinate in simulation frame and returns a rotated and 
		shifted	value in the ROS coordinates
		"""
		return [self.real_lower[0] + 0.5*self.res_x + sim_coord[0]*self.res_x, 
				self.real_upper[1] - 0.5*self.res_y - sim_coord[1]*self.res_y]

	def real_to_sim_coord(self, real_coord, round_vals=True):
		"""
		Takes [x,y] coordinate in the ROS real frame, and returns a rotated and 
		shifted	value in the simulation frame
		-- round_vals: 
					True - gives an [i,j] integer-valued grid cell entry.
					False - gives a floating point value on the grid cell
		"""
		if round_vals:
			x = np.floor((real_coord[0] - self.real_lower[0])/self.res_x)
			y = np.floor((self.real_upper[1] - real_coord[1])/self.res_y)
		else:
			x = (real_coord[0] - self.real_lower[0])/self.res_x
			y = (self.real_upper[1] - real_coord[1])/self.res_y

		i_coord = np.minimum(self.sim_width-1, np.maximum(0.0,x));
		j_coord = np.minimum(self.sim_height-1, np.maximum(0.0,y));

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

			marker.scale.x = self.res_x
			marker.scale.y = self.res_y
			marker.scale.z = self.human_height
			for t in range(time):
				#grid = self.interpolate_grid(t)
				grid = self.occupancy_grids[t]
				if grid is not None:
					for i in range(len(grid)):
						(row, col) = self.gridworld.state_to_coor(i)
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
			grid_msg.resolution = self.res_x
			grid_msg.width = self.sim_width
			grid_msg.height = self.sim_height
			grid_msg.lower_x = self.real_lower[0]
			grid_msg.lower_y = self.real_lower[1]
			grid_msg.upper_x = self.real_upper[0]
			grid_msg.upper_y = self.real_upper[1]

			# Rotated maps are not supported... 
			origin_x=0.0 
			origin_y=0.0 
			grid_msg.origin = Pose(Point(origin_x, origin_y, 0), Quaternion(0, 0, 0, 1))

			# Assert that all occupancies are in [0, 1].
			assert self.occupancy_grids[t].max() <= 1.0 +1e-8 and self.occupancy_grids[t].min() >= 0.0 - 1e-8
			if self.model_type == "BOLTZMANN":
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
		marker.scale.x = self.res_x
		marker.scale.y = self.res_y
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
