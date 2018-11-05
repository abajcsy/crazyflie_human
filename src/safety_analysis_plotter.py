#!/usr/bin/env python2.7
from __future__ import division
import rospy
import sys, select, os
import numpy as np
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib.cm as cmx
import colormaps as cmaps

import pickle

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm

from std_msgs.msg import String, Float32, ColorRGBA
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Pose2D, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from crazyflie_human.msg import OccupancyGridTime, ProbabilityGrid

# Get the path of this file, go up two directories, and add that to our 
# Python path so that we can import the pedestrian_prediction module.
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")

from pedestrian_prediction.pp.mdp import GridWorldMDP
from pedestrian_prediction.pp.mdp.expanded import GridWorldExpanded
from pedestrian_prediction.pp.inference import hardmax as inf

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

Actions = GridWorldMDP.Actions

class SafetyAnalysisPlotter(object):
	"""
	This class listens to human data and compares: 
		(a) the forward reach set (FRS)
		(b) the fixed low-confidence predictions 
		(c) the fixed high-confidence predictions 
		(d) the adaptive confidence predictions 
	for t secs in the future. It plots the all of these for
	one snapshot in time. 
	"""

	def __init__(self, real_plot_time, pred_plot_time):

		rospy.init_node('safety_analysis_plotter', anonymous=True)

		# simulates world until this time and then computes 
		# compute FRS and prediction comparison
		self.real_plot_time = real_plot_time

		# plots predictions this far into the future (in seconds)
		self.pred_plot_time = pred_plot_time

		# load all the prediction params and setup subscriber/publishers
		self.load_parameters()
		self.register_callbacks()

		rate = rospy.Rate(100)

		while not rospy.is_shutdown():
			rate.sleep()

	def load_parameters(self):
		"""
		Loads all the important paramters of the human sim
		"""

		# --- real-world params ---# 

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

		# grid world representing the experimental environment
		self.gridworld = GridWorldExpanded(self.sim_height, self.sim_width)

		self.irr_betas = rospy.get_param("pred/beta_irrational")
		self.rat_betas = rospy.get_param("pred/beta_rational")
		self.adapt_betas = rospy.get_param("pred/beta_adaptive")

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
		self.real_start = rospy.get_param("pred/human_real_start")
		self.real_goals = rospy.get_param("pred/human_real_goals")

		# (simulation) start and goal locations
		self.sim_start = self.real_to_sim_coord(self.real_start)
		self.sim_goals = [self.real_to_sim_coord(g) for g in self.real_goals]

		# color to use to represent this human
		self.color = rospy.get_param("pred/human_color")

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

		# ----- analysis parameters ----- #

		# used to check if its time to plot comparison
		self.first_human_msg_time = None 

	def register_callbacks(self):
		"""
		Sets up all the publishers/subscribers needed.
		"""
		self.human_sub = rospy.Subscriber('/human_pose'+self.human_number, PoseStamped, 
											self.human_state_callback, queue_size=1)

	def human_state_callback(self, msg):
		"""
		Grabs the human's state from the mocap publisher
		"""
		curr_time = rospy.Time.now()
		xypose = self.make_valid_state([msg.pose.position.x, msg.pose.position.y])

		if self.first_human_msg_time is None:
			self.first_human_msg_time = curr_time

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

			# ------- SAFETY ANALYSIS PLOTTING ------- #

			elapsed_time = (curr_time - self.first_human_msg_time).to_sec() 
			print "elapsed time: ", elapsed_time
			if np.abs(elapsed_time - self.real_plot_time) < 0.25: #elapsed_time >= self.real_plot_time:
				print "starting safety analysis..."
				self.plot_safety_analysis()
				rospy.signal_shutdown("finished safety analysis. exiting...")
				#self.plot_threshold_analysis()
				#rospy.signal_shutdown("finished collision threshold analysis. exiting...")

			# ------- SAFETY ANALYSIS PLOTTING ------- #

			# adjust the deltat based on the observed measurements
			if self.prev_pos is not None:
				self.human_vel = np.linalg.norm((np.array(xypose) - np.array(self.prev_pos)))/time_diff
				self.deltat = np.minimum(np.maximum(self.res/self.human_vel,0.05),0.2)
				#self.deltat = np.minimum(np.maximum(self.res/self.human_vel,0.05), self.res/0.5)
				print "vel: ", self.human_vel
				print "dt: ", self.deltat

			self.prev_pos = xypose	

	def plot_safety_analysis(self):
		"""
		Plots a snapshot in time of the forward reachable set and the 
		predictions for low, high, and adaptive confidence. 
		"""

		# infer the new human occupancy map from the current state
		(irr_occugrids, _, _) = self.infer_occupancies(self.irr_betas)
		(rat_occugrids, _, _) = self.infer_occupancies(self.rat_betas)
		(adapt_occugrids, _, beta_prob) = self.infer_occupancies(self.adapt_betas)

        # keep track of a matplotlib figure for rendering
		fig = plt.figure()

		# 1,1,1, = num rows, num cols, idx
		ax = fig.add_subplot(1,1,1, aspect="equal")
		ax.set_xlim(self.real_lower[0]-self.res, self.real_upper[0]+self.res)
		ax.set_ylim(self.real_lower[1]-self.res, self.real_upper[1]+self.res)
		plt.hold(True)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		# color setup
		r = 29.0/255.0
		g = 45.0/255.0
		b = 71.0/255.0

		# get the colormap
		plt.register_cmap(name='viridis', cmap=cmaps.viridis)
		plt.set_cmap(cmaps.viridis)
		colormap = plt.cm.get_cmap('viridis')

		#cmap_name = 'greenish_map'
		#(30./255., 56./255., 78./255.), (23./255.,125./255.,123./225.), (196./255., 222./255., 118./255.)
		#(105./255., 0, 150./255.)
		#green_colors = [(0., 105./255., 226/255.), (114./255., 214./255., 0./255.)]
		#green_colors = [(12./255., 0./255., 150./255.), (0./255., 237./255., 225./255.), (0./255., 150./255., 50./255.)]
		#n_bins = 10  # Discretizes the interpolation into bins
		#colormap = colors.LinearSegmentedColormap.from_list(cmap_name, green_colors, N=n_bins)
		cNorm = colors.Normalize(vmin=np.log(0.05),vmax=np.log(10))
		scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)
		
		# get the last human state and plot the human
		human_position = self.real_human_traj[-1:]
		human_x = human_position[0][0]
		human_y = human_position[0][1]

		# plot the forward reachable set
		vel = self.res*np.sqrt(2)/self.deltat
		radius = vel*self.pred_plot_time
		fwd_reach_patch = patches.Circle((human_x, human_y), radius, fill=None, linewidth=2.0, linestyle='dashed', edgecolor=[r,g,b], alpha=1)
		ax.add_patch(fwd_reach_patch)

		def reverse_log_beta(beta):
			max_beta = 10.0
			min_beta = 0.05
			return np.log(max_beta) - (np.log(beta) - np.log(min_beta))

		# ----- plot IRRATIONAL predictions ----- #
		irrgrid = self.interpolate_grid(irr_occugrids, self.pred_plot_time)
		irr_beta = 0.05
		irr_color = scalarMap.to_rgba(reverse_log_beta(irr_beta))
		diff = irr_color- np.array([.2,.2,.2,0])
		dark_irr_color  = np.clip(diff, 0, 1)
		print "irr color: ", irr_color
		print "dark: ", dark_irr_color

		for i in range(len(irrgrid)):
			(row, col) = self.state_to_coor(i)
			real_coord = self.sim_to_real_coord([row, col])

			if irrgrid[i] >= self.prob_thresh:
				bottom_left = [real_coord[0]-self.res/2.0, real_coord[1]-self.res/2.0]
				#grid_patch1 = patches.Rectangle(bottom_left, self.res, self.res, alpha=1, fill=None, edgecolor=[r,g,b])
				grid_patch1 = patches.Rectangle(bottom_left, self.res, self.res, alpha=0.7, linewidth=0.5, edgecolor=dark_irr_color, facecolor=irr_color) #facecolor=[196./255., 222./255., 118./255.])
				ax.add_patch(grid_patch1)
		# --------------------------------------- #

		# ----- plot ADAPTIVE predictions ----- #
		adaptgrid = self.interpolate_grid(adapt_occugrids, self.pred_plot_time)
		avg_beta = 0.0
		for (ii,b) in enumerate(self.adapt_betas):
			avg_beta += b*beta_prob[0][ii]
		print "avg beta: ", avg_beta
		adapt_color = scalarMap.to_rgba(reverse_log_beta(avg_beta))
		diff = adapt_color- np.array([.1,.1,.1,0])
		dark_adapt_color = np.clip(diff, 0, 1)

		for i in range(len(adaptgrid)):
			(row, col) = self.state_to_coor(i)
			real_coord = self.sim_to_real_coord([row, col])
			
			if adaptgrid[i] >= self.prob_thresh:
				bottom_left = [real_coord[0]-self.res/2.0, real_coord[1]-self.res/2.0]
				#grid_patch2 = patches.Rectangle(bottom_left, self.res, self.res, alpha=0.4, linewidth=0.0, facecolor=[r,g,b])
				grid_patch2 = patches.Rectangle(bottom_left, self.res, self.res, alpha=0.7, linewidth=0.5, edgecolor=dark_adapt_color, facecolor=adapt_color) #facecolor=[23./255.,125./255.,123./225.])
				ax.add_patch(grid_patch2)
		# --------------------------------------- #
		
		# ----- plot RATIONAL predictions ----- #
		ratgrid = self.interpolate_grid(rat_occugrids, self.pred_plot_time)
		rat_beta = 10.0
		rat_color = scalarMap.to_rgba(reverse_log_beta(rat_beta))
		diff = rat_color- np.array([.1,.1,.1,0])
		dark_rat_color  = np.clip(diff, 0, 1)

		for i in range(len(ratgrid)):
			(row, col) = self.state_to_coor(i)
			real_coord = self.sim_to_real_coord([row, col])

			if ratgrid[i] >= self.prob_thresh:
				bottom_left = [real_coord[0]-self.res/2.0, real_coord[1]-self.res/2.0]
				#grid_patch3 = patches.Rectangle(bottom_left, self.res, self.res, alpha=1.0, fill=None, edgecolor=[r,g,b], hatch="///")
				grid_patch3 = patches.Rectangle(bottom_left, self.res, self.res, alpha=0.7, linewidth=0.5, edgecolor=dark_rat_color, facecolor=rat_color) #facecolor=[30./255., 56./255., 78./255.])
				ax.add_patch(grid_patch3)
		# --------------------------------------- #

		# plot the human
		human_patch = patches.Circle((human_x, human_y), self.res/2.0, color="k")
		ax.add_patch(human_patch)

		# plot the goal
		goal = self.real_goals[0]
		goal_patch = patches.Circle((goal[0], goal[1]), self.res/2.0, color="r")
		ax.add_patch(goal_patch)

		# plot legend
		leg = ax.legend((fwd_reach_patch, grid_patch1, grid_patch3, grid_patch2), (r"FRS", r"$\beta$ low", r"$\beta$ high", r"$\beta$ infer"), prop={'size': 20}, loc="center right")

		# plot scale marker
		xmin = -0.5
		xmax = xmin + self.res
		yval = human_y - self.human_vel*self.pred_plot_time - 1
		plt.text(xmin, yval+0.05, r'1 ft', fontsize=16)
		plt.text(xmin, yval-0.1, r'$\approx$ 0.3 m', fontsize=16)
		plt.plot([xmin, xmax], [yval, yval], color='k', linestyle='-', linewidth=2)

		#plt.axis('off')
		plt.xticks([])
		plt.yticks([])
		#ax.axis('off')
		plt.show()
		plt.pause(0.1)

		# save out image
		path = "/home/abajcsy/GitHub/human_estimator/src/crazyflie_human/imgs/"
		filename = path + "comp_r" + str(self.real_plot_time) + "p" + str(self.pred_plot_time) + ".png"
		print "saving image ", filename
		plt.savefig(filename, bbox_inches='tight')


	def plot_threshold_analysis(self):
		"""
		Plots a snapshot in time of the forward reachable set and the 
		predictions for low, high, and adaptive confidence
		over a range of colision probability thresholds. 
		"""

		# infer the new human occupancy map from the current state
		(irr_occugrids, _, _) = self.infer_occupancies(self.irr_betas)
		(rat_occugrids, _, _) = self.infer_occupancies(self.rat_betas)
		(adapt_occugrids, _, beta_prob) = self.infer_occupancies(self.adapt_betas)

		# save pickle files
		#pickle.dump(irr_occugrids, open("/home/abajcsy/GitHub/human_estimator/src/crazyflie_human/src/irrational_distribution.p", "wb"))
		#pickle.dump(rat_occugrids, open("/home/abajcsy/GitHub/human_estimator/src/crazyflie_human/src/rational_distribution.p", "wb"))
		#pickle.dump(adapt_occugrids, open("/home/abajcsy/GitHub/human_estimator/src/crazyflie_human/src/adaptive_distribution.p", "wb"))
		#print "SAVED OUT DISTRIBUTIONS"

        # keep track of a matplotlib figure for rendering
		fig = plt.figure()

		max_coll_thresh = 0.05
		min_coll_thresh = 0.0001
		coll_thresh_step = 0.001
		# enable 3d plotting
		ax = fig.gca(projection='3d')
		ax.set_xlim(self.real_lower[0]-self.res, self.real_upper[0]+self.res)
		ax.set_ylim(self.real_lower[1]-self.res, self.real_upper[1]+self.res)
		#ax.set_zlim(min_coll_thresh, max_coll_thresh)
		ax.set_zlim(0,0.1)
		#ax.set_zscale("log")
		plt.hold(True)
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')

		# get the colormap
		plt.register_cmap(name='viridis', cmap=cmaps.viridis)
		plt.set_cmap(cmaps.viridis)
		colormap = plt.cm.get_cmap('viridis')
		cNorm = colors.Normalize(vmin=np.log(0.05),vmax=np.log(10))
		scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)
		
		# color setup
		r = 29.0/255.0
		g = 45.0/255.0
		b = 71.0/255.0

		# get the last human state and plot the human
		human_position = self.real_human_traj[-1:]
		human_x = human_position[0][0]
		human_y = human_position[0][1]

		# plot the forward reachable set
		vel = self.res*np.sqrt(2)/self.deltat
		radius = vel*self.pred_plot_time
		fwd_reach_patch = patches.Circle((human_x, human_y), radius, fill=None, linewidth=2.0, linestyle='dashed', edgecolor=[r,g,b], alpha=1)
		ax.add_patch(fwd_reach_patch)
		art3d.pathpatch_2d_to_3d(fwd_reach_patch, z=0, zdir="z")

		def reverse_log_beta(beta):
			max_beta = 10.0
			min_beta = 0.05
			return np.log(max_beta) - (np.log(beta) - np.log(min_beta))

		# ----- plot IRRATIONAL predictions ----- #
		
		irrgrid = self.interpolate_grid(irr_occugrids, self.pred_plot_time)
		irr_beta = 0.05
		irr_color = scalarMap.to_rgba(reverse_log_beta(irr_beta))
		diff = irr_color- np.array([.2,.2,.2,0])
		dark_irr_color  = np.clip(diff, 0, 1)


		#for coll_thresh in np.arange(min_coll_thresh, max_coll_thresh, coll_thresh_step):
		for i in range(len(irrgrid)):
			(row, col) = self.state_to_coor(i)
			real_coord = self.sim_to_real_coord([row, col])

			#if irrgrid[i] >= coll_thresh:
			if irrgrid[i] > 0.0:
				bottom_left = [real_coord[0]-self.res/2.0, real_coord[1]-self.res/2.0]
			
				#grid_patch1 = patches.Rectangle(bottom_left, self.res, self.res, alpha=1, fill=None, edgecolor=[r,g,b])
				#grid_patch1 = patches.Rectangle(bottom_left, self.res, self.res, alpha=0.2, linewidth=0.2, edgecolor=dark_irr_color, facecolor=irr_color) #facecolor=[196./255., 222./255., 118./255.])
				#ax.add_patch(grid_patch1)
				#print "prob: ", irrgrid[i]
				#art3d.pathpatch_2d_to_3d(grid_patch1, z=irrgrid[i], zdir="z")
				ax.bar3d(real_coord[0], real_coord[1], 0, self.res, self.res, irrgrid[i], linewidth=0, color=irr_color, alpha=0.3)#, shade=True)
		# --------------------------------------- #
		
		
		# ----- plot ADAPTIVE predictions ----- #
		adaptgrid = self.interpolate_grid(adapt_occugrids, self.pred_plot_time)
		avg_beta = 0.0
		for (ii,b) in enumerate(self.adapt_betas):
			avg_beta += b*beta_prob[0][ii]
		adapt_color = scalarMap.to_rgba(reverse_log_beta(avg_beta))
		diff = adapt_color- np.array([.1,.1,.1,0])
		dark_adapt_color = np.clip(diff, 0, 1)

		#for coll_thresh in np.arange(min_coll_thresh, max_coll_thresh, coll_thresh_step):
		for i in range(len(adaptgrid)):
			(row, col) = self.state_to_coor(i)
			real_coord = self.sim_to_real_coord([row, col])
			
			#if adaptgrid[i] >= coll_thresh:
			if adaptgrid[i] > 0.0:
				bottom_left = [real_coord[0]-self.res/2.0, real_coord[1]-self.res/2.0]
				#grid_patch2 = patches.Rectangle(bottom_left, self.res, self.res, alpha=0.4, linewidth=0.0, facecolor=[r,g,b])
				#grid_patch2 = patches.Rectangle(bottom_left, self.res, self.res, alpha=0.8, linewidth=0.2, edgecolor=dark_adapt_color, facecolor=adapt_color) #facecolor=[23./255.,125./255.,123./225.])
				#ax.add_patch(grid_patch2)
				#art3d.pathpatch_2d_to_3d(grid_patch2, z=np.log(adaptgrid[i]), zdir="z")
				ax.bar3d(real_coord[0], real_coord[1], 0, self.res, self.res, adaptgrid[i], linewidth=0, color=adapt_color, alpha=0.8)#, shade=True)
		# --------------------------------------- #
		
		
		# ----- plot RATIONAL predictions ----- #
		ratgrid = self.interpolate_grid(rat_occugrids, self.pred_plot_time)
		rat_beta = 10.0
		rat_color = scalarMap.to_rgba(reverse_log_beta(rat_beta))
		diff = rat_color- np.array([.1,.1,.1,0])
		dark_rat_color  = np.clip(diff, 0, 1)

		#for coll_thresh in np.arange(min_coll_thresh, max_coll_thresh, coll_thresh_step):
		for i in range(len(ratgrid)):
			(row, col) = self.state_to_coor(i)
			real_coord = self.sim_to_real_coord([row, col])

			#if ratgrid[i] >= coll_thresh:
			if ratgrid[i] > 0.0:
				bottom_left = [real_coord[0]-self.res/2.0, real_coord[1]-self.res/2.0]
				#grid_patch3 = patches.Rectangle(bottom_left, self.res, self.res, alpha=1.0, fill=None, edgecolor=[r,g,b], hatch="///")
				#grid_patch3 = patches.Rectangle(bottom_left, self.res, self.res, alpha=0.2, linewidth=0.2, edgecolor=dark_rat_color, facecolor=rat_color) #facecolor=[30./255., 56./255., 78./255.])
				#ax.add_patch(grid_patch3)
				#art3d.pathpatch_2d_to_3d(grid_patch3, z=np.log(ratgrid[i]), zdir="z")
				ax.bar3d(real_coord[0], real_coord[1], 0, self.res, self.res, ratgrid[i], linewidth=0, color=rat_color, alpha=0.3)#, shade=True)
		# --------------------------------------- #
		
		# plot the human
		human_patch = patches.Circle((human_x, human_y), self.res/2.0, color="k")
		ax.add_patch(human_patch)
		art3d.pathpatch_2d_to_3d(human_patch, z=0, zdir="z")

		# plot the goal
		#goal = self.real_goals[0]
		#goal_patch = patches.Circle((goal[0], goal[1]), self.res/2.0, color="r")
		#ax.add_patch(goal_patch)

		# plot legend
		#leg = ax.legend((fwd_reach_patch, grid_patch1, grid_patch3, grid_patch2), (r"FRS", r"$\beta$ low", r"$\beta$ high", r"$\beta$ infer"), prop={'size': 20}, loc="center right")

		#plt.axis('off')
		#plt.xticks([])
		#plt.yticks([])
		ax.set_xlabel(r'$h_x$', fontsize=20)
		ax.set_ylabel(r'$h_y$', fontsize=20)
		ax.set_zlabel(r'$P(x_H)$', fontsize=20)
		plt.show()
		#plt.pause(0.1)

		# save out image
		path = "/home/abajcsy/GitHub/human_estimator/src/crazyflie_human/imgs/"
		filename = path + "coll_thresh_analysis.png"
		print "saving image ", filename
		plt.savefig(filename, bbox_inches='tight')

	def prob_to_color(self, time, prob):
		"""
		Converts probability value to color.
		"""
		alpha = np.sqrt((1 - (time-1)/self.fwd_tsteps)*prob)
		r = np.sqrt(prob)
		g = np.minimum(np.absolute(np.log(prob)),5.0)/5.0
		b = 0.9*np.minimum(np.absolute(np.log(prob)),5.0)/5.0 + 0.5*np.sqrt(prob)
		return (r,g,b,alpha)

	def infer_occupancies(self, betas):
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

		# OPTION 2: The line below feeds in the last human (s,a) pair and previous posterior
		# 			and does a recursive Bayesian update.
		(occupancy_grids, beta_occu, dest_beta_prob) = inf.state.infer_joint(self.gridworld, 
			dest_list, betas, T=self.fwd_tsteps, use_gridless=True, priors=None,
			traj=traj[-2:], epsilon_dest=self.epsilon_dest, epsilon_beta=self.epsilon_beta, verbose_return=True)

		return (occupancy_grids, beta_occu, dest_beta_prob)

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

	def interpolate_grid(self, occupancy_grids, future_time):
		"""
		Interpolates the grid at some future time (seconds)
		"""
		print "future_time: ", future_time

		if occupancy_grids is None:
			print "Occupancy grids are not created yet!"
			return None

		if future_time < 0:
			print "Can't interpolate for negative time!"
			return None

		if future_time > self.fwd_tsteps*self.deltat:
			print "Can't interpolate more than", self.fwd_tsteps*self.deltat, "seconds into future!"
			print "future_time =", future_time
			return None

		in_idx = -1
		for i in range(self.fwd_tsteps+1):
			if np.isclose(i*self.deltat, future_time, rtol=1e-05, atol=1e-08):
				in_idx = i
				break
	
		if in_idx != -1:
			# if interpolating exactly at the timestep
			return occupancy_grids[in_idx]
		else:
			prev_idx = np.floor(future_time/self.deltat)
			next_idx = np.ceil(future_time/self.deltat)

			prev_t = prev_idx*self.deltat
			next_t = next_idx*self.deltat

			print "prev t: ", prev_t
			print "next t: ", next_t

			low_grid = occupancy_grids[prev_idx]
			high_grid = occupancy_grids[next_idx]

			interpolated_grid = np.zeros((self.sim_height*self.sim_width))

			for i in range(self.sim_height*self.sim_width):
				prev = low_grid[i]
				next = high_grid[i]
				curr = prev + (next - prev) *((future_time - prev_t) / (next_t - prev_t))
				interpolated_grid[i] = curr

			return interpolated_grid

if __name__ == '__main__':
	print "Starting safety analysis predictor!"
	real_plot_time = 6 	# plot the analysis after this many seconds of world evolving  
	pred_plot_time = 1 	# compute forward reachable set for this many seconds in the future
	plotter = SafetyAnalysisPlotter(real_plot_time, pred_plot_time)