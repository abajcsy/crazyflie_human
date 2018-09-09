#!/usr/bin/env python2.7
import rospy
import numpy as np
import time

from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import  Vector3
from crazyflie_human.msg import OccupancyGridTime, ProbabilityGrid
from visualization_msgs.msg import Marker, MarkerArray

class PredictionVisualizer(object):
	"""
	This class visualizies predictions of a human's motion in 
	a 2D planar environment
	"""

	def __init__(self):
		# create ROS node
		rospy.init_node('prediction_visualization', anonymous=True)

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

		# measurements of gridworld
		self.sim_height = int(rospy.get_param("pred/sim_height"))
		self.sim_width = int(rospy.get_param("pred/sim_width"))

		# simulation forward prediction parameters
		self.fwd_tsteps = rospy.get_param("pred/fwd_tsteps")

		self.human_height = rospy.get_param("pred/human_height")
		self.prob_thresh = rospy.get_param("pred/prob_thresh")

		# resolution (m/cell)
		self.res = rospy.get_param("pred/resolution")

		# simulation forward prediction parameters
		self.fwd_tsteps = rospy.get_param("pred/fwd_tsteps")

		# --- real-world params ---# 

		low = rospy.get_param("state/lower")
		up = rospy.get_param("state/upper")

		# get real-world measurements of experimental space
		self.real_height = up[1] - low[1] 
		self.real_width = up[0] - low[0] 
		# store the lower and upper measurements
		self.real_lower = low
		self.real_upper = up

		# store the occupancy grid
		self.occupancy_grids = None

		# visualize every 0.1 seconds
		self.visualization_delta = 0.05
		self.prev_t = rospy.Time().now()

	def register_callbacks(self):
		"""
		Sets up publishers and subscribers
		"""

		self.occu_sub = rospy.Subscriber('/occupancy_grid_time', OccupancyGridTime, self.occu_grid_callback, queue_size=1)	
		self.grid_vis_pub = rospy.Publisher('/occu_grid_marker', Marker, queue_size=0)
			
	def occu_grid_callback(self, msg):
		# convert the message into the data structure
		self.from_ROSMsg(msg)

		# show the forward prediction
		#for i in range(1,5):
		#	curr_time = rospy.Time.now()
		#	diff_t = (curr_time - self.prev_t).to_sec()
		#	if (diff_t >= self.visualization_delta):
		#		self.prev_t += rospy.Duration.from_sec(self.visualization_delta)

		#s = rospy.Time().now()

		# show fixed block of fwd_tsteps
		self.visualize_occugrid(3)

		#e = rospy.Time().now()
		# print "time to visualize grid: ", (e.to_sec()-s.to_sec())

		# show waves
		#for time in range(self.fwd_tsteps):
		#	self.visualize_waves(time)

	def from_ROSMsg(self, msg):
		"""
		Converts occugrid message into structure
		"""

		self.occupancy_grids = [None]*self.fwd_tsteps

		for i, grid in enumerate(msg.gridarray):
			self.occupancy_grids[i] = grid.data

	def visualize_occugrid(self, time):
		"""
		Visualizes occupancy grid for all grids in time
		"""

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

					#print "real_coord: ", real_coord
					#print "coord prob: ", grid[i]
					
			self.grid_vis_pub.publish(marker)

	def visualize_waves(self, time):
		"""
		Visualizes occupancy grid at time
		"""

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

			grid = self.interpolate_grid(time)

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

				#print "real_coord: ", real_coord
				#print "coord prob: ", grid[i]
					
			self.grid_vis_pub.publish(marker)


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
				if curr > self.prob_thresh:
					print "At location", self.gridworld.state_to_coor(i)
					print "(prev_t, next_t): ", (prev_t, next_t)
					print "prev: ", prev
					print "next: ", next
					print "curr: ", curr
				interpolated_grid[i] = curr

			return interpolated_grid

if __name__ == '__main__':
	visualizer = PredictionVisualizer()
