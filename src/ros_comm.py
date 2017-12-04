#!/usr/bin/env python
import rospy
import sys, select, os
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Pose2D
from visualization_msgs.msg import Marker
from human_msgs import OccupancyGridTime, ProbabilityGrid
import os
import numpy as np
import time

from human_pred import HumanPredMap
from sim_human import SimHuman

class SimWorld(object):

	def __init__(self, height, width):
		# ---- Human Map Setup ---- #

		# create human prediction map
		self.human_map = HumanPredMap(height, width)

		# start time of the experiment
		self.startT = time.time()

		# ---- ROS Setup ---- #

		# create ROS node
		rospy.init_node('sim_world', anonymous=True)

		# occupancy grid publisher
		self.occupancy_pub = rospy.Publisher('/occupancy_grid_time', OccupancyGridTime, queue_size=1)

		# small publishers for visualizing the start/goal
		self.goal_pub = rospy.Publisher('/goal_marker', Marker, queue_size=10)
		self.start_pub = rospy.Publisher('/start_marker', Marker, queue_size=10)

		# subscribe to the info of the human walking around the space
		self.human_subscriber = rospy.Subscriber('/human_pose', PoseStamped, self.human_state_callback, queue_size=1)

		rate = rospy.Rate(100) # 100hz

		self.human_state = None
		#self.human_map.update_human_traj(human_state)
		#self.tstep = 0.0
	
		#print "Press 'ENTER' to move to next timestep, anything else to quit..."
		#print "	t_step: ", self.tstep
		#print "	h_state: ", human_state
		#print "	occu_grid: ", np.around(self.human_map.occupancy_grid, decimals=2)
		while not rospy.is_shutdown():
			if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
				line = raw_input()
				
				#if line is '':
				#	human_state[0] += 1.0
				#	human_state[1] += 1.0
				#	self.tstep += 1
				#	print "	t_step: ", self.tstep
				#	print "	h_state: ", human_state
				#	print "	occu_grid: ", np.around(self.human_map.occupancy_grid, decimals=2)
				#	self.human_map.update_human_traj(human_state)
				#else:
				#	break
				
				break

			self.occupancy_pub.publish(self.grid_to_message())

			# plot start location
			self.start_pub.publish(self.state_to_marker(xy=[0,0], color="G"))

			# plot goal location
			self.goal_pub.publish(self.state_to_marker(xy=[self.human_map.goal_pos[0]/10.0, self.human_map.goal_pos[1]/10.0], color="R"))
			rate.sleep()

	def grid_to_message(self):
		"""
		Converts OccupancyGridTime structure to ROS msg
		"""
    timed_grid = OccupancyGridTime()
    timed_grid.gridarray = [None]*self.human_map.fwd_pred_tsteps

    for t in range(self.human_map.fwd_pred_tsteps):
      grid_msg = ProbabilityGrid()

		  # Set up the header.
		  grid_msg.header.stamp = rospy.Time.now()
		  grid_msg.header.frame_id = "map"

		  # .info is a nav_msgs/MapMetaData message. 
		  grid_msg.resolution = 0.1
		  grid_msg.width = self.human_map.width
		  grid_msg.height = self.human_map.height

		  # Rotated maps are not supported... 
		  origin_x=0.0 
		  origin_y=0.0 
		  grid_msg.origin = Pose(Point(origin_x, origin_y, 0), Quaternion(0, 0, 0, 1))

		  # Flatten the numpy array into a list of doubles from 0-1
		  grid_msg.data = list(self.human_map.occupancy_grids[t].reshape((self.human_map.occupancy_grids[t].size,)))

      timed_grid.gridarray[t] = grid_msg
 
		return timed_grid

	def human_state_callback(self, msg):
		"""
		Grabs the human's state from the mocap publisher
		"""

		# update the map with where the human is at the current time
		self.human_state = [msg.pose.position.x, msg.pose.position.y]

		self.human_map.update_human_traj(self.human_state)
		
		# infer the new human occupancy map at the current timestamp
		self.human_map.infer_occupancies(msg.header.stamp.secs) 

	def state_to_marker(self, xy=[0,0], color="R"):
		"""
		Converts xy position to marker type to vizualize human
		"""
		marker = Marker()
		marker.header.frame_id = "/root"

		marker.type = marker.SPHERE
		marker.action = marker.ADD
		marker.pose.orientation.w = 1
		marker.pose.position.z = 0
		marker.scale.x = 0.1
		marker.scale.y = 0.1
		marker.scale.z = 0.1
		marker.color.a = 1.0
		if color is "R":
			marker.color.r = 1.0
		elif color is "G":
			marker.color.g = 1.0
		else:
			marker.color.b = 1.0

		marker.pose.position.x = xy[1]
		marker.pose.position.y = xy[0]

		return marker

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print "ERROR: Not enough arguments. Specify height, width of world"
	else:	

		height = int(sys.argv[1])
		width = int(sys.argv[2])

		SimWorld(height, width)


