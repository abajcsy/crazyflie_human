#!/usr/bin/env python  
import rospy
import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import sys
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np

"""
Displays an image pixel by pixel by creating a CubeList.
"""

if __name__ == '__main__':
	rospy.init_node('world_publisher')

	# Setup the image publisher.
	pub_topic = '/world_marker'
	pose_pub = rospy.Publisher(pub_topic, Marker, queue_size=1)

	# get real-world measurements of experimental space
	#lower = [-15, -11.5, 0] 
	#upper = [15, 11.7, 2.0] 
	lower = [-4.0, -1.04, 0] 
	upper = [3.66, 2.62, 2.0] 
	height = upper[1] - lower[1]
	width = upper[0] - lower[0]

	# Construct the world message
	marker = Marker()
	marker.header.frame_id = "/world"

	marker.type = marker.CUBE
	marker.action = marker.ADD
	marker.pose.orientation.w = 1
	marker.pose.position.z = 1.0
	marker.scale.x = width
	marker.scale.y = height
	marker.scale.z = 2.0
	marker.color.a = 0.3		
	marker.color.r = 0.1
	marker.color.g = 0.0
	marker.color.b = 0.9

	marker.pose.position.x = upper[0] - width/2.0 
	marker.pose.position.y = upper[1] - height/2.0
	marker.pose.position.z = 1.0

	# Publish the background image at 10 Hz
	rate = rospy.Rate(10.0)
	while not rospy.is_shutdown():
		pose_pub.publish(marker)
		rate.sleep()
