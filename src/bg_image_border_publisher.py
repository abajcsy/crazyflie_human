#!/usr/bin/env python  
import rospy
import os
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

def pixel_to_point(x, y, width, height, scale_x, scale_y, scale_z):
	point = Point()
	point.x = (x - (width/2.0))*scale_x
	point.y = (y - height/2.0)*scale_y
	point.z = 0.0*scale_z
	return point

def pixel_to_color(x, y, image):
	color = ColorRGBA()
	color.r = image[x][y][0]
	color.g = image[x][y][1]
	color.b = image[x][y][2]
	color.a = 1.0
	return color

if __name__ == '__main__':
	rospy.init_node('bg_image_publisher')

	# Setup the image publisher.
	pub_topic = '/bg_image'
	pose_pub = rospy.Publisher(pub_topic, Marker, queue_size=1)

	# Read the background image.
	full_path = os.path.dirname(os.path.realpath(__file__))
	print full_path
	image = mpimg.imread(full_path+"/../config/testbed_downsized.png")
	height = 300
	width = 233

	# Construct the Image message
	image_msg = Marker()
	image_msg.header.frame_id = "/world"
	image_msg.header.stamp = rospy.Time.now()
	image_msg.ns = "image_cube_list"
	image_msg.id = 0
	image_msg.type = image_msg.CUBE_LIST
	image_msg.pose.orientation.x = 0.0
	image_msg.pose.orientation.y = 0.0
	image_msg.pose.orientation.z = -0.707
	image_msg.pose.orientation.w = 0.707
	image_msg.scale.x = 0.1
	image_msg.scale.y = 0.1
	image_msg.scale.z = 0.01

	# Create the top borders of the the image.
	for x in range(width):
		for y in range(height):
			if x == 0 or x == width-1:
				# Create point at pixel.
				point = pixel_to_point(x, y, width, height, 
					image_msg.scale.x, image_msg.scale.y, image_msg.scale.z)
				image_msg.points.append(point)
				# Create color at pixel.
				color = pixel_to_color(x, y, image)
				image_msg.colors.append(color)
			elif y == 0 or y == height-1:
				# Create point at pixel.
				point = pixel_to_point(x, y, width, height, 
					image_msg.scale.x, image_msg.scale.y, image_msg.scale.z)
				image_msg.points.append(point)
				# Create color at pixel.
				color = pixel_to_color(x, y, image)
				image_msg.colors.append(color)

	# Publish the background image at 10 Hz
	rate = rospy.Rate(10.0)
	while not rospy.is_shutdown():
		pose_pub.publish(image_msg)
		rate.sleep()
