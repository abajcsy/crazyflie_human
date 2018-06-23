#!/usr/bin/env python  
import rospy
import math
import tf
import geometry_msgs.msg
import sys

if __name__ == '__main__':
	if len(sys.argv) < 2:
		human_num = "1"
		print "human_state_estimator: no human_num arg specified. Setting human_num = 1."
	else:
		human_num = sys.argv[1]

	rospy.init_node('human_state_estimator')

	listener = tf.TransformListener()

	# TODO MAKE THIS FOR MULTIPLE HUMANS
	pose_pub = rospy.Publisher('/human_pose'+human_num, geometry_msgs.msg.PoseStamped, queue_size=1)

	rate = rospy.Rate(100.0)
	while not rospy.is_shutdown():
		try:
			(trans,rot) = listener.lookupTransform('/world', '/vicon/hat/hat'+human_num, rospy.Time(0))
		except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
			continue

		human_pose = geometry_msgs.msg.PoseStamped()
		human_pose.header.frame_id="/world"
		human_pose.header.stamp = rospy.Time.now()
		
		human_pose.pose.position.x = trans[0]
		human_pose.pose.position.y = trans[1]
		human_pose.pose.position.z = trans[2]

		pose_pub.publish(human_pose)
	
		rate.sleep()
