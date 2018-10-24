#!/usr/bin/env python  
import rospy
import math
import tf
import geometry_msgs.msg
import sys

if __name__ == '__main__':
	rospy.init_node('human_state_estimator')

	human_num = str(rospy.get_param('human_number'))

	listener = tf.TransformListener()

	pub_topic = '/human_pose'+human_num
	pose_pub = rospy.Publisher(pub_topic, geometry_msgs.msg.PoseStamped, queue_size=1)

	rate = rospy.Rate(100.0)
	while not rospy.is_shutdown():
		try:
			hat_tf = '/vicon/hat/hat'#'/Human'+human_num+'/base_link'
			(trans,rot) = listener.lookupTransform('/world', hat_tf, rospy.Time(0))
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
