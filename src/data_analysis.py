# !/usr/bin/env python2.7
# import rospy
# import os
# import numpy as np
# import rosbag
# import sys
# import subprocess, yaml
# import sys, select, os
# import std_srvs.srv 
import time


import roslaunch
import rospy

def shutdown():
	print('peace out mofo')

rospy.init_node('tester', anonymous=True)
rospy.on_shutdown(shutdown)

uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)
launch = roslaunch.parent.ROSLaunchParent(uuid,["/home/abajcsy/GitHub/meta_fastrack/ros/src/meta_planner/launch/coffee.launch"])

launch.start()
time.sleep(4.0)
launch.shutdown()

# if __name__ == '__main__':

# 	# launch the crazyflie dood
# 	s = subprocess.Popen(['roslaunch', '/home/abajcsy/GitHub/meta_fastrack/ros/src/meta_planner/launch/coffee.launch'], stdout=subprocess.PIPE).communicate()[0]

# 	time.sleep(5.0)

# 	rospy.wait_for_service('/takeoff')
# 	try:
# 		emptyboi = rospy.ServiceProxy('takeoff', EmptySrv)
# 		emptybaby = emptyboi()
# 	except rospy.ServiceException, e:
# 		print "Service call failed: %s"%e

# 	time.sleep(1.0)

	# launch the prediction
#	p = subprocess.Popen(['roslaunch', 'crazyflie_human', 'sim.launch'], stdout=subprocess.PIPE).communicate()[0]


