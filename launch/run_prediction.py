# Script to run all launch files and send takeoff command for software demo.

import subprocess
import signal
import sys
import time

# Process handles.
total_number_of_humans = 2
human_launch = [None]*total_number_of_humans

# Custom signal handler for clean shutdown.
def sigint_handler(sig, frame):
    print "Cleaning up."

    # Register global variables in this scope.
    global human_launch

    # Send termination signals and wait for success.
    for launch in human_launch:
        if launch is not None and launch.poll() is None:
            launch.terminate()

        if launch is not None:
            launch.wait()

    # Exit successfully.
    sys.exit(0)


# Register custom signal handler on SIGINT.
signal.signal(signal.SIGINT, sigint_handler)

for ii in range(total_number_of_humans):
    # Run one process for each human
    human_ID = "human_number:="+str(ii+1)
    #human_launch[ii] = subprocess.Popen(["roslaunch", "crazyflie_human", "simulated_human_launcher.launch", human_ID])
    human_launch[ii] = subprocess.Popen(["roslaunch", "crazyflie_human", "simulated_demo.launch", "human1:="+str(ii+1), "human1_namespace:=human"+str(ii+1)])

signal.pause()
