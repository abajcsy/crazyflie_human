# crazyflie_human

ROS and crazyflie-integrated prediction of human motion. 

# Dependencies
This repository depends on the ```pedestrian_prediction``` repository which may be found [here](https://github.com/sirspinach/pedestrian_prediction) and the ```crazyflie_clean``` package which may be found [here](https://github.com/HJReachability/crazyflie_clean).

# Usage
This repository needs to be placed inside of a ROS workspace. Clone it inside of the ```src/``` folder in your workspace:
```
cd catkin_ws/src/
git clone https://github.com/abajcsy/crazyflie_human
```

## Simulated Human Motion
To run predictions with simulated human data, type:
```
roslaunch crazyflie_human simulated_demo.launch
```

### Changing the Simulated Human Model
There are many ways to simulate human pedestrian data. This repository supports:
* ```linear_human.py``` -- Human motion is simply a straight line between goals. Ignores obstacles. 
* ```potential_field_human.py``` -- Human motion follows attractive-repuslive forces towards goals and away from obstacles.

Once you know which simulation you want to use, open ```simulated_human_launcher.launch```. Change the simulation node to point to the appropriate python file. Now just rerun the simulated demo!

## Real (Optitrack) Human Motion
To run predictions with real human data, type:
```
roslaunch crazyflie_human real_demo.launch
```
## Changing the Number of Humans
Open up either ```simulated_demo.launch``` or ```real_demo.launch```. To add another human into the launch file, simply add a new namespace for this human, following the convention:
```
<arg name="humanN_namespace" default="humanN" />
```
then add a new human ID:
```
<arg name="humanN" default="N" />
```
then add a new human prediction node:
```
<group ns="$(arg humanN_namespace)">
	<include file="$(find crazyflie_human)/launch/real_human_launcher.launch">
		<arg name="human_number" value="$(arg humanN)" />
		<arg name="beta" value="$(arg prediction_model)" />
	</include>
</group>
```

## Changing the Human Start and Goals
Open ```/config/pedestrian_pred.yaml```. For each human (numbered 1-N) make sure that they each have specified starts and goals:
```
humanN_real_start: [1.0,-1.0]
humanN_real_goals: [[1.0, 2.0], [2.0, 1.0]]
```
These are used for predicting future states of the human, and are used by the human simulator to move the simulated human through the environment. 

# Visualization
To see the visualizations of the human, the predictions, and the modeled goals, run RVIZ:
```
roslaunch crazyflie_human rviz.launch
```
