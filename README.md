# crazyflie_human

ROS and crazyflie-integrated prediction of human motion. 

# Choosing where your human data comes from

## Simulated human data
Open up ```multi_sim.launch``` and uncomment the following line:
```
<node name="simulate_human1" pkg="crazyflie_human" type="sim_human.py" args="1" output="screen"/> 
```
and make sure the optitrack simulator is commented out:
```
<!-- <node name="human_state_estimator1" pkg="crazyflie_human" type="human_state_estimator.py" args="1" output="screen"/> -->
```

## Real (Optitrack) human data
Open up ```multi_sim.launch``` and uncomment the following line:
```
<node name="human_state_estimator1" pkg="crazyflie_human" type="human_state_estimator.py" args="1" output="screen"/>
```
and make sure the simulated human data is commented out:
```
<!-- <node name="simulate_human1" pkg="crazyflie_human" type="sim_human.py" args="1" output="screen"/>  -->
```

# Running the code
```
roslaunch crazyflie_human multi_sim.launch beta:=adaptive
```
Arguments:
* beta:=[irrational,rational,adaptive] changes the observation model of human actions

# Changing the number of humans
Two files are important when changing the number of humans:
* ```pedestrian_pred.yaml``` (in ```/config```)
Edit the number of humans by changing the parameter:
```
num_humans: N
```
where N is the number of humans in your environment. For each human (numbered 1-N) make sure that they each have specified starts and goals:
```
humanN_real_start: [1.0,-1.0]
humanN_real_goals: [[1.0, 2.0], [2.0, 1.0]]
```

* ```multi_sim.launch``` (in ```/launch```)
Add in a human-prediction node for each human by adding in a line for each human:
```
<node name="human_prediction1" pkg="crazyflie_human" type="human_pred.py" args="1" output="screen"/> 
<node name="human_prediction2" pkg="crazyflie_human" type="human_pred.py" args="2" output="screen"/> 
....
<node name="human_predictionN" pkg="crazyflie_human" type="human_pred.py" args="N" output="screen"/> 
```
If running with simulated human data, add in a simulated human node for each human:
```
<node name="simulate_human1" pkg="crazyflie_human" type="sim_human.py" args="1" output="screen"/> 
<node name="simulate_human2" pkg="crazyflie_human" type="sim_human.py" args="2" output="screen"/> 
...
<node name="simulate_humanN" pkg="crazyflie_human" type="sim_human.py" args="N" output="screen"/> 
```
If running with real (optitrack) human data, add in a human state estimator for each human:
```
<node name="human_state_estimator1" pkg="crazyflie_human" type="human_state_estimator.py" args="1" output="screen"/> 
<node name="human_state_estimator2" pkg="crazyflie_human" type="human_state_estimator.py" args="2" output="screen"/> 
...
<node name="human_state_estimatorN" pkg="crazyflie_human" type="human_state_estimator.py" args="N" output="screen"/> 
```

# Visualization
To see the visualizations in action, run RVIZ:
```
roslaunch crazyflie_human rviz.launch
```
