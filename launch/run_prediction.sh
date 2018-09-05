#!/bin/bash
# Basic range in for loop


for value in {1..2}
do
cmd='roslaunch crazyflie_human sim.launch human_number:='	
echo $cmd$value
$cmd$value &
done