#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
from demo_programs.msg import prox_sensor, line_sensor

# Global variables
front_obstacle_detected = False
left_obstacle_detected = False
right_obstacle_detected = False

previous_move = None

line_detected_right = False
line_detected_middle = False
line_detected_left = False

# Set up publisher
twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
# Set up the Twist message
twist_cmd = Twist()

def proximity_callback(data):
	global front_obstacle_detected, left_obstacle_detected, right_obstacle_detected

	# Check if an obstacle is detected in front
	front_sensors = [0.1 < data.prox_front_left < 0.5,
			0.1 < data.prox_front < 0.5,
			0.1 < data.prox_front_right < 0.5]

	front_obstacle_detected = any(front_sensors)

	# Check if an obstacle is detected on the left 
	left_sensors = [0.1 < data.prox_front_left < 0.5, 
			0.1 < data.prox_front_left_left < 0.5]

	left_obstacle_detected = any(left_sensors)

	# Check if an obstacle is detected on the right
	right_sensors = [0.1 < data.prox_front_right < 0.5,
			0.1 < data.prox_front_right_right < 0.5]

	right_obstacle_detected = any(right_sensors)

def line_callback(data):
	global line_detected_right, line_detected_middle, line_detected_left
	
	# Update line sensor data
	line_detected_right = data.line_right
	line_detected_middle = data.line_middle
	line_detected_left = data.line_left

def move_forward():
	global twist_cmd

	# Set linear velocity to move forward
	twist_cmd.linear.x = 0.25
	twist_cmd.angular.z = 0.0
	twist_pub.publish(twist_cmd)

def turn_left():
	global twist_cmd

	# Rotate robot when obstacle is detected on the right
	twist_cmd.linear.x = 0.0
	twist_cmd.angular.z = -0.8
	twist_pub.publish(twist_cmd)

	# Wait for the robot to rotate
	rospy.sleep(1.0)
	
	# Stop the robot
	stop_robot()	

def turn_right():
	global twist_cmd

	# Rotate robot when obstacle is detected on the left
	twist_cmd.linear.x = 0.0
	twist_cmd.angular.z = 0.8
	twist_pub.publish(twist_cmd)

	# Wait for the robot to rotate
	rospy.sleep(1.0)
	
	# Stop the robot
	stop_robot()
	
def stop_robot():
	global twist_cmd

	# Stop the robot
	twist_cmd.linear.x = 0.0
	twist_cmd.angular.z = 0.0
	twist_pub.publish(twist_cmd)

def main():
	print('robot is running...')
	global front_obstacle_detected, left_obstacle_detected, right_obstacle_detected
	global line_detected_right, line_detected_middle, line_detected_left
	global previous_move
	
	rospy.init_node('robot_controller', anonymous=True)

	# Set up subscribers
	prox_sensor_sub = rospy.Subscriber('/cop/prox_sensors', prox_sensor, proximity_callback)
	line_sub = rospy.Subscriber('/cop/line_sensors', line_sensor, line_callback)

	# Set up publisher
	twist_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
	
	# Set up the Twist message
	twist_cmd = Twist()

	# Set the rate at which to publish commands
	rate = rospy.Rate(10) # 10Hz

	while not rospy.is_shutdown():
		if not front_obstacle_detected:
			if previous_move == 'left':
				move_forward()
				rospy.sleep(1.5)
				turn_right() # readjust to upright position after left turn
				previous_move = 'forward'
			elif previous_move == 'right':
				move_forward()
				rospy.sleep(1.5)
				turn_left() # readjust to upright position after right turn
				previous_move = 'forward'
			else:
				move_forward()
				previous_move = 'forward'
		elif left_obstacle_detected:
			turn_right() # turn right if left obstacle detected
			previous_move = 'right'
		elif right_obstacle_detected:
			turn_left() # turn left if right obstacle detected
			previous_move = 'left'
		elif line_detected_middle or line_detected_left or line_detected_right:
			stop_robot()
			print('robot reached the end of the platform')
		else:
			stop_robot()
	
		rate.sleep()	

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass

