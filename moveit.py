#!/usr/bin/env python
"""This algorithm utlizes MoveIt to contorl the manipulator,
which can be used as a benchmark to evaluate the performance of PMP.
Please install/configure ROS and MoveIt for runing it"""

import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import math
from moveit_commander.conversions import pose_to_list

def all_close(goal, actual, tolerance):
    """Convenience method for testing if a list of values are within a tolerance of their counterparts in another list"""
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False
    return True

def moveit_joint_angle_calculator():
    # Initialize moveit_commander and rospy node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('moveit_joint_angle_calculator', anonymous=True)

    # Instantiate RobotCommander, PlanningSceneInterface, and MoveGroupCommander
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Define the target pose for the end effector
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = 0.04758
    target_pose.position.y = 0.31246
    target_pose.position.z = 0.40986
    target_pose.orientation.w = 1.0

    # Set the target pose for the move group
    move_group.set_pose_target(target_pose)

    # Plan to the new pose
    plan = move_group.go(wait=True)

    # Stop the robot after planning
    move_group.stop()

    # Clear the target pose
    move_group.clear_pose_targets()

    # Get the joint values after the movement
    current_joints = move_group.get_current_joint_values()

    # Convert joint angles to degrees
    current_joints_degrees = [math.degrees(angle) for angle in current_joints]

    # Print out the joint angles in radians and degrees
    print("Calculated Joint Angles (radians):", current_joints)
    print("Calculated Joint Angles (degrees):", current_joints_degrees)

    # Shut down moveit_commander
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    try:
        moveit_joint_angle_calculator()
    except rospy.ROSInterruptException:
        pass
