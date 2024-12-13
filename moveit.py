"""This algorithm utlizes MoveIt to contorl the manipulator,
which can be used as a benchmark to evaluate the performance of PMP.
Please install/configure ROS and MoveIt for runing it"""

#!/usr/bin/env python
import numpy as np
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import math
from moveit_commander.conversions import pose_to_list
from moveit_commander import MoveGroupCommander, RobotCommander
from moveit_msgs.msg import Constraints, JointConstraint

def all_close(goal, actual, tolerance):
    """Convenience method for testing if a list of values are within a tolerance of their counterparts in another list"""
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False
    return True

# Function to set the initial joint state
def set_initial_state(move_group, initial_joint_values):
    joint_goal = move_group.get_current_joint_values()
    for i, value in enumerate(initial_joint_values):
        joint_goal[i] = value
    move_group.set_joint_value_target(joint_goal)
    move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    end_effector_pose = move_group.get_current_pose().pose
    print("Initial position:", end_effector_pose.position.x, end_effector_pose.position.y, end_effector_pose.position.z)

def moveit_joint_angle_calculator():
    # Initialize moveit_commander and rospy node
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('moveit_joint_angle_calculator', anonymous=True)

    # Instantiate RobotCommander, PlanningSceneInterface, and MoveGroupCommander
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = "manipulator"
    move_group = moveit_commander.MoveGroupCommander(group_name)

    # Replace these values with the initial joint states from PMP
    initial_joint_values = [np.radians(94.68), np.radians(-86.32), np.radians(-142.83), np.radians(-133.48), np.radians(-84.74), np.radians(86.31)]  # Example joint angles
    set_initial_state(move_group, initial_joint_values)

    # Define the target pose for the end effector
    target_pose = geometry_msgs.msg.Pose()
    target_pose.position.x = -0.07531
    target_pose.position.y = -0.59926
    target_pose.position.z = 0.50517
    target_pose.orientation.w = 1.0

    # Define path constraints
    constraints = Constraints()

    joint_constraint = JointConstraint()
    joint_constraint.joint_name = "wrist_1_joint"  # Replace with your actual joint name
    joint_constraint.position = np.radians(45)  # Fix Joint
    joint_constraint.tolerance_above = np.radians(15.0)
    joint_constraint.tolerance_below = np.radians(15.0)
    joint_constraint.weight = 1.0
    constraints.joint_constraints.append(joint_constraint)

    joint_constraint2 = JointConstraint()
    joint_constraint2.joint_name = "wrist_2_joint"  # Replace with your actual joint name
    joint_constraint2.position = np.radians(90)  # Fix Joint
    joint_constraint2.tolerance_above = np.radians(15.0)
    joint_constraint2.tolerance_below = np.radians(15.0)
    joint_constraint2.weight = 1.0
    constraints.joint_constraints.append(joint_constraint2)

    #move_group.set_path_constraints(constraints) #uncomment to add constraints

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

    end_effector_pose = move_group.get_current_pose().pose

    # Print out the joint angles in radians and degrees
    print("Calculated Joint Angles (radians):", current_joints)
    print("Calculated Joint Angles (degrees):", current_joints_degrees)
    print("Final solution:", end_effector_pose.position.x, end_effector_pose.position.y, end_effector_pose.position.z)

    # Shut down moveit_commander
    moveit_commander.roscpp_shutdown()

if __name__ == "__main__":
    try:
        moveit_joint_angle_calculator()
    except rospy.ROSInterruptException:
        pass
