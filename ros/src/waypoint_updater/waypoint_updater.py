#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped , TwistStamped
from styx_msgs.msg import Lane, Waypoint , TrafficLightArray , TrafficLight
from std_msgs.msg import Int32, Float32
import numpy as np
import math
import tf
from threading import Thread, Lock
from copy import deepcopy
'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
USE_GT_TRAFFIC = True
TAFFIC_ACTIVATED = False

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        # Stub ground truth information of Traffic light status
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.gt_traffic_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        self.mutex = Lock()

        self.base_waypoints = None
        self.current_pose = None
        self.current_velocity = None
        self.next_waypoint_id = None
        self.traffic_light_waypoint_id = None
        self.obstacle_waypoint_id = None

        self.max_velocity_mps = rospy.get_param("/waypoint_loader/velocity") * 0.277778 # transform km/h to m/s
        self.max_deceleration = np.fabs(rospy.get_param("/dbw_node/decel_limit"))
        self.tl_stop_buffer = 5.0  # safe buffer distance before traffic light in meters
        self.gt_tl_waypoint_id = None


        # Loop Event for updating final_waypoints
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.waypoints_updater()
            rate.sleep()

    def waypoints_updater(self):
        if self.base_waypoints and self.current_pose:
            self.next_waypoint_id = self.get_next_waypoint(self.current_pose, self.base_waypoints)
            if self.next_waypoint_id:
                rospy.loginfo("next waypoint id = %d" , self.next_waypoint_id)
            if self.gt_tl_waypoint_id:
                rospy.loginfo("TL waypoint id = %d with state = %d", self.gt_tl_waypoint_id[0], self.gt_tl_waypoint_id[1])
            final_waypoints = self.get_final_waypoints(self.base_waypoints.waypoints, self.next_waypoint_id,
                                                       self.next_waypoint_id+LOOKAHEAD_WPS)

            lane = Lane()
            lane.header.stamp = rospy.Time().now()
            lane.header.frame_id = '/world'
            lane.waypoints = final_waypoints
            self.final_waypoints_pub.publish(lane)

    def nearest_waypoint(self,x,y,waypoints_list):
        min_dist = float('inf')
        nearest_point_id = -1
        for id , waypoint in enumerate(waypoints_list.waypoints):
            waypoint_x = waypoint.pose.pose.position.x
            waypoint_y = waypoint.pose.pose.position.y

            dist = np.sqrt((waypoint_x-x)**2 + (waypoint_y-y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_point_id = id

        return nearest_point_id

    def get_next_waypoint(self,current_pose,waypoints_list):

        current_pose_x = current_pose.pose.position.x
        current_pose_y = current_pose.pose.position.y
        nearest_point_id = self.nearest_waypoint(current_pose_x,current_pose_y,waypoints_list)
        nearest_point_x = waypoints_list.waypoints[nearest_point_id].pose.pose.position.x
        nearest_point_y = waypoints_list.waypoints[nearest_point_id].pose.pose.position.y
        heading = np.arctan2((nearest_point_y-current_pose_y) , (nearest_point_x-current_pose_x))

        x = current_pose.pose.orientation.x
        y = current_pose.pose.orientation.y
        z = current_pose.pose.orientation.z
        w = current_pose.pose.orientation.w
        euler_angles_xyz = tf.transformations.euler_from_quaternion([x, y, z, w])
        theta = euler_angles_xyz[-1]
        angle = math.fabs(theta - heading)
        if angle > math.pi / 4.0:
            nearest_point_id += 1

        return nearest_point_id

    def get_final_waypoints(self, waypoints, start_wp, end_wp):

        final_waypoints = []
        for i in range(start_wp, end_wp):
            index = i % len(waypoints)
            wp = Waypoint()
            wp.pose.pose.position.x = waypoints[index].pose.pose.position.x
            wp.pose.pose.position.y = waypoints[index].pose.pose.position.y
            wp.pose.pose.position.z = waypoints[index].pose.pose.position.z
            wp.pose.pose.orientation = waypoints[index].pose.pose.orientation
            wp.twist.twist.linear.x = waypoints[index].twist.twist.linear.x
            final_waypoints.append(wp)

        return final_waypoints

    def gt_traffic_cb(self,msg):

        # process ground truth information to get nearest Traffic light and its corrosponding waypoint id
        # self.gt_tl_waypoint_id = None

        trafficlight_array = msg.lights

        # rospy.loginfo("state = %d", np.uint8(trafficlight_array[0].state))

        if self.base_waypoints and self.current_pose is not None: #and not trafficlight_array[0].state:
            current_pose = deepcopy(self.current_pose)
            current_pose_x = current_pose.pose.position.x
            current_pose_y = current_pose.pose.position.y

            min_dist = float('inf')

            nearest_point_id = None
            for id in range(len(trafficlight_array)):
                tl_x = trafficlight_array[id].pose.pose.position.x
                tl_y = trafficlight_array[id].pose.pose.position.y

                dist = np.sqrt((current_pose_x - tl_x) ** 2 + (current_pose_y - tl_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_point_id = id

            if nearest_point_id is not None:
                self.gt_tl_waypoint_id = (self.nearest_waypoint(
                    trafficlight_array[nearest_point_id].pose.pose.position.x,
                    trafficlight_array[nearest_point_id].pose.pose.position.y,
                    self.base_waypoints),trafficlight_array[nearest_point_id].state)


    def velocity_cb(self,msg):
        self.current_velocity = msg.twist.linear.x
        # rospy.loginfo("current_velocity = %s", str(self.current_velocity))

    def pose_cb(self, msg):

        self.current_pose = msg

    def waypoints_cb(self, waypoints):

        self.base_waypoints = waypoints

    def traffic_cb(self, msg):

        self.traffic_light_waypoint_id = msg.data

    def obstacle_cb(self, msg):
        self.obstacle_waypoint_id = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
