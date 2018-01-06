#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, TrafficLightArray , TrafficLight
from std_msgs.msg import Int32
import numpy as np
from threading import Thread, Lock
from copy import deepcopy

class GT_TL_Pub(object):
    def __init__(self):
        rospy.init_node('gt_TL_Publisher')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.gt_traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.gt_TL_pub = rospy.Publisher('traffic_waypoint', Int32, queue_size=1)

        self.mutex = Lock()
        self.base_waypoints = None
        self.current_pose = None
        self.next_waypoint_id = None
        self.traffic_light_waypoint_id = None
        self.gt_tl_waypoint_id = -1
        self.t_0 = rospy.get_time()

        # Loop Event for updating final_waypoints
        rate = rospy.Rate(40)
        while not rospy.is_shutdown():
            self.mutex.acquire()
            self.publish_gt_TL_waypoint()
            self.mutex.release()
            rate.sleep()

    def publish_gt_TL_waypoint(self):
        if self.gt_tl_waypoint_id is not None:
            self.gt_TL_pub.publish(data=self.gt_tl_waypoint_id)
            # rospy.loginfo("tl waypoint id = %d", self.gt_tl_waypoint_id)

    def nearest_waypoint(self,x,y,waypoints_list):
        min_dist = float('inf')
        nearest_point_id = -1
        for id , waypoint in enumerate(waypoints_list.waypoints):
            waypoint_x = waypoint.pose.pose.position.x
            waypoint_y = waypoint.pose.pose.position.y

            dist = (waypoint_x-x)**2 + (waypoint_y-y)**2
            if dist < min_dist:
                min_dist = dist
                nearest_point_id = id

        return nearest_point_id


    def gt_traffic_cb(self,msg):
        # t_0 = rospy.get_time()
        # self.mutex.acquire()

        # process ground truth information to get nearest Traffic light and its corrosponding waypoint id
        self.gt_tl_waypoint_id = -1

        trafficlight_array = msg.lights

        # rospy.loginfo("state = {}".format(np.uint8(trafficlight_array[0].state)))

        if self.base_waypoints is not None and self.current_pose is not None: #and not trafficlight_array[0].state:
            current_pose_x = self.current_pose.pose.position.x
            current_pose_y = self.current_pose.pose.position.y

            min_dist = float('inf')

            nearest_point_id = -1
            for id in range(len(trafficlight_array)):
                tl_x = trafficlight_array[id].pose.pose.position.x
                tl_y = trafficlight_array[id].pose.pose.position.y

                dist = (current_pose_x - tl_x) ** 2 + (current_pose_y - tl_y) ** 2

                if dist < min_dist:
                    min_dist = dist
                    nearest_point_id = id

            if nearest_point_id != -1 and not np.uint8(trafficlight_array[0].state):
                self.gt_tl_waypoint_id = self.nearest_waypoint(
                    trafficlight_array[nearest_point_id].pose.pose.position.x,
                    trafficlight_array[nearest_point_id].pose.pose.position.y,
                    self.base_waypoints)
            elif np.uint8(trafficlight_array[0].state):
                self.gt_tl_waypoint_id = -1

        # self.mutex.release()
        # rospy.loginfo("processig time = {}".format(t_0 - rospy.get_time()))

    def pose_cb(self, msg):

        self.current_pose = msg

    def waypoints_cb(self, waypoints):

        self.base_waypoints = waypoints

if __name__ == '__main__':
    try:
        GT_TL_Pub()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start GT_TL_Pub node.')