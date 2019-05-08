""" This file defines utilities for the ROS agents. """
import numpy as np

from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy
# from gps_agent_pkg.proto.python_compiled.ControllerParams_pb2 import ControllerParams
# from gps_agent_pkg.proto.python_compiled.LinGaussParams_pb2 import LinGaussParams
# from gps_agent_pkg.proto.python_compiled.TfParams_pb2 import TfParams
# from gps_agent_pkg.proto.python_compiled.CaffeParams_pb2 import CaffeParams
# from gps_agent_pkg.proto.python_compiled.TfActionCommand_pb2 import TfActionCommand

from gps.sample.sample import Sample
from gps.proto.gps_pb2 import LIN_GAUSS_CONTROLLER, CAFFE_CONTROLLER, TF_CONTROLLER, END_EFFECTOR_POINTS,\
    END_EFFECTOR_ROTATIONS, JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINT_JACOBIANS
import logging
import time
import threading

import zmq

LOGGER = logging.getLogger(__name__)

def msg_to_sample(ros_msg, agent):
    """
    Convert a SampleResult ROS message into a Sample Python object.
    """
    sample = Sample(agent)

    velocity = np.array(ros_msg.velocity).reshape(7)
    joint_angles = np.array(ros_msg.joint_angles).reshape(7)
    ee_pos = np.array(ros_msg.ee_pos).reshape(9)
    ee_jacobians = np.array(ros_msg.ee_points_jacobian, order="F").reshape(9, 7)
#    ee_orient = np.array(ros_msg.ee_orient).reshape(3)

    sample.set(JOINT_VELOCITIES, velocity)
    sample.set(JOINT_ANGLES, joint_angles)
    sample.set(END_EFFECTOR_POINTS, ee_pos)
    sample.set(END_EFFECTOR_POINT_JACOBIANS, ee_jacobians)
    #sample.set(END_EFFECTOR_ROTATIONS, ee_orient)

    return sample

def tf_obs_msg_to_numpy(obs_message):
    data = np.array(obs_message.data)
    data = data.reshape(obs_message.shape)
    return data


def image_msg_to_cv(image_message):

    width = image_message.width
    height = image_message.height
    depth = image_message.height

    if image_message.isInt:
        data_type = np.int8
    else:
        data_type = np.float32

    image = np.array(image_message.data, dtype=data_type)
    image = image.reshape(height, width, depth)  # cv uses HWC order
    return image


class TimeoutException(Exception):
    """ Exception thrown on timeouts. """
    def __init__(self, sec_waited):
        Exception.__init__(self, "Timed out after %f seconds", sec_waited)


class ServiceEmulator(object):
    """
    Emulates a ROS service (request-response) from a
    publisher-subscriber pair.
    Args:
        pub_topic: Publisher topic.
        pub_type: Publisher message type.
        sub_topic: Subscriber topic.
        sub_type: Subscriber message type.
    """
    def __init__(self, pub_topic, pub_type, sub_topic, sub_type, pub_port, sub_ports):

        self._pub = PublisherEmulator(pub_topic, pub_type, pub_port)

        self._sub = SubscriberEmulator(sub_topic, sub_type, self._callback, sub_ports)

        self._waiting = False
        self._subscriber_msg = None

    def _callback(self, message):
        if self._waiting:
            #print("Saved received message")
            self._subscriber_msg = message
            self._waiting = False


    def publish(self, pub_msg):
        """ Publish a message without waiting for response. """
        self._pub.publish(pub_msg)

    def publish_and_wait(self, pub_msg, timeout=5.0, poll_delay=0.01,
                         check_id=False):
        """
        Publish a message and wait for the response.
        Args:
            pub_msg: Message to publish.
            timeout: Timeout in seconds.
            poll_delay: Speed of polling for the subscriber message in
                seconds.
            check_id: If enabled, will only return messages with a
                matching id field.
        Returns:
            sub_msg: Subscriber message.
        """
        if check_id:  # This is not yet implemented in C++.
            raise NotImplementedError()

        self._waiting = True
        self.publish(pub_msg)

        time_waited = 0
        while self._waiting:
            time.sleep(poll_delay)
            time_waited += 0.01
            if time_waited > timeout:
                raise TimeoutException(time_waited)
        return self._subscriber_msg


class PublisherEmulator:

    def __init__(self, pub_topic, pub_type, port):
        self._pub_topic = pub_topic
        self._pub_type = pub_type
        context = zmq.Context()
        self._pub = context.socket(zmq.PUB)
        self._pub.bind("tcp://127.0.0.1:%s" % port)
        print("bind pub to file tcp://127.0.0.1:%s" % port)

    def publish(self, message):
        assert type(message) == self._pub_type
        message_string = message.SerializeToString()
        self._pub.send(self._pub_topic.encode(encoding='ASCII') + " ".encode(encoding='ASCII') + message_string)
        # self._pub.send(self._pub_topic.encode(encoding='UTF-8') + " " + message_string)
        #print("published to %s" % self._pub_topic)


class SubscriberEmulator:

    def __init__(self, sub_topic, sub_type, callback, ports):
        self._sub_topic = sub_topic
        self._sub_type = sub_type
        self._callback = callback

        context = zmq.Context()
        self._sub = context.socket(zmq.SUB)
        for port in ports:
            self._sub.connect("tcp://127.0.0.1:%s" % port)
            print("connect sub to file tcp://127.0.0.1:%s" % port)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, sub_topic)
        # self._sub.setsockopt(zmq.SUBSCRIBE, sub_topic.encode(encoding='UTF-8'))
        print("subscribed to topic %s" % sub_topic)

        self._sub_message = sub_type()

        self._start_callback_thread()

    def _start_callback_thread(self):
        thread = threading.Thread(target=self._callback_thread)
        thread.setDaemon(True)
        thread.start()

    def _callback_thread(self):
        #  sub_message_string = ""
        while True:
            sub_message_string = self._sub.recv()
            #print("received message in topic %s" % self._sub_topic)
            sub_message_string = sub_message_string[len(self._sub_topic)+1:]
            self._sub_message.ParseFromString(sub_message_string)
            self._callback(self._sub_message)


