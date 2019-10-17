"""Utilities for ROS agents."""
import logging
import time
import threading

import zmq

LOGGER = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception thrown on timeouts."""

    def __init__(self, sec_waited):
        """Initializes the exception.

        Args:
            sec_waited: Duration in seconds after which the timeout is raised.

        """
        Exception.__init__(self, "Timed out after %f seconds", sec_waited)


class ServiceEmulator:
    """Emulates a ROS service (request-response) from a publisher-subscriber pair."""

    def __init__(self, pub_topic, pub_type, sub_topic, sub_type, pub_url, sub_url):
        """Initializes the service emulator.

        Args:
            pub_topic: Publisher topic.
            pub_type: Publisher message type.
            sub_topic: Subscriber topic.
            sub_type: Subscriber message type.

        """
        self._pub = PublisherEmulator(pub_topic, pub_type, pub_url)

        self._sub = SubscriberEmulator(sub_topic, sub_type, self._callback, sub_url)

        self._waiting = False
        self._subscriber_msg = None

    def _callback(self, message):
        if self._waiting:
            self._subscriber_msg = message
            self._waiting = False

    def publish(self, pub_msg):
        """Publishes a message without waiting for a response.

        Args:
            pub_msg: Message to publish.

        """
        self._pub.publish(pub_msg)

    def publish_and_wait(self, pub_msg, timeout=5.0, poll_delay=0.01, check_id=False):
        """Publishes a message and waits for the response.

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
    """Emulates a ROS publisher."""

    def __init__(self, pub_topic, pub_type, url):
        """Initializes the publisher.

        Args:
            pub_topic: Publisher topic.
            pub_type: Publisher message type.
            url: Server url.

        """
        self._pub_topic = pub_topic
        self._pub_type = pub_type
        context = zmq.Context()
        self._pub = context.socket(zmq.PUB)
        self._pub.bind(url)

    def publish(self, message):
        """Publishes a message.

        Args:
            pub_msg: Message to publish.

        """
        assert type(message) == self._pub_type
        message_string = message.SerializeToString()
        self._pub.send(self._pub_topic.encode(encoding='ASCII') + " ".encode(encoding='ASCII') + message_string)


class SubscriberEmulator:
    """Emulates a ROS suscriber."""

    def __init__(self, sub_topic, sub_type, callback, url):
        """Initializes the suscriber.

        Args:
            sub_topic: Subscriber topic.
            sub_type: Subscriber message type.
            callback: Callback to notify on received messages.
            url: Server url.

        """
        self._sub_topic = sub_topic
        self._sub_type = sub_type
        self._callback = callback

        context = zmq.Context()
        self._sub = context.socket(zmq.SUB)
        self._sub.connect(url)

        self._sub.setsockopt_string(zmq.SUBSCRIBE, sub_topic)

        self._sub_message = sub_type()

        self._start_callback_thread()

    def _start_callback_thread(self):
        thread = threading.Thread(target=self._callback_thread)
        thread.setDaemon(True)
        thread.start()

    def _callback_thread(self):
        while True:
            sub_message_string = self._sub.recv()
            sub_message_string = sub_message_string[len(self._sub_topic) + 1:]
            self._sub_message.ParseFromString(sub_message_string)
            self._callback(self._sub_message)
