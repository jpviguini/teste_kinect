#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Publisher para simular as imagens que o kinect vai capturar
class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.get_logger().info("Iniciando webcam...")
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, 'kinect_topic', 10)
        self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            raise RuntimeError("Failed to open camera")

        self.timer = self.create_timer(1.0, self.publish_frame) 

    def publish_frame(self):
        ret, frame = self.capture.read()
        if ret:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher.publish(img_msg)
        else:
            self.get_logger().warn("Falha ao capturar o frame")

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
