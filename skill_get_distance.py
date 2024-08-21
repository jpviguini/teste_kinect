#!/usr/bin/env python3
import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

kinect_topic_name = "kinect_topic" # Nome do tópico que o kinect publica as imagens

# Skill para obter a distância de um objeto a partir do sensor de profundidade do kinect
class GetDistance(Node):
    def __init__(self):
        super().__init__("set_model_client")
        self.bridge = CvBridge() # Bridge para converter mensagens do ROS (Image) em imagem para o openCV
        self.subscriber_ = self.create_subscription(Image, kinect_topic_name, self.callback_distance, 10)


    # Processa a distância do objeto
    def callback_distance(self, msg):
        self.get_logger().info("Iniciando callback de cálculo de distância...")

        # mantém a mensagem do ros e a imagem do opencv com o mesmo formato
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough') 

        # Pega a distância do objeto no centro da imagem
        height, width = depth_image.shape
        center_x, center_y = width // 2, height // 2
        distance = depth_image[center_y, center_x]

        if not (0 < distance < float('inf')): # inf é um valor máximo
            self.get_logger().warn(f'Distância inválida ou fora do alcance detectada: {distance}')
        else:
            self.get_logger().info(f'Distância ao objeto no centro: {distance} metros')


def main(args=None):
    rclpy.init(args=args)
    get_distance = GetDistance()

    # Mantém o nó rodando para processar os callbacks
    rclpy.spin(get_distance)

    get_distance.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()


