#!/usr/bin/env python3

from ultralytics import YOLO
import rclpy 
import cv2
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

sensor_node_name = "sensor_kinect_node" # Nome do nó de sensor Kinect
kinect_topic_name = "kinect_topic"      # Tópico onde as imagens do Kinect são publicadas


# Bridge para converter uma mensagem do ROS em uma imagem do OpenCV
bridge = CvBridge()

class SensorKinectNode(Node): 
    def __init__(self):
        super().__init__(sensor_node_name)
        self.get_logger().info("Nó de sensor Kinect inicializado.")

        # Inicializa o YOLOv8
        self.model = YOLO('yolov8n.pt')

        # Se inscreve no tópico do Kinect para receber imagens RGB
        self.subscriber_rgb = self.create_subscription(Image, kinect_topic_name, self.callback_detection, 10)
        
        # Se inscreve no tópico de profundidade para receber imagens de profundidade
        self.subscriber_depth = self.create_subscription(Image, "/kinect/depth/image_raw", self.callback_depth, 10)
        
        self.depth_image = None  # Armazenará a última imagem de profundidade recebida

    # Callback para processar a imagem RGB e realizar a detecção de objetos
    def callback_detection(self, msg):
        self.get_logger().info("Executando callback de detecção...")

        # Converte a mensagem do ROS em uma imagem para o OpenCV (BGR)
        img = bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Aplica a detecção de objetos na imagem e guarda as informações em results
        results = self.model(img)

        # Verifica se a imagem de profundidade está disponível
        if self.depth_image is None:
            self.get_logger().warn("Imagem de profundidade não disponível ainda.")
            return

        # Extrai as bounding boxes e calcula a distância
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                c = box.cls
                class_name = self.model.names[int(c)]
                confidence = box.conf[0]
                top, left, bottom, right = int(b[1]), int(b[0]), int(b[3]), int(b[2])
                
                # Calcula o centro da bounding box
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2

                # Calcula a distância usando a imagem de profundidade
                distance = self.get_distance_at(center_x, center_y)
                if distance:
                    self.get_logger().info(f"Objeto detectado: {class_name} com confiança {confidence:.2f}. Distância: {distance:.2f} metros")

                # Desenha as bounding boxes
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Mostra o nome da classe, confiança e distância
                text = f"{class_name}: {confidence:.2f}, Distância: {distance:.2f}m"
                cv2.putText(img, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mostra a imagem com as bounding boxes e informações adicionais
        cv2.imshow("Object Detection with Distance", img)
        cv2.waitKey(1)  

    # Callback para processar a imagem de profundidade
    def callback_depth(self, msg):
        try:
            self.depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")  # A profundidade normalmente é em formato de 16 bits por canal (16UC1)
        except CvBridgeError as e:
            self.get_logger().error(f"Erro na conversão da imagem de profundidade: {e}")
    
    # Função para obter a distância a partir da imagem de profundidade
    def get_distance_at(self, x, y):
        if self.depth_image is None:
            return None
        distance = self.depth_image[y, x]  # Obtém a distância em milímetros
        return distance / 1000.0  # Converte para metros

def main(args=None):
    rclpy.init(args=args)
    node = SensorKinectNode() 

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
