#!/usr/bin/env python3
from ultralytics import YOLO
import rclpy 
import cv2
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

sensor_node_name = "sensor_kinect_node" #sensor_kinect_node
kinect_topic_name = "kinect_topic"      # kinect_topic


# Bridge para converter uma mensagem do ROS em uma imagem do OpenCV
bridge = CvBridge()


# Subscriber para pegar as imagens do kinect e fazer a detecção de objetos em cada uma
class SensorKinectNode(Node): 
    def __init__(self):
        super().__init__(sensor_node_name)
        self.get_logger().info("Oi sensor kinect")

        # Inicializa o YOLOv8
        self.model = YOLO('yolov8n.pt')

        # Se inscreve no tópico do kinect pra pegar as mensagens publicadas
        self.subscriber_ = self.create_subscription(Image, kinect_topic_name, self.callback_detection, 10)



    # Callback pra processar a mensagem (imagem) recebida
    def callback_detection(self, msg):
        print("Executando callback de detecção...")

        # Converte a mensagem do ROS em uma imagem para o OpenCV (blue//green//red//alpha//8 bits)
        img = bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Aplica a detecção de objetos na imagem e guarda as informações em results
        results = self.model(img)

        # Extrai as bounding boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0].to('cpu').detach().numpy().copy()
                c = box.cls
                class_name = self.model.names[int(c)]
                confidence = box.conf[0]
                top, left, bottom, right = int(b[1]), int(b[0]), int(b[3]), int(b[2])
                
                # Desenha as bounding boxes
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Mostra o nome da classe e a confiança
                text = f"{class_name}: {confidence:.2f}"
                cv2.putText(img, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Mostra a imagem com as bounding boxes
        cv2.imshow("Object Detection", img)
        cv2.waitKey(1)  
    

def main(args=None):
    rclpy.init(args=args)
    node = SensorKinectNode() 

    # Esse callback é para pegar os resultados da detecção e imprimir na tela
    def callback_wrapper(msg):
        inference_results = node.callback_detection(msg)
        print(inference_results) # Imprime o resultado da detecção

    # Mesma subscription, mas agora chamando o callback_wrapper pra printar os resultados da detecção
    node.create_subscription(Image, kinect_topic_name, callback_wrapper, 10)

    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

