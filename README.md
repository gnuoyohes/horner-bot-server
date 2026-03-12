# Horner Bot

A server running on a raspberry pi that controls a laser to play with Horner (my cat). The laser can be controlled manually (through a simple web interface that includes a live video feed from the rpi) or automatically. There are two modes of motion detection that calculate the cat's location and speed in real time: using MOG2 Background Subtraction, or using the YOLO object detection model on a Hailo AI accelerator.

Built with Flask and socket.io.


https://github.com/user-attachments/assets/98e2e49a-f407-4e0f-993f-ef3cddc54730
