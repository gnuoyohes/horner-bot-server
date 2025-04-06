from adafruit_servokit import ServoKit
import time

servos = ServoKit(channels=16)
servos.servo[0].set_pulse_width_range(500, 2500)
servos.servo[1].set_pulse_width_range(500, 2500)

while True:
    servos.servo[0].angle = 0
    servos.servo[1].angle = 90
    print("0")
    time.sleep(1)
    servos.servo[0].angle = 90
    servos.servo[1].angle = 135
    print("90")
    time.sleep(1)
    servos.servo[0].angle = 180
    servos.servo[1].angle = 180
    print("180")
    time.sleep(1)
