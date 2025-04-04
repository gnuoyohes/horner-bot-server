import math
from gpiozero import LED, AngularServo
from threading import Lock
import time

class Laser:
    def __init__(self, height, width, depth, gpio_diode, gpio_s1, gpio_s2, fps, socket):
        self.h = height
        self.w = width
        self.d = depth
        # self.diode = LED(gpio_diode)
        # self.s1 = AngularServo(gpio_s1, min_angle=-90, max_angle=90)
        # self.s2 = AngularServo(gpio_s2, min_angle=-90, max_angle=90)
        self.fps = fps
        self._obj_coords = [0.5, 0.5] # XY coordinates of the object in 2D space, in ([0, 1], [0, 1])
        self._laser_coords = [0.5, 0.5] # XY coordinates of the object in 2D space, in ([0, 1], [0, 1])
        self._socket = socket
        self._lock = Lock()
        self._on = False
        self._manual_mode = False
        self._running_thread = False
    
    # computes new coordinates of laser based on detected object location
    @staticmethod
    def _compute_coords(x, y):
        return x, y
    
    # converts point (x, y) in the 2D coordinate plane to angles (a, b) of the servos
    def _xy_to_ab(self, x, y):
        a = math.atan((x*self.w - self.w/2) / (self.w - y*self.w + self.d))
        b = math.acos(self.h / (math.sqrt((self.w - y*self.w + self.d) ** 2 + (x*self.w - self.w/2) ** 2) + self.h ** 2))
        return a, b

    def on(self):
        with self._lock:
            self._on = True
        # self.diode.on()
    
    def off(self):
        with self._lock:
            self._on = False
        # self.diode.off()

    def manual_on(self):
        with self._lock:
            self._manual_mode = True

    def manual_off(self):
        with self._lock:
            self._manual_mode = False

    def set_obj_coords(self, new_coords):
        with self._lock:
            self._obj_coords = new_coords

    def set_laser_coords(self, new_coords):
        with self._lock:
            self._laser_coords = new_coords

    def get_laser_coords(self):
        with self._lock:
            return self._laser_coords
    
    def moveLaser(self):
        # implement
        
        if self._manual_mode:
            with self._lock:
                x, y = self._laser_coords
        else:
            with self._lock:
                obj_x, obj_y = self._obj_coords
            x, y = Laser._compute_coords(obj_x, obj_y)
            self.set_laser_coords([x, y])
        a, b = self._xy_to_ab(x, y)
        a_deg = math.degrees(a)
        b_deg = math.degrees(b)
        print(f'{a_deg}, {b_deg}')
        # self.s1.angle = a
        # self.s2.angle = b
        self._socket.emit('laser_coords', [x, y])

    def stop_thread(self):
        with self._lock:
            self._running_thread = False
    
    def run(self):
        self._running_thread = True
        period = 1.0 / self.fps
        running = True
        while(running):
            self._socket.sleep(period)
            with self._lock:
                on = self._on
            if on:
                self.moveLaser()
            with self._lock:
                running = self._running_thread