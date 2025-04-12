import math
from gpiozero import LED
from adafruit_servokit import ServoKit
from threading import Lock
import numpy as np

MIN_SPEED = 0.01
MAX_SPEED = 0.5

class Laser:
    def __init__(self, height, width, depth, a_offset, b_offset, gpio_diode, gpio_servers, fps, socket):
        self.h = height
        self.w = width
        self.d = depth
        self.a_offset = a_offset
        self.b_offset = b_offset
        self.diode = LED(gpio_diode)
        self.diode.off()
        self.servo_power = LED(gpio_servers)
        self.servo_power.off()
        self.servos = None
        self.fps = fps
        self._obj_coords = [0.5, 0.5] # XY coordinates of the object in 2D space, in ([0, 1], [0, 1])
        self._prev_obj_coords = [0.5, 0.5]
        self._laser_coords = [0.5, 0.5] # XY coordinates of the object in 2D space, in ([0, 1], [0, 1])
        self._direction_vector = np.array([1, 0])
        self._socket = socket
        self._lock = Lock()
        self._on = False
        self._manual_mode = False
        self._running_thread = False
    
    # computes new coordinates of laser
    def _compute_coords(self):
        with self._lock:
            ox, oy = self._obj_coords
            px, py = self._prev_obj_coords
            lx, ly = self._laser_coords
        # theta = math.atan2(ly - oy, lx - ox)
        # new_theta = np.random.normal(theta, math.pi/9)
        new_direction_vector = np.array([np.random.normal(x, 0.3) for x in self._direction_vector])
        diff_vector = np.array([lx - ox, ly - oy])
        d = np.linalg.norm(diff_vector)
        if d != 0:
            diff_vector = diff_vector / d
        diff_multiplier = 5 * 0.3 ** d
        # print(diff_multiplier)
        # multiplier = 0.5
        mag = np.linalg.norm(np.array([ox - px, oy - py]))
        self._direction_vector = new_direction_vector + diff_multiplier * diff_vector
        speed = max(min(2 * mag, MAX_SPEED), MIN_SPEED)
        # print(speed)
        # print(f'speed: {speed}, vector: {self._direction_vector}')
        wall_multiplier = 4
        if lx < 0.3:
            self._direction_vector[0] += wall_multiplier * (1 - lx)
        if lx > 0.7:
            self._direction_vector[0] -= wall_multiplier * (lx)
        if ly < 0.3:
            self._direction_vector[1] += wall_multiplier * (1 - ly)
        if ly > 0.7:
            self._direction_vector[1] -= wall_multiplier * (ly)

        norm = np.linalg.norm(self._direction_vector)
        if norm != 0:
            self._direction_vector = self._direction_vector / norm
        
        new_x = lx + speed * self._direction_vector[0]
        new_y = ly + speed * self._direction_vector[1]
        
        new_x = min(max(new_x, 0), 1)
        new_y = min(max(new_y, 0), 1)
        self.set_laser_coords([new_x, new_y])
    
    # converts point (x, y) in the 2D coordinate plane to angles (a, b) of the servos, in degrees
    def _xy_to_ab(self, x, y):
        a = math.atan((x*self.w - self.w/2) / (self.w - y*self.w + self.d))
        b = math.acos(self.h / (math.sqrt((self.w - y*self.w + self.d) ** 2 + (x*self.w - self.w/2) ** 2 + self.h ** 2)))
        return math.degrees(a) + self.a_offset, math.degrees(b) + self.b_offset

    def on(self):
        with self._lock:
            self.servo_power.on()
            self.diode.on()
            self.servos = ServoKit(channels=16)
            self.servos.servo[0].set_pulse_width_range(500, 2500)
            self.servos.servo[1].set_pulse_width_range(500, 2500)
            self.servos.servo[0].angle = 90
            self.servos.servo[1].angle = 160
            self._on = True
    
    def off(self):
        with self._lock:
            self.servo_power.off()
            self.diode.off()
            self.servos = None
            self._on = False

    def manual_on(self):
        with self._lock:
            self._manual_mode = True

    def manual_off(self):
        with self._lock:
            self._manual_mode = False

    def set_obj_coords(self, new_coords):
        with self._lock:
            self._prev_obj_coords = self._obj_coords.copy()
            self._obj_coords = new_coords
        self._compute_coords()

    def set_laser_coords(self, new_coords):
        with self._lock:
            self._laser_coords = new_coords

    def get_laser_coords(self):
        with self._lock:
            return self._laser_coords
    
    def moveLaser(self):
        with self._lock:
            x, y = self._laser_coords
        a, b = self._xy_to_ab(x, y)
        # print(f'{a}, {b}')
        with self._lock:
            if self.servos:
                self.servos.servo[0].angle = 90 - a
                self.servos.servo[1].angle = 180 - b
        self._socket.emit('laser_coords', [x, y])

    def stop_thread(self):
        with self._lock:
            self._running_thread = False
    
    def run(self):
        self._running_thread = True
        period = 1.0 / self.fps
        running = True
        while(True):
            self._socket.sleep(period)
            with self._lock:
                on = self._on
            if on:
                self.moveLaser()
            with self._lock:
                running = self._running_thread
            if not running:
                self.off()
                return