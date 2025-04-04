from flask import Flask, render_template, request
from flask_socketio import SocketIO
from threading import Event, Lock
import cv2
from ultralytics import YOLO
import torch
import base64
import psutil
from laser import Laser

app = Flask(__name__)
socket = SocketIO(app)

state = {
    'laser_running': False,
    'use_yolo': False,
    'minimum_contour_area': 2000,
    'manual_mode': False,
    'classes_to_detect': [0] # 0: person, 15: cat, 67: cell phone
}

thread_lock = Lock()
thread = None
thread_event = Event()

laser = Laser(1, 2, 1, 17, 27, 22, 60, socket)
laser_thread = None

CAMERA_INDEX = 1
camera = cv2.VideoCapture(CAMERA_INDEX)
CAMERA_WIDTH = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
CAMERA_HEIGHT = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
CAMERA_FPS =  camera.get(cv2.CAP_PROP_FPS)
print(f"CAMERA WIDTH: {CAMERA_WIDTH}")
print(f"CAMERA HEIGHT: {CAMERA_HEIGHT}")
print(f"CAMERA FPS: {CAMERA_FPS}")
backsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {yolo_device}')
model = YOLO("yolo11s.pt").to(yolo_device)
# model = YOLO("yolo11n.pt").to(yolo_device)
# model = YOLO("yolov8n.pt")

client_count = 0

def detect_objects_yolo(frame):
    results = model.track(
        frame,
        verbose=False,
        persist=True,
        conf=0.3,
        classes=state['classes_to_detect'],
        imgsz=(CAMERA_HEIGHT, CAMERA_WIDTH),
        half=False,
        max_det=3
    )
    # print(results)
    result = results[0]
    if len(result.boxes) > 0:
        det = result.boxes[0]
        x, y, width, height = det.xywhn[0].tolist()
        if state['laser_running'] and not state['manual_mode']:
            laser.set_obj_coords([x, y])
        # print(f"{x}, {y}, {width}, {height}")

    return result.plot()

def detect_objects_backsub(frame):
    fg_mask = backsub.apply(frame)
    retval, mask_thresh = cv2.threshold( fg_mask, 180, 255, cv2.THRESH_BINARY)
    # set the kernal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Apply erosion
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
    # Find contours
    contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_out = frame.copy()

    biggest_contour = None
    biggest_cnt_area = 0
    for cnt in contours:
        cnt_area = cv2.contourArea(cnt)
        if (biggest_contour is None or cnt_area > biggest_cnt_area) and cnt_area > state['minimum_contour_area']:
            biggest_contour = cnt
            biggest_cnt_area = cv2.contourArea(biggest_contour)
    
    if biggest_contour is not None:
        x, y, w, h = cv2.boundingRect(biggest_contour)
        if state['laser_running'] and not state['manual_mode']:
            laser_x = (x + w/2) / CAMERA_WIDTH
            laser_y = (y + h/2) / CAMERA_HEIGHT
            laser.set_obj_coords([laser_x, laser_y])
        frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)

    # large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > state['minimum_contour_area']]
    # for cnt in large_contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)

    return frame_out

def generate_frames(event):
    global thread
    try:
        while event.is_set():
            success, frame = camera.read()
            if success:
                if state['use_yolo']:
                    annotated_frame = detect_objects_yolo(frame)
                else:
                    annotated_frame = detect_objects_backsub(frame)
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                socket.emit('video_frame', {'image': frame_b64})
            else:
                break
    finally:
        event.clear()
        # thread = None

@app.route("/")
def index():
    return render_template('index.html')
    
# @app.route('/system_info', methods=['GET'])
# def system_info():
#     stats = {
#         'cpu_percent': psutil.cpu_percent(),
#         'memory_percent': psutil.virtual_memory().percent,
#         'disk_percent': psutil.disk_usage('/').percent
#     }
#     return stats

@socket.on('connect')
def handle_connect():
    client_id = request.sid
    print(f'Client connected with id: {client_id}')
    global thread, laser_thread, client_count
    client_count += 1

    socket.emit('state', state)
    socket.emit('laser_coords', laser.get_laser_coords())

    if client_count == 1:
        # with thread_lock:
        if thread is None:
            thread_event.set()
            thread = socket.start_background_task(generate_frames, thread_event)
            laser_thread = socket.start_background_task(laser.run)
            print('Video stream and laser threads started')

@socket.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    print(f'Client disconnected with id: {client_id}')
    global thread, laser_thread, client_count
    client_count -= 1

    if client_count == 0:
        thread_event.clear()
        laser.stop_thread()
        # with thread_lock:
        if thread is not None:
            thread.join()
            thread = None
            laser_thread.join()
            laser_thread = None
            print('Video stream and laser threads stopped')

@socket.on('update_state')
def update_state(params):
    global state
    for p in params:
        new_val = params[p]
        if p == 'laser_running':
            if new_val:
                laser.on()
            else:
                laser.off()
        elif p == 'manual_mode':
            if new_val:
                laser.manual_on()
            else:
                laser.manual_off()
        elif p == 'minimum_contour_area':
            new_val = float(params[p])
            
        state[p] = new_val
        print(f"State property \"{p}\" updated to {new_val}")
    socket.emit('state', params)

@socket.on('update_class')
def update_class(data):
    class_id = int(data[0])
    if data[1]:
        if class_id not in state['classes_to_detect']:
            state['classes_to_detect'].append(class_id)
            print(f"Class {class_id} added to \"classes_to_detect\"")
    else:
        if class_id in state['classes_to_detect']:
            state['classes_to_detect'].remove(class_id)
            print(f"Class {class_id} removed from \"classes_to_detect\"")
    socket.emit('state', {'classes_to_detect': state['classes_to_detect']})
    
@socket.on('update_laser_coords')
def update_laser_coords(new_coords):
    laser.set_laser_coords(new_coords)
    print(f"Laser coordinates updated to {new_coords}")

if __name__ == '__main__':
    socket.run(app, debug=True, host="0.0.0.0")