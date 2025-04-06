from flask import Flask, render_template, request
from flask_socketio import SocketIO
from threading import Event, Lock
import cv2
from ultralytics import YOLO
import torch
import base64
import psutil
from laser import Laser
from picamera2 import Picamera2

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

laser = Laser(1, 3, 1, 17, 60, socket)
laser_thread = None

# CAMERA_INDEX = 0
# camera = cv2.VideoCapture(CAMERA_INDEX)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 1280
# CAMERA_RES_W = 3280
# CAMERA_RES_H = 2464
camera = Picamera2()
# camera.configure(camera.create_preview_configuration(main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}, sensor={'output_size': (1296, 972)}))
camera.preview_configuration.main.size = (CAMERA_WIDTH, CAMERA_HEIGHT)
camera.preview_configuration.main.format = "RGB888"
camera.preview_configuration.align()
camera.configure("preview")
camera.start()
print(f"CAMERA WIDTH: {CAMERA_WIDTH}")
print(f"CAMERA HEIGHT: {CAMERA_HEIGHT}")
backsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {yolo_device}')
model = YOLO("yolo11n_ncnn_model")
MODEL_IMGSZ = 320
# model = YOLO("yolo11n.pt").to(yolo_device)
# model = YOLO("yolov8n.pt")

client_count = 0

def detect_objects_yolo(frame):
    results = model.track(
        frame,
        imgsz = MODEL_IMGSZ,
        verbose=False,
        persist=False,
        conf=0.2,
        classes=state['classes_to_detect'],
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
    else:
        laser.set_obj_coords([0.5, 0.5])

    annotated_frame = result.plot()

    # Get inference time
    inference_time = result.speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top

    # Draw the text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated_frame

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
            frame = camera.capture_array()
            if state['use_yolo']:
                annotated_frame = detect_objects_yolo(frame)
            else:
                annotated_frame = detect_objects_backsub(frame)
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            socket.emit('video_frame', {'image': frame_b64})
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
    socket.run(app, debug=True, use_reloader=False, host="0.0.0.0")