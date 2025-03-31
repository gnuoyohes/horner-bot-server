from flask import Flask, render_template, request
from flask_socketio import SocketIO
from threading import Event, Lock
import cv2
from ultralytics import YOLO
import base64
import psutil

app = Flask(__name__)
socket = SocketIO(app)

thread = None
thread_lock = Lock()
thread_event = Event()

client_count = 0

state = {
    'laser_running': False,
    'use_yolo': False,
    'minimum_contour_area': 1500,
    'laser_coords': [0, 0] # [0, 0] is top left, [1, 1] is bottom right of #laser-container
}

camera = cv2.VideoCapture(0)
backsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
# model = YOLO("yolo11n.pt")
model = YOLO("yolov8n.pt")

def detect_objects_yolo(frame):
    results = model.track(frame, persist=False)
    return results[0].plot()

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

    # biggest_contour = None
    # for cnt in contours:
    #     if biggest_contour is None or cv2.contourArea(cnt) > cv2.contourArea(biggest_contour):
    #         biggest_contour = cnt
    
    # if biggest_contour is not None:
    #     x, y, w, h = cv2.boundingRect(biggest_contour)
    #     frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)

    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > state['minimum_contour_area']]
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)

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
        thread = None

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
    global thread, client_count
    client_count += 1

    socket.emit('state', state)

    if client_count == 1:
        with thread_lock:
            if thread is None:
                thread_event.set()
                thread = socket.start_background_task(generate_frames, thread_event)
                print('Video stream thread started')

@socket.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    print(f'Client disconnected with id: {client_id}')
    global thread, client_count
    client_count -= 1

    if client_count == 0:
        thread_event.clear()
        with thread_lock:
            if thread is not None:
                thread.join()
                thread = None
                print('Video stream thread stopped')

@socket.on('update_state')
def update_state(params):
    global state
    for p in params:
        if p == 'minimum_contour_area':
            new_val = float(params[p])
        else:
            new_val = params[p]
        state[p] = new_val
        print(f"State property \"{p}\" updated to {new_val}")
    socket.emit('state', params)
    


if __name__ == '__main__':
    socket.run(app, debug=True, host="0.0.0.0")