from flask import Flask, render_template, request
from flask_socketio import SocketIO
from threading import Event, Lock
import cv2
from ultralytics import YOLO
import base64
import psutil

app = Flask(__name__)
socketio = SocketIO(app)

thread = None
thread_lock = Lock()
thread_event = Event()

client_count = 0

running = False

use_yolo = False

camera = cv2.VideoCapture(0)
backsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
# model = YOLO("yolo11n.pt")
model = YOLO("yolov8n.pt")

def detect_objects_yolo(frame):
    results = model.track(frame, persist=True)
    return results[0].plot()

def detect_objects_backsub(frame):
    fg_mask = backsub.apply(frame)
    # Find contours
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    retval, mask_thresh = cv2.threshold( fg_mask, 180, 255, cv2.THRESH_BINARY)
    # set the kernal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Apply erosion
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
    # min_contour_area = 1000  # Define your minimum area threshold
    # large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    biggest_contour = None
    for cnt in contours:
        if biggest_contour is None or cv2.contourArea(cnt) > cv2.contourArea(biggest_contour):
            biggest_contour = cnt
    
    frame_out = frame.copy()
    if biggest_contour is not None:
        x, y, w, h = cv2.boundingRect(biggest_contour)
        frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)

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
                if use_yolo:
                    annotated_frame = detect_objects_yolo(frame)
                else:
                    annotated_frame = detect_objects_backsub(frame)
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'image': frame_b64})
            else:
                break
    finally:
        event.clear()
        thread = None

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get_status", methods=['GET'])
def get_status():
    global running
    return "running" if running else "stopped"

@app.route("/start", methods=['POST'])
def start():
    global running
    if not running:
        running = True
        return "success"
    else:
         return "error"
    
@app.route("/stop", methods=['POST'])
def stop():
    global running
    if running:
        running = False
        return "success"
    else:
         return "error"
    
@app.route('/system_info', methods=['GET'])
def system_info():
    stats = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    }
    return stats

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    print(f'Client connected with id: {client_id}')
    global thread, client_count
    client_count += 1

    if client_count == 1:
        with thread_lock:
            if thread is None:
                thread_event.set()
                thread = socketio.start_background_task(generate_frames, thread_event)
                print('Video stream thread started')

@socketio.on('disconnect')
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

@socketio.on('switch_object_detection')
def switch_object_detection():
    global use_yolo
    use_yolo = not use_yolo
    alg = 'yolo' if use_yolo else 'backsub'
    print(f"Object detection algorithm switched to {alg}")


if __name__ == '__main__':
    socketio.run(app, debug=True, host="0.0.0.0")