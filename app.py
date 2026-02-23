from flask import Flask, render_template, redirect, url_for, flash, request, Response, session
from flask_socketio import SocketIO
from flask_login import LoginManager, UserMixin, current_user, login_user, logout_user, fresh_login_required

from threading import Event, Lock
import cv2
# from ultralytics import YOLO
# import torch
import psutil
import time
from datetime import timedelta

from picamera2 import Picamera2
from picamera2.devices import Hailo


from laser import Laser
from login_form import LoginForm

# import logging
# logging.basicConfig(level=logging.DEBUG, filename="log", filemode="a+",
                        # format="%(asctime)-15s %(levelname)-8s %(message)s")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hornerbot328'
app.config['TESTING'] = False
socket = SocketIO(app, async_mode='threading', cors_allowed_origins="*")
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
        def __init__(self, id, username, password):
            self.id = id
            self.username = username
            self.password = password

users = [
    User(0, 'seho', 'horner328'),
    User(1, 'elizabeth', 'horner328')
]

client_count = 0

state = {
    'laser_running': False,
    'use_yolo': True,
    'minimum_contour_area': 2000,
    'manual_mode': False,
    'classes_to_detect': [15] # 0: person, 15: cat, 67: cell phone
}

thread_lock = Lock()
thread = None
thread_event = Event()

laser = Laser(1, 2.5, 0.5, 0, 0, 27, 17, 60, socket)
laser_thread = None

hailo = Hailo('resources/yolov8m_h8l.hef')
# hailo_thread_lock = Lock()
# Load class names from the labels file
with open('resources/coco.txt', 'r', encoding="utf-8") as f:
    CLASS_NAMES = f.read().splitlines()
CONFIDENCE_THRESH = 0.2

# CAMERA_INDEX = 0
# camera = cv2.VideoCapture(CAMERA_INDEX)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 960
MODEL_HEIGHT, MODEL_WIDTH, _ = hailo.get_input_shape()
print(f"Model Width: {MODEL_WIDTH}, Model Height: {MODEL_HEIGHT}")
# CAMERA_RES_W = 3280
# CAMERA_RES_H = 2464
camera = Picamera2()
main = {'size': (CAMERA_WIDTH, CAMERA_HEIGHT), 'format': 'XRGB8888'}
lores = {'size': (MODEL_WIDTH, MODEL_HEIGHT), 'format': 'RGB888'}
controls = {'FrameRate': 60}
config = camera.create_preview_configuration(main, lores=lores, controls=controls)
camera.configure(config)
camera.start()

frame_out_bytes = None

backsub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

# yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {yolo_device}')
# model = YOLO("resources/yolo11n_ncnn_model")
# MODEL_IMGSZ = 320

def detect_objects_hailo(frame):
    """Extract detections from the HailoRT-postprocess output."""
    hailo_output = hailo.run(frame)
    results = []
    frame_out = frame.copy()
    for class_id, detections in enumerate(hailo_output):
        if class_id in state['classes_to_detect'] and len(detections) > 0:
            detection = detections[0]
            score = detection[4]
            if score >= CONFIDENCE_THRESH:
                y0, x0, y1, x1 = detection[:4]
                if state['laser_running'] and not state['manual_mode']:
                    laser.set_obj_coords([x0 + (x1 - x0)/2, y0 + (y1 - y0)/2])
                bbox = (int(x0 * MODEL_WIDTH), int(y0 * MODEL_HEIGHT), int(x1 * MODEL_WIDTH), int(y1 * MODEL_HEIGHT))
                results.append([CLASS_NAMES[class_id], bbox, score])
                break

    for class_name, bbox, score in results:
                x0, y0, x1, y1 = bbox
                label = f"{class_name} %{int(score * 100)}"
                cv2.rectangle(frame_out, (x0, y0), (x1, y1), (0, 255, 0, 0), 2)
                cv2.putText(frame_out, label, (x0 + 5, y0 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0, 0), 1, cv2.LINE_AA)
                
    return frame_out

# def detect_objects_yolo(frame):
#     results = model.track(
#         frame,
#         imgsz = MODEL_IMGSZ,
#         verbose=False,
#         persist=False,
#         conf=CONFIDENCE_THRESH,
#         classes=state['classes_to_detect'],
#         half=False,
#         max_det=3
#     )
#     # print(results)
#     result = results[0]
#     if len(result.boxes) > 0:
#         det = result.boxes[0]
#         x, y, width, height = det.xywhn[0].tolist()
#         if state['laser_running'] and not state['manual_mode']:
#             laser.set_obj_coords([x, y])
#         # print(f"{x}, {y}, {width}, {height}")
#     else:
#         laser.set_obj_coords([0.5, 0.5])

#     annotated_frame = result.plot()

#     # Get inference time
#     inference_time = result.speed['inference']
#     fps = 1000 / inference_time  # Convert to milliseconds
#     text = f'FPS: {fps:.1f}'

#     # Define font and position
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text_size = cv2.getTextSize(text, font, 1, 2)[0]
#     text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
#     text_y = text_size[1] + 10  # 10 pixels from the top

#     # Draw the text on the annotated frame
#     cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

#     return annotated_frame

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
    global frame_out_bytes
    while event.is_set():
        start_time = time.time()
        frame = camera.capture_array('lores')
        if state['use_yolo']:
            # annotated_frame = detect_objects_yolo(frame)
            annotated_frame = detect_objects_hailo(frame)
        else:
            annotated_frame = detect_objects_backsub(frame)
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        # frame_b64 = base64.b64encode(buffer).decode('utf-8')
        # print(f'Size: {len(frame_b64) * 3 / 4}')

        # Save the image to a BytesIO object
        with thread_lock:
            frame_out_bytes = buffer.tobytes()
        # socket.emit('video_frame', {'image': frame_b64})
        

        fps = 1.0 / (time.time() - start_time)
        stats = {
            'server_fps': round(fps, 1),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }
        socket.emit('stats', stats)

# def send_frames():
#     while True:
#         # socket.sleep(0.1)
#         if frame_out_bytes:
#             # print("Size: " + len(frame_out_bytes))
#             yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n' 
#                     b'Content-Length: ' + str(len(frame_out_bytes)).encode() + b'\r\n\r\n'+ 
#                     frame_out_bytes + b'\r\n')

@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(seconds=10)

@app.route("/")
@fresh_login_required
def index():
    return render_template('index.html')

@app.route('/video_frame')
@fresh_login_required
def video_frame():
    with thread_lock:
        if frame_out_bytes:
            response = Response(frame_out_bytes)
            return response
        else:
            return '', 204

    # return Response(send_frames(),
    #                 mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = None
        for u in users:
            if u.username == form.username.data and u.password == form.password.data:
                user = u
        if user is None:
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=False)
        return redirect(url_for('index'))
    return render_template('login.html', title='Sign In', form=form)

@login_manager.user_loader
def load_user(user_id):
    for user in users:
        if user.id == int(user_id):
            return user
    return None

@socket.on('connect')
def handle_connect():
    client_id = request.sid
    print(f'Client connected with id: {client_id}')
    global thread, laser_thread, client_count
    client_count += 1
    print(f'Client count: {client_count}')

    socket.emit('state', state)
    socket.emit('laser_coords', laser.get_laser_coords())

    if thread is None:
        thread_event.set()
        thread = socket.start_background_task(generate_frames, thread_event)
        laser_thread = socket.start_background_task(laser.run)
        print('Video stream and laser threads started')

@socket.on('disconnect')
def handle_disconnect():
    logout_user()
    session.clear()
    client_id = request.sid
    print(f'Client disconnected with id: {client_id}')
    global thread, laser_thread, client_count, frame_out_bytes
    client_count -= 1
    print(f'Client count: {client_count}')

    if client_count == 0:
        thread_event.clear()
        laser.stop_thread()
        update_state({'laser_running': False})
        if thread is not None:
            thread.join()
            thread = None
            laser_thread.join()
            laser_thread = None
            frame_out_bytes = None
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
    # print(f"Laser coordinates updated to {new_coords}")

if __name__ == '__main__':
    socket.run(app, host="0.0.0.0", port=5000)