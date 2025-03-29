import cv2

def return_camera_indices():
    # checks the first 10 indexes.
    arr = []
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            arr.append(i)
            cap.release()
    return arr

print(return_camera_indices())