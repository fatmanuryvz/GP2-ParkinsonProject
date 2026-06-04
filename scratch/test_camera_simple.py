import cv2
import time

def test_camera():
    print("Testing index 0 with DEFAULT backend...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("Success on index 0 (DEFAULT)")
        ret, frame = cap.read()
        if ret:
            print("Read frame successful")
        cap.release()
    else:
        print("Failed on index 0 (DEFAULT)")

if __name__ == "__main__":
    test_camera()
