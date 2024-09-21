import cv2
import numpy as np

class LBPHFaceRecognizer:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.camera = None

    def train(self, images, labels):
        self.recognizer.train(images, np.array(labels))

    def recognize(self, image):
        label, confidence = self.recognizer.predict(image)
        return label, confidence

    def save_model(self, path):
        self.recognizer.save(path)

    def load_model(self, path):
        self.recognizer.read(path)

    def start_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)

    def stop_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def capture_frame(self):
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return frame, gray_frame
        return None, None
