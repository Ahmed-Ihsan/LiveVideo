from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import pickle
import os
from PIL import Image
import numpy as np
import logging
import uvicorn
# import asyncio
# import threading
# import socketio

app = FastAPI()
# sio = socketio.AsyncServer(cors_allowed_origins="*")

STOP = 0
# START = 1

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/")
def read_root():
    return {"message": "Face Recognition API"}

@app.post("/train-face-recognition")
async def face_learning():
    # Path for face image database
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create() 
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:
            
            try:
                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
            except:
                pass
        return faceSamples,ids

    logger.info ("\nTraining for the faces has been started. It might take a while.\n")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') 

    # Print the number of faces trained and end program
    logger.info(f"{len(np.unique(ids))} faces trained. Exiting Training Program")


@app.post("/add-face")
async def add_face(name: str = Form(...), files: list[UploadFile] = File(...)):
    print(name)
    try:
        with open('names.pkl', 'rb') as f:
            names = pickle.load(f)

        if name not in names:
            names.append(name)
        
        id = names.index(name)
        with open('names.pkl', 'wb') as f:
            pickle.dump(names, f)

        # Ensure the dataset directory exists
        if not os.path.exists('dataset'):
            os.makedirs('dataset')

        count = len([f for f in os.listdir('dataset') if f.startswith(f"{name}.{id}.")])

        # Load the Haar Cascade face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        for file in files:
            image = Image.open(file.file).convert('L')
            image_np = np.array(image, 'uint8')

            faces = face_cascade.detectMultiScale(image_np, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = image_np[y:y+h, x:x+w]
                face_image = Image.fromarray(face)
                count += 1
                face_image.save(f"dataset/{name}.{id}.{count}.jpg")
                print(count)
                break  # Save only one face per image

        return {"message": f"{len(files)} images for {name} received and saved successfully."}
    except Exception as e:
        return {"error": str(e)}


# @sio.event
# async def connect(sid, environ):
#     print("Client connected:", sid)

# @sio.event
# async def disconnect(sid):
#     print("Client disconnected:", sid)
    
def generate_frames():
    global STOP
    
    # if STOP == 0:
    #     STOP = 1
    #     logger.info("Starting to capture frames")
    # else:
    #     logger.info("Stopping capturing frames")
    #     STOP = 0
    #     return 0
    
    logger.info("Starting to capture frames")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Starting realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less than 100 ==> "0" is a perfect match
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# @sio.on('start_capture')
# async def start_capture(sid):
#     global STOP
#     if STOP == 0:
#         STOP = 1
#         threading.Thread(target=generate_frames).start()
#     await sio.emit('status', {'status': 'started'})

# @sio.on('stop_capture')
# async def stop_capture(sid):
#     global STOP
#     STOP = 0
#     await sio.emit('status', {'status': 'stopped'})

# app.add_middleware(
#     socketio.ASGIApp,
#     socketio_app=sio,
# )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
