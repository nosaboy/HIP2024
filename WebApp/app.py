from flask import Flask, Response, render_template
import flask
import torch
from torch import nn
import cv2
import matplotlib
from matplotlib import pyplot as plt
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linearStack = nn.Sequential(nn.Linear(784,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,10))

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linearStack(x)
        return logits

modelPath = "model.pth" # add one when we have a model
app = Flask(__name__)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
count = 0
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(modelPath))
model.eval()

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video capture
cap = cv2.VideoCapture(0)

def generate_frames():
    cv2.namedWindow("bob",cv2.WINDOW_NORMAL)
    while True:
        # Read a frame from the webcam
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        #assuming only one face cause that's what the program seems to be doing
        if len(faces)>0:
            croppedFace = None
            # Draw bounding boxes around the faces
            x,y,w,h = faces[0] # get dimensions of face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            croppedFace = frame[y:y+h, x:x+h]
            croppedFace = cv2.cvtColor(croppedFace,cv2.COLOR_BGR2RGB)

            heartRate = 0
            #heartRate = model(croppedFace)
            frame = cv2.flip(frame,1) # flip horizontally
            cv2.putText(frame,f"Heart Rate: {heartRate}",(len(frame[0])-x-w-3,y-4),cv2.FONT_ITALIC,0.4,(0,0,0),1)
        else:
            frame = cv2.flip(frame, 1)  # flip horizontally
            cv2.putText(frame,f"Theres no face ( ._.)",(50,50),cv2.FONT_ITALIC,1,(0,0,0),2)
        # will probably use putText to display the heartrate I dont want to figure out how to update the html file without reloading the page
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html',heartRate=0)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True)