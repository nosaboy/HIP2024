import flask
import torch
import cv2
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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
app = flask.Flask(__name__)

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


@app.route('/', methods = ['GET','POST'])
def index():
    count = 0
    while(True):
        count+=1
        load(count)
def load(count):
    return flask.render_template('index.html',count=count)


app.run(debug=True)