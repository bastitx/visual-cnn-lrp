from flask import Flask, request, jsonify
import torch
from convnet import ConvNet

app = Flask(__name__)

model = ConvNet()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

@app.route("/metaData", methods=['GET'])
def metaData():
    return jsonify(model.getMetaData())
