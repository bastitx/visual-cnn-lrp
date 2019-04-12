from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from convnet import ConvNet

app = Flask(__name__)
CORS(app)

model = ConvNet()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

@app.route("/metaData", methods=['GET'])
def metaData():
    return jsonify(model.getMetaData())

@app.route("/getActivations", methods=['POST'])
def getActivations():
    data = request.json
    input = torch.Tensor(data)
    output = model(input)
    new_output = []
    for out in output:
        new_output.append(out.tolist())
    return jsonify(new_output)
