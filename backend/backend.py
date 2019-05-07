from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from convnet import ConvNet

app = Flask(__name__)
CORS(app)

model = ConvNet()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()
outputLayers = [1, 2, 4, 5, 8, 10, 11]

@app.route("/metaData", methods=['GET'])
def metaData():
    return jsonify(model.getMetaData())

@app.route("/activations", methods=['POST'])
def getActivations():
    data = request.json
    input_data = data['data']
    input = torch.Tensor(input_data)
    model(input)
    new_output = []
    for out in model.getActivations(outputLayers):
        new_output.append(out.tolist())
    return jsonify(new_output)

@app.route("/lrpsimple", methods=['POST'])
def getHeatmap():
    data = request.json
    input_data = data['data']
    heatmap_selection = data['heatmap_selection']
    input = torch.Tensor(input_data)
    output = model(input)
    output_ = torch.zeros(10)
    if heatmap_selection != None and -1 < heatmap_selection < 10:
        output_[heatmap_selection] = 1
    else:
        output_[output.argmax()] = 1
    output_ = output_[None,:]
    model.relprop(output_)
    new_output = []
    for out in model.getRelevances(outputLayers):
        new_output.append(out.tolist())
    return jsonify(new_output)
