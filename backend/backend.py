from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import torch
import numpy as np


app = Flask(__name__)
CORS(app)

CONVNET = "old" #TODO: Make this better
if CONVNET == "lrptoolbox":
    from convnet_tb import ConvNet
    import pickle
    import modules_tb
    sys.modules['modules'] = modules_tb
    with open("LeNet-5.nn", 'rb') as f:
    #with open('long-tanh.nn', 'rb') as f:
        model = pickle.load(f, encoding='latin1')
    model.drop_softmax_output_layer()
    model = ConvNet(model)
else:
    import torch.nn.functional as F
    from linearnet import LinearNet as ConvNet
    model = ConvNet()
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    model.eval()

if CONVNET == "lrptoolbox":
    outputLayers = [0, 2, 3, 5, 6, 8, 9, 10]
else:
    #outputLayers = [0, 2, 3, 5, 6, 9, 11, 12]
    outputLayers = [0, 3, 5, 7, 8]

@app.route("/metaData", methods=['GET'])
def metaData():
    if CONVNET == "lrptoolbox":
        metadata = model.getMetaData(torch.zeros(1,32,32,1))
    else:
        metadata = model.getMetaData(torch.zeros(1,1,28,28))
    return jsonify([metadata[i] for i in outputLayers])

@app.route("/activations", methods=['POST'])
def getActivations():
    data = request.json
    input_data = data['data']
    input = torch.Tensor(input_data)
    if CONVNET == "lrptoolbox":
        model.forward(input.permute(0,3,2,1))
    else:
        model(input)
    new_output = []
    for out in model.getActivations(outputLayers):
        if CONVNET == "lrptoolbox" and len(out.shape) == 4:
            new_output.append(np.swapaxes(out, 1, 3).tolist())
        else:
            new_output.append(out.tolist())
    return jsonify(new_output)

@app.route("/lrp/<kind>", methods=['POST'])
def getHeatmap(kind):
    data = request.json
    input_data = data['data']
    heatmap_selection = data['heatmap_selection']
    input = torch.Tensor(input_data)
    if CONVNET == "lrptoolbox":
        output = model.forward(np.array(input.permute(0, 3, 2, 1)))
    else:
        output = model(input)
    output_ = torch.zeros(10)
    if heatmap_selection != None and -1 < heatmap_selection < 10:
        output_[heatmap_selection] = 1
    else:
        output_[output.argmax()] = 1
    output_ = output_[None,:]
    if CONVNET == "lrptoolbox":
        model.lrp(np.array(output_), kind, 0.01)
    else:
        model.relprop(output_, kind, 0.01)
    new_output = []
    for out in model.getRelevances(outputLayers):
        if CONVNET == "lrptoolbox" and len(out.shape) == 4:
            new_output.append(np.swapaxes(out, 1, 3).tolist())
        else:
            new_output.append(out.tolist())
    return jsonify(new_output)
