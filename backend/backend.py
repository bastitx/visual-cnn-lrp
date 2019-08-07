from flask import Flask, abort, request, jsonify
from flask_cors import CORS
import sys
import torch
import numpy as np

CONVNET = "lrptoolbox" 

app = Flask(__name__)
CORS(app)

models = {}
if CONVNET == "lrptoolbox":
    import pickle
    import modules_tb
    sys.modules['modules'] = modules_tb
    from linearnet_tb import LinearNet
    from convnet_tb import ConvNet
    with open('long-tanh.nn', 'rb') as f:
        model = pickle.load(f, encoding='latin1')
        model.drop_softmax_output_layer()
        models['linear'] = LinearNet(model)
    with open("LeNet-5.nn", 'rb') as f:
        model = pickle.load(f, encoding='latin1')
        models['conv'] = ConvNet(model)
else:
    from linearnet import LinearNet
    from convnet import ConvNet
    import torch.nn.functional as F
    
    models['linear'] = LinearNet()
    models['linear'].load_state_dict(torch.load("mnist_linear.pt"))
    models['linear'].eval()

    models['conv'] = ConvNet()
    models['conv'].load_state_dict(torch.load("mnist_cnn_PN.pt"))
    models['conv'].eval()

def checkNetwork(network):
    if network not in ['linear', 'conv']:
        abort(400) # Bad Request

@app.route("/metaData/<network>", methods=['GET'])
def metaData(network):
    network = network.lower()
    checkNetwork(network)
    metadata = models[network].getMetaData(torch.zeros(1,1,28,28))
    return jsonify(metadata)

@app.route("/activations/<network>", methods=['POST'])
def getActivations(network):
    network = network.lower()
    checkNetwork(network)
    data = request.json
    input_data = data['data']
    input = torch.Tensor(input_data)
    if CONVNET == "lrptoolbox":
        models[network].forward(input)
    else:
        models[network](input)
    new_output = []
    for out in models[network].getActivations():
        new_output.append(out.tolist())
    return jsonify(new_output)

@app.route("/lrp/<network>/<method>", methods=['POST'])
def getHeatmap(network, method):
    network = network.lower()
    checkNetwork(network)
    method = method.lower()
    if method not in ['simple', 'epsilon', 'alphabeta']:
        abort(400)
    data = request.json
    input_data = data['data']
    heatmap_selection = data['heatmap_selection']
    parameter = data['parameter']
    try:
        parameter = float(parameter)
    except:
        parameter = 0.01
    input = torch.Tensor(input_data)
    if CONVNET == "lrptoolbox":
        output = models[network].forward(input)
    else:
        output = models[network](input)
    output_ = torch.zeros(10)
    if heatmap_selection != None and -1 < heatmap_selection < 10:
        output_[heatmap_selection] = 1
    else:
        output_[output.argmax()] = 1
    output_ = output_[None,:]
    models[network].relprop(output_, method, parameter)
    new_output = []
    for out in models[network].getRelevances():
        new_output.append(out.tolist())
    return jsonify(new_output)
