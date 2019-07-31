from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import torch
import numpy as np


app = Flask(__name__)
CORS(app)

CONVNET = "tb_linear" 
if CONVNET.startswith("tb"):
    import pickle
    import modules_tb
    sys.modules['modules'] = modules_tb
    if CONVNET.endswith("linear"):
        from linearnet_tb import LinearNet as MyNet
        f = open('long-tanh.nn', 'rb')
    else:
        from convnet_tb import ConvNet as MyNet
        f = open("LeNet-5.nn", 'rb')
    model = pickle.load(f, encoding='latin1')
    f.close()
    model.drop_softmax_output_layer()
    model = MyNet(model)
else:
    if CONVNET.endswith("linear"):
        from linearnet import LinearNet as MyNet
        path = "mnist_linear.pt"
    else: # conv
        from convnet import ConvNet as MyNet
        path = "mnist_cnn_PN.pt"
    import torch.nn.functional as F
    
    model = MyNet()
    model.load_state_dict(torch.load(path))
    model.eval()

if CONVNET == "tb_conv":
    outputLayers = [0, 2, 3, 5, 6, 8, 9, 10]
elif CONVNET == "linear": 
    outputLayers = [0, 3, 5, 7, 8]
elif CONVNET == "tb_linear":
    outputLayers = [0, 3, 5, 7, 8]
else: # conv
    outputLayers = [0, 2, 3, 5, 6, 9, 11, 12]

@app.route("/metaData", methods=['GET'])
def metaData():
    if CONVNET == "tb_conv":
        metadata = model.getMetaData(torch.zeros(1,32,32,1))
    else:
        metadata = model.getMetaData(torch.zeros(1,1,28,28))
    return jsonify([metadata[i] for i in outputLayers])

@app.route("/activations", methods=['POST'])
def getActivations():
    data = request.json
    input_data = data['data']
    input = torch.Tensor(input_data)
    if CONVNET.startswith("tb"):
        if CONVNET == "tb_conv":
            input = input.permute(0,3,2,1)
        model.forward(np.array(input))
    else:
        model(input)
    new_output = []
    for out in model.getActivations(outputLayers):
        if CONVNET.startswith("tb") and len(out.shape) == 4:
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
    if CONVNET.startswith("tb"):
        if CONVNET == "tb_conv":
            input = input.permute(0,3,2,1)
        output = model.forward(np.array(input))
    else:
        output = model(input)
    output_ = torch.zeros(10)
    if heatmap_selection != None and -1 < heatmap_selection < 10:
        output_[heatmap_selection] = 1
    else:
        output_[output.argmax()] = 1
    output_ = output_[None,:]
    if CONVNET.startswith("tb"):
        model.lrp(np.array(output_), kind, 0.01)
    else:
        model.relprop(output_, kind, 0.01)
    new_output = []
    for out in model.getRelevances(outputLayers):
        if CONVNET.startswith("tb") and len(out.shape) == 4:
            new_output.append(np.swapaxes(out, 1, 3).tolist())
        else:
            new_output.append(out.tolist())
    return jsonify(new_output)
