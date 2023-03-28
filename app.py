import io
import numpy as np
import onnxruntime
import onnx
from flask_bootstrap import Bootstrap
from PIL import Image
from flask import Flask, render_template, request, jsonify
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

app = Flask(__name__)
Bootstrap(app)

 # Path to the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'squeezenet1.0-12-int8.onnx')
# Load the model
inference = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Load the model
#model = onnx.load('squeezenet1.0-12-int8.onnx')

# Get the input node (the first node of the graph)
input_name = inference.get_inputs()[0].name
print(input_name)

with open('static/synset.txt', 'r') as f:
    class_dict = [line.strip() for line in f.readlines()]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_bytes = request.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')    
    img = img.resize((224, 224))  # Resize the image to the expected input size for AlexNet
    img_arr = np.array(img).astype(np.float32)  # Convert the image to a numpy array
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_arr = img_arr / 255.0  # convert pixel values to the range [0, 1]
    img_arr = (img_arr - mean) / std  # normalize using the provided mean and std

    img_arr = np.expand_dims(img_arr, axis=0)  # Add a batch dimension
    img_arr = np.transpose(img_arr, (0, 3, 1, 2))  # Transpose to (batch_size, channels, height, width)

    output = inference.run(None, {"data_0": img_arr})
    
    # make predictions
    probs = np.squeeze(output)
    top_indices = np.argsort(probs)[::-1][:5]
    top_probs = probs[top_indices]
    top_classes = [class_dict[i] for i in top_indices]

    # return the result as a JSON object
    results = [{'class': cls, 'probability': str(prob)} for cls, prob in zip(top_classes, top_probs)]
    return jsonify(results)

if __name__ == "__main__":
  #  from waitress import serve
  #  serve(app, host="0.0.0.0", port=8000)
  app.run(host='0.0.0.0', port=8000)