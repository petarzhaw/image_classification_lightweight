import io
import numpy as np
import onnxruntime
from flask_bootstrap import Bootstrap
from PIL import Image
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)
Bootstrap(app)

 # Path to the model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bvlcalexnet-12-qdq.onnx')
# Load the model
#The CPUExecutionProvider is used to ensure that the model is executed using CPU resources for computation, which can be useful in cases where a GPU is either not available or not required for the specific task.
inference = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider']) # CPUExecutionProvider is used for CPU inference


# Get the input node (the first node of the graph) - this is the input
input_name = inference.get_inputs()[0].name
print(input_name)

# Load the class labels
with open('static/synset.txt', 'r') as f: 
    class_dict = [line.strip() for line in f.readlines()] 

@app.route('/', methods=['GET'])
def index():
    """
    Renders the home page of the web application.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Takes an image file as input and returns a list of top 5 
    predicted classes along with their probabilities.
    """
    img_bytes = request.files['image'].read()
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert('RGB')   
     # Resize the image to the expected input size for AlexNet 
    img = img.resize((224, 224)) 
    # Convert the image to a numpy array
    img_arr = np.array(img).astype(np.float32)  
    # Update the mean pixel values to match the expected input
    mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)  
    # Set the standard deviation to 1 for each channel
    std = np.array([1, 1, 1], dtype=np.float32)  

    img_arr = img_arr - mean  
    img_arr = img_arr / std  

    # Add a batch dimension
    img_arr = np.expand_dims(img_arr, axis=0) 
    # Transpose to (batch_size, channels, height, width)
    img_arr = np.transpose(img_arr, (0, 3, 1, 2))  

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
  app.run(host='0.0.0.0', port=8000)