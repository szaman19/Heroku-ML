import io
import json 

# from torchvision import models
# import torchvision.transforms  as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def hello():
	return "Hello"

@app.route('/predict', methods = ['POST'])
def predict():
	if(request.method == "POST"):
		file = request.files['Image']
		# print(file)
		img_bytes = file.read()
		image = Image.open(io.BytesIO(img_bytes))
		image.save("test.png")

		response = jsonify({'body': 'data'})
		response.headers.add('Access-Control-Allow-Origin', '*')
		return response
