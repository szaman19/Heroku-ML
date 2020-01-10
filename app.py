import io
import json 

# from torchvision import models
import torchvision.transforms  as transforms
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def hello():
	return "Hello"


def transform_image(img_bytes):
	transformations = transforms.Compose([transforms.Resize(28)])
	image = Image.open(io.BytesIO(img_bytes)).convert('LA')
	transformed_image = transformations(image)
	transformed_image.save("test.png")


@app.route('/predict', methods = ['POST'])
def predict():
	if(request.method == "POST"):
		file = request.files['Image']
		# print(file)
		img_bytes = file.read()
		transform_image(img_bytes)
		

		response = jsonify({'body': 'data'})
		response.headers.add('Access-Control-Allow-Origin', '*')
		return response
