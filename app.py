import io
import json 

# from torchvision import models
import torchvision.transforms  as transforms
from PIL import Image
from flask import Flask, jsonify, request
import numpy as np 

app = Flask(__name__)

@app.route('/')
def hello():
	return "Hello"


def transform_image(img_bytes):
	transformations = transforms.Compose([transforms.Grayscale(), 
										  transforms.Resize(28)])
	image = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
	image.save("pretransform.png")
	x = np.array(image)
	r, g, b, a = np.rollaxis(x, axis = -1)
	r[a == 0] = 255
	g[a == 0] = 255
	b[a == 0] = 255
	x = np.dstack([r, g, b, a])
	img = Image.fromarray(x, 'RGBA')
	# background = Image.new('RGBA', png.size, (255,255,255))
	# img = Image.alpha_composite(background, Image)
	# img.save("")
	transformed_image = transformations(img)
	transformed_image.save("test_3.png")


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
