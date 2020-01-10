import io
import json 
import torch 

# from torchvision import models
import torchvision.transforms  as transforms
from PIL import Image
from flask import Flask, jsonify, request
import numpy as np 
import model

app = Flask(__name__)

@app.route('/')
def hello():
	return "Hello"

def prediction(img_bytes):
	tensor = transform_image(img_bytes)
	device = torch.device("cpu")
	m = model.Net()
	m.load_state_dict(torch.load("mnist_cnn.pt", map_location=torch.device('cpu')))
	m.eval()

	with torch.no_grad():
		tensor = tensor.to(device)
		outputs = m.forward(tensor)
		# print(outputs)
		y_hat = outputs.argmax(dim=1, keepdim = True)

	return y_hat.item()






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
	
	img = transformations(img)

	img = img.convert('L')

	a = np.array(image)
	val = a.flatten()
	mean = np.mean(val) / 255
	std = np.std(val) / 255

	trans = transforms.Compose([transforms.ToTensor(),
		transforms.Normalize((mean,), (std,))])

	tensor = trans(img)

	re_trans = transforms.ToPILImage()

	re_img = re_trans(tensor)
	# re_img.save("transform_and_back_test.png")
	# print(tensor.shape)
	# transformed_image = transformations(img)
	# transformed_image.save("test_3.png")

	return tensor.unsqueeze(0)


@app.route('/predict', methods = ['POST'])
def predict():
	if(request.method == "POST"):
		file = request.files['Image']
		# print(file)
		img_bytes = file.read()
		predicted = prediction(img_bytes)
		
		print(predicted)

		response = jsonify({'prediction': str(predicted)})
		response.headers.add('Access-Control-Allow-Origin', '*')
		response.headers.status = 200
		response.headers.mimetype = 'application/json'
		return response
