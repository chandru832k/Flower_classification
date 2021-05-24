from flask import Flask,render_template,url_for,request,redirect,flash
from flask_material import Material
import pandas as pd
import numpy as np
import joblib
import os
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# pytorch imports
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F




app = Flask(__name__)
Material(app)


# custom CNN model class
device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class ConvNet(nn.Module):
    def __init__(self,model,num_classes):
        super(ConvNet,self).__init__()
        self.base_model = nn.Sequential(*list(model.children())[:-1])
        self.linear1 = nn.Linear(in_features=2048,out_features=512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=512,out_features=num_classes)
    
    def forward(self,x):
        x = self.base_model(x)
        x = torch.flatten(x,1)
        lin = self.linear1(x)
        x = self.relu(lin)
        out = self.linear2(x)
        return lin, out


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
transformations = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    df = pd.read_csv("data/iris.csv")
    return render_template("preview.html",df_view = df)

@app.route('/cnn_prediction')
def cnn_prediction():
	return render_template("cnn_prediction.html")

@app.route('/cnn_prediction',methods=["POST"])
def cnn():
	if request.method=='POST':
		pic=request.files['myfile']
		pic.save(pic.filename)

		MODEL_PATH=('data/resnet50_TL_model_94%acc.pth')
		num_classes=5
		model = torchvision.models.resnet50(pretrained=True)
		model = ConvNet(model,num_classes)
		model = model.to(device)
		model.load_state_dict(torch.load(MODEL_PATH))


		#print("Hello")

		temp=pic.filename

		#print(temp)

		pic1 = Image.open(temp)
		pix = transformations['train'](pic1)
		#print(pix.size())

		with torch.no_grad():
			model.eval()
			images = pix.to(device).unsqueeze(0)
			ftrs,outputs = model(images)
			_,preds = torch.max(outputs,1)
			print(outputs)
		
		out=-999999999
		for i in range(0,5):
			if(outputs[0][i]>out):
				out=outputs[0][i]

		flower="The flower predicted is"

		if(out==outputs[0][0]):
			flower+="daisy"
		elif(out==outputs[0][1]):
			flower+="dandelion"
		elif(out==outputs[0][2]):
			flower+="rose"
		elif(out==outputs[0][3]):
			flower+="sunflower"
		else:
			flower+="tulip"


		return render_template('cnn_prediction.html',prediction=flower)

		



@app.route('/',methods=["POST"])
def analyze():
	if request.method == 'POST':
		petal_length=request.form['petal_length']
		sepal_length=request.form['sepal_length']
		petal_width=request.form['petal_width']
		sepal_width=request.form['sepal_width']
		model_choice=request.form['model_choice']

		sample_data=[sepal_length,sepal_width,petal_length,petal_width]
		clean_data=[float(i) for i in sample_data]
		ex1 = np.array(clean_data).reshape(1,-1)


		if model_choice == 'logitmodel':
		    logit_model = joblib.load('data/logit_model_iris.pkl')
		    result_prediction = logit_model.predict(ex1)
		elif model_choice == 'knnmodel':
			knn_model = joblib.load('data/knn_model_iris.pkl')
			result_prediction = knn_model.predict(ex1)
		elif model_choice == 'svmmodel':
			svm_model = joblib.load('data/svm_model_iris.pkl')
			result_prediction = svm_model.predict(ex1)
		elif model_choice == 'decisiontree':
			dt_model = joblib.load('data/dtree_model_iris.pkl')
			result_prediction = dt_model.predict(ex1)
		elif model_choice == 'nbmodel':
			nb_model = joblib.load('data/nb_model_iris.pkl')
			result_prediction = nb_model.predict(ex1)
		elif model_choice == 'ldamodel':
			lda_model = joblib.load('data/lda_model_iris.pkl')
			result_prediction = lda_model.predict(ex1)

	return render_template('index.html', petal_width=petal_width,
		sepal_width=sepal_width,
		sepal_length=sepal_length,
		petal_length=petal_length,
		clean_data=clean_data,
		result_prediction=result_prediction,
		model_selected=model_choice)


if __name__ == '__main__':
	app.run(debug=True)
