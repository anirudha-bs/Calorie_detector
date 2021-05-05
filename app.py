import os
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify
import darknet
import requests
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from imageClassification.validateImageClassification import checkForFood

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static/images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#api setup
url = 'https://api.edamam.com/api/food-database/v2/parser?ingr={}&category=generic-foods&app_id=4ae4276c&app_key=700f9952ee31704f78345031cbbe2ece'


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/service-worker.js')
def sw():
    return app.send_static_file('service-worker.js'), 200, {'Content-Type': 'text/javascript'}

@app.route('/manifest.json')
def manf():
    return app.send_static_file('manifest.json')

@app.route("/red")
def red():
    return render_template("upload.html")


@app.route('/predict', methods=['GET','POST'])
def predict():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)
    if checkForFood(f):
        output = imageProcessing(f)
        try:
            label_name = output[0][0].decode("utf-8")
        except:
            return render_template('error.html')

        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        image = mpimg.imread(path)
        plt.imshow(image)
        plt.title(label_name)
        plt.savefig(path)
        
        if label_name == 'burger':
            label_name = 'vegan%\20burger'

        r = requests.get(url.format(label_name)).json()

        try:
            nutrition = {
            'Label': label_name,
            'Energy_Kcal': r["parsed"][0]["food"]["nutrients"]["ENERC_KCAL"],
            'Protein_g': r["parsed"][0]["food"]["nutrients"]["PROCNT"],
            'Fat_g': r["parsed"][0]["food"]["nutrients"]["FAT"],
            'Carbs_g': r["parsed"][0]["food"]["nutrients"]["CHOCDF"],
            }
        except:
            return render_template('error.html')
                
        return render_template('prediction.html',predicted_path = path,nutrition = nutrition)

    else:
        return render_template('error.html')


def imageProcessing(image):
    return darknet.getImageAndModel(image)


if __name__ == '__main__':
    app.run(debug=True,threaded=True)
