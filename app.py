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

recipe = {'dosa': 'Wash and soak urad dal for 30 minutes.Soak rice flour for 30 minutes adding water just to cover it.Grind the urad dal to a smooth,thick paste adding enough water,say 1/2 to 1 cup.Now add the soaked rice flour to urad dal batter.Add the required salt.Grind again to a smooth paste.Make sure the batter is not too thick or too thin.Transfer it to a wide bowl, Mix the batter with your hands.Let it ferment for 15-20 hours based on the weather.The next morning,batter would have raised well.Mix the batter well and make dosa on a hot pan.Make it thick or crispy as per your wish and enjoy with chutney,sambar !!', 
'poha': 'Wash the poha in a colander. Poha should be moist but not mashed. Add salt and sugar and mix lightly.Heat oil in a non-stick pan. Add mustard seeds and when they splutter add green chillies, curry leaves and peanuts.Sauté for half a minute and add onions and potato, continue to sauté.Add salt, turmeric powder and mix. Sauté for two minutes and add poha. Mix and cook till poha is heated through. Add lemon juice and mix lightly.Garnish with coriander leaves and serve hot.', 'burger': 'In a mixing bowl, add in the mashed potatoes along with the veggies, bread crumbs, salt, pepper and your choice of spice. Mix the potato mixture well.Shape the tikkis, and roll them in panko breadcrumbs and keep aside. Dip the breadcrumbs coated tikkis in maida-cornflour slurry and then coat again with breadcrumbs, and keep aside.Shallow fry the tikkis in hot oil until golden both sides and take down.  Slice a burger bun into two horizontally, and toast it in a pan with some butter. Assemble the burger to suit your taste buds; begin with some chili sauce+ketchup in the bottom part of the bun, place some greens, then goes in the patty & salads, some more ketchup over top, and some garlic mayonnaise in the top bun. Cover & Serve.', 
'samosa': 'In a pan add ghee, ginger, garlic, green chillies and saute for a minute.Now add raisins, turmeric powder, degi red chilli powder, asafoetida, potatoes, green peas and mash it coarsely, mix everything properly and cover and cook for 4-5 minutes on medium heat.Now remove the cover and cook on high flames for 3-4 minutes or until light charred.Add the prepared masala, black pepper powder, dry Mango powder, salt and mix everything properly. And keep aside for further use.', 'frech fries': 'firstly, peel the skin of potato. recommend using maris piper potatoes as they have creamy white flesh and fluffy texture.cut into 1 cm thick sticks.rinse in ice cold water until the starch runs out clean.pat dry in kitchen towel to remove excess moisture.now deep fry in hot oil. make sure the oil is approximately 140 degree celcius.deep fry for 6 minutes or until the potatoes turn tender. they will not go brown at this stage.drain off over kitchen towel and cool completely. if you are looking to freeze the potatoes, then you can freeze the fries in zip lock bag upto 3 months.once they are cooled completely, deep fry in hot oil. make sure the oil is approximately 180 degree celcius.stir occasionally and fry until it turns golden brown and crisp.drain off to remove excess oil.now sprinkle ½ tsp chilli powder and ½ tsp salt. mix well.finally, enjoy homemade french fries with eggless mayonnaise as an evening snack.',
'grape': 'A grape is a fruit, botanically a berry, of the deciduous woody vines of the flowering plant genus Vitis. Grapes can be eaten fresh as table grapes or they can be used for making wine, jam, grape juice, jelly, grape seed extract, raisins, vinegar, and grape seed oil.', 'banana': 'Bananas are among the most important food crops on the planet. They come from a family of plants called Musa that are native to Southeast Asia and grown in many of the warmer areas of the world. Bananas are a healthy source of fiber, potassium, vitamin B6, vitamin C, and various antioxidants and phytonutrients'}

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
                
        return render_template('prediction.html',predicted_path = path,nutrition = nutrition,recipe=recipe[label_name])

    else:
        return render_template('error.html')


def imageProcessing(image):
    return darknet.getImageAndModel(image)


if __name__ == '__main__':
    app.run(debug=True,threaded=True)
