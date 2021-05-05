# Calorie_detector
Flask PWA to classify food items and displays their nutritional values

The app lets the user take an image of any food they consume on a daily basis and immediately calculates calories of it. Also, the app can be extended to kep track of daily consumed food items and their calories so that it can recommend a highly balanced daily diet to the user. Users can set goals for daily calories to lose or gain weight and the app helps them to keep up with their goals.

##Working 

## Models

1. **ImageClassification:**
	* First model which takes image data as `224 x 224` array and process it to generate the probability of any food in the image
	* The model consists of 8 layers including input and output layers
	* To implement the model **tflearn** library is used
	* The accuracy of this CNN is 92.37%.

2. **YOLO**:
	* **YOLO** stands for **You Only Look for Once** and it is a famous **Object Detection** model 
	* What it means by **YOLO** is that this model scans the whole image just once to get all the objects that the image contains
	* Then it generates the probability of all objects in the image and determines whether any particular object is present in that or not
	* This model is trained on a dataset of 7 food labels and the parameters have been adjusted to achieve the desired accuracy
	* To know more about this model [click here](https://pjreddie.com/darknet/yolov2/ "YOLO")
	* The accuracy of predicting object ranges between 70% to 95%.
	
##Usage

Newest python and pip is prerequisite

```
pip install -r requirements.txt
pip install virtualenv
virtualenv venv
```
To activate virtual environment - 
Mac/linux
```
source venv/bin/activate
```
Windows
```
venv\Scripts\activate
```
To start the application -

Mac/linux (supports gunicorn)
```
export FLASK_APP=app
gunicorn --bind 127.0.0.1:5000 wsgi:app
```

Windows
```
set FLASK_APP=app
flask run
```

The application will start on localhost:5000
