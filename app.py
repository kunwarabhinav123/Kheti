
from flask import Flask, render_template, request
import numpy as np
import requests
import config
import pickle



crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


fertilizer_recommendation_model_path = 'models/RandomForest1.pkl'
fertilizer_recommendation_model = pickle.load(
    open(fertilizer_recommendation_model_path, 'rb'))

col_transformer_model_path = 'models/col_transformer.pkl'
col_transformer_model = pickle.load(
    open(col_transformer_model_path, 'rb'))




def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None



app = Flask(__name__)


#home page
@app.route('/')
def home():
    title = 'Kheti - Home'
    return render_template('index.html', title=title)



#crop recommendation input and output page
@app.route('/crop-predict', methods=['POST','GET'])
def crop_prediction():
    title = 'Kheti - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)
    else:
        return render_template('crop.html', title=title)


#fertilizer recomendation input and output page
@app.route('/fertilizer-predict', methods=['POST','GET'])
def fertilizer_prediction():
    title = 'Kheti - Fertilizer Recommendation'

    if request.method=='POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        soil = str(request.form['soil type'])
        crop_name = str(request.form['cropname'])
        moisture = int(request.form['moisture'])
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data=[[temperature,humidity,moisture,soil,crop_name,N,P,K]]
            print(temperature,humidity,moisture,soil,crop_name,N,P,K)
            new_data=np.array(col_transformer_model.transform(data))
            print(new_data)
            my_prediction=fertilizer_recommendation_model.predict(new_data)
            final_prediction=my_prediction[0]
            print(final_prediction)
            return render_template('fertilizer-result.html', prediction=final_prediction, title=title)
        else:
            return render_template('try_again.html', title=title)
    else:
        return render_template('fertilizer.html', title=title)
        
        

@app.route('/disease-predict')
def disease_prediction():
    title = 'Kheti - Disease Detection'
    return render_template('disease.html', title=title)

#blog page
@app.route('/blog')
def blogs():
    title = 'Kheti - Blogs'
    return render_template('blogs.html', title=title)

#governmentscheme page
@app.route('/govermnetSchemes')
def govtsch():
    title = 'Kheti - Schemes'
    return render_template('govt.html', title=title)

#aboutus
@app.route('/AboutUS')
def about():
    title = 'Kheti - About Us'
    return render_template('about.html', title=title)



if __name__ == '__main__':
    app.secret_key = "12ddededd"
    app.run(debug=True)
