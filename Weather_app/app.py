from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('weather_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict_weather():
    if request.method == 'POST':
        # Get the input data from the form
        precipitation = float(request.form['precipitation'])
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        wind = float(request.form['wind'])
        
        # Create a DataFrame from the input data
        data = pd.DataFrame({
            'precipitation': [precipitation],
            'temp_max': [temp_max],
            'temp_min': [temp_min],
            'wind': [wind]
        })
        
        # Make predictions using the pre-trained model
        predicted_weather = model.predict(data)
        
        # Render the results template with the predicted weather
        return render_template('results.html', predicted_weather=predicted_weather[0])
    
    # Render the input form template for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
