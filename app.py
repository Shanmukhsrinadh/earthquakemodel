from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
app = Flask(__name__)
app.secret_key = 'earthq'

# Read CSV file with space delimiter
df = pd.read_csv('Earthquake_Data.csv', delimiter=r'\s+')

new_column_names = ["Date(YYYY/MM/DD)",  "Time(UTC)", "Latitude(deg)", "Longitude(deg)", "Depth(km)", "Magnitude(ergs)", 
                    "Magnitude_type", "No_of_Stations", "Gap", "Close", "RMS", "SRC", "EventID"]

df.columns = new_column_names
ts = pd.to_datetime(df["Date(YYYY/MM/DD)"] + " " + df["Time(UTC)"])
df = df.drop(["Date(YYYY/MM/DD)", "Time(UTC)"], axis=1)
df.index = ts

from sklearn.model_selection import train_test_split

ty=ts.dt.time.sample().to_string(index=False)
print(ty[0:])

# Select relevant columns
X = df[['Latitude(deg)', 'Longitude(deg)', 'Depth(km)', 'No_of_Stations']]
y = df['Magnitude(ergs)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor

# Initialize a random forest regressor with 100 trees
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the regressor to the training data
rf.fit(X_train, y_train)



scores= {"Model name": [ "Random Forest"], "mse": [], "R^2": [], "mae":[]}

# Predict on the testing set
y_pred = rf.predict(X_test)

# Compute R^2 and MSE
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

scores['mse'].append(mse)
scores['R^2'].append(r2)
scores['mae'].append(mae)

#print("R^2: {:.2f}, MSE: {:.2f}, MAE: {:.2f}".format(r2, mse, mae))

print('Mean Squared Error: ', mse)
print('R^2 Score: ', r2)
print('Mean Absolute Error:', mae)




@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/predict')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        latitude = request.form.get('longitude')
        longitude = request.form.get('latitude')
        depth = request.form.get('depth')
        nos = request.form.get('nos')
        alert_message = "Input value is within the range (0 to -150)."
        
        cleaned_lal = ''.join(char for char in latitude if char.isdigit() or char == '.')
        cleaned_log = ''.join(char for char in longitude if char.isdigit() or char == '.')

        if not all(char.isdigit() or char == '.' for char in latitude):
            return render_template('index.html', alert_message=alert_message)
        elif not all(char.isdigit() or char == '.' for char in longitude):
            return render_template('index.html', alert_message=alert_message)
        

        if float(latitude) >= 0 and float(latitude) <= 150 and float(longitude) >= 0 and float(longitude )<= 150 :
            pass
        else:
            return render_template('index.html', alert_message=alert_message)      
        
        input_data = [[latitude, longitude, depth, nos ]]
        input_data = [[float(val) for val in input_data[0]]]  # Convert input data to float
        predicted_growth = rf.predict(input_data)

        print(depth)
        
        return render_template('result.html', predicted_growth=predicted_growth, mse=mse, r2=r2, mae=mae, ty=ty)
    
    return render_template('result.html')
 
 
 
 
if __name__ == '__main__':
    app.run(host='0.0.0.0')