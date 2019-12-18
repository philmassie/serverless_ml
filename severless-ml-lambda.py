import json
import pandas as pd
from pandas.io.json import json_normalize
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", -1)
import requests
import pickle
import sklearn
import boto3
import datetime as dt
from decimal import Decimal


def lambda_handler(event, context):
    # 01 Code start
    #---------------
    # get the city name from the query string
    try:
        city = event["queryStringParameters"]["city"]
    except:
        city = "Cape Town"
    
    print("city: " + city)

    # build a success string
    success = "<h1>City: " + city.title() + "</h1>"

    # 02 Code start
    #---------------
    # retrieve our API keys for LocationIQ and DarkSky
    # this is definitely not the best way to store keys. But these aren't too important so it doesn't matter
    keys=pd.read_csv("/opt/keys.csv")
    # keys = pd.read_csv("layers/03_rain_model/keys.csv") # local keys - put yours here
    lociq_api = keys["lociq_api"][0]
    darksky_api = keys["darksky_api"][0]
    
    print("LocationIQ API key: " + lociq_api)
    print("DarkSky API key: " + darksky_api)

    # use LocationIQ to do geolocation
    lociq = "https://eu1.locationiq.com/v1/search.php?key=" + lociq_api + "&q=" + city + "&format=json"
    lociq_info = requests.get(lociq)
    lat = json.loads(lociq_info.content)[0]["lat"]
    lon = json.loads(lociq_info.content)[0]["lon"]

    print("Lat: " + str(lat))
    print("Lon: " + str(lon))

    # build a success string
    success = success + "Latitude: " + str(lat) + "</br>Longitude: " + str(lon) + "</br>"

    # 03 Code start
    #---------------
    # Retrieve the current weather conditions from Dark Sky
    darksky = "https://api.darksky.net/forecast/" + darksky_api + "/" + str(lat) + "," + str(lon) + "?exclude=minutely,daily,hourly,alerts,flags"
    darksky_response = json.loads(requests.get(darksky).content)["currently"]
    weather_df = json_normalize(darksky_response, sep="_")

    print("Weather:")
    print(weather_df)
    
    # build a success string
    success = success + "<h2>Current weather</h2>" + weather_df.to_html() + "</br></br>"

    # 04 Code start
    #---------------
    # prep the data for the model, load the model and make a prediction
    x = weather_df[['apparentTemperature', 'cloudCover', 'dewPoint', 'humidity','ozone', 'pressure', 'temperature', 'uvIndex', 'visibility','windBearing', 'windGust', 'windSpeed']]

    # load the model from disk, do prediction
    filename = "/opt/models/rf_rain.pkl"
    # filename = "layers/03_rain_model/models/rf_rain.pkl" local copy
    rf_rain = pickle.load(open(filename, 'rb'))

    # predictions
    probs = rf_rain.predict_proba(x)
    p_rain = probs[0][1]

    pred = rf_rain.predict(x)
    pred_str = "No" if pred[0] == 0.0 else "Yes"
    
    print("Rain? " + pred_str)
    print("Rain probability: " + str(p_rain))

    # build a success string
    success = success + "Will it rain tomorrow? " + pred_str + "</br>probability of rain: " + str(p_rain) + "</br>"

    # 05 Code start
    #---------------
    # write something about the query to S3
    # prep the data we want to write
    # store the timestamp
    ts = int(weather_df["time"][0])
    # add predictions to df and make legible time
    weather_df["datetime"] = str(dt.datetime.fromtimestamp(weather_df["time"]))
    weather_df["prob_rain"] =  p_rain
    weather_df["pred_rain"] =  pred_str   

    # S3
    # keeping this mega simple. This will write a file with the name: timestamp_city.csv to your bucket.
    # overwites are possible and im not trying to append.
    # pandas and AWS make it very easy to write csv
    bucket_name = "severless-ml-tutorial"
    filename = str(ts) + "_" + city.lower().replace(" ", "_") + ".csv"
    weather_df.to_csv("s3://" + bucket_name + "/lambdas/" + filename, index=False)

    # build a success string
    success = success + "S3 write success? I think so...</br>"

    # send the output to the browsert
    return {
        'statusCode': 200,
        'body': success,
        "headers": {
            'Content-Type': 'text/html',
            }
        }