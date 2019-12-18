import json
import datetime as dt
import requests
import pytz
import pandas as pd
from pandas.io.json import json_normalize
import pickle

# API keys
keys=pd.read_csv("model_training/keys.csv")
google_api = keys["google_api"][0]
darksky_api = keys["darksky_api"][0]

# location
google = "https://maps.googleapis.com/maps/api/geocode/json?address=Cape Town&key=" + google_api
resp_loc = requests.get(google)
lat = json.loads(resp_loc.content)["results"][0]["geometry"]["location"]["lat"]
lon = json.loads(resp_loc.content)["results"][0]["geometry"]["location"]["lng"]

# weather
dates = [str(int((dt.datetime.now(pytz.utc) - dt.timedelta(days=x)).timestamp())) for x in range(365)]
weather_df = pd.DataFrame()
i = 0
for d in dates:
    print(str(i) + ": " + str(d))
    darksky = "https://api.darksky.net/forecast/" + darksky_api + "/" + str(lat) + "," + str(lon) + "," + d + "?exclude=hourly,alerts,flags"
    weather_df = weather_df.append(json_normalize(json.loads(requests.get(darksky).content)["currently"]))
    i += 1

# weather_df.to_pickle("/Users/phil/vscode/weather_lambda/model_training/data/weather_df_raw.pkl")
weather_df = pd.read_pickle("/Users/phil/vscode/weather_lambda/model_training/data/weather_df_raw.pkl")

# datetime and order
weather_df = weather_df.reset_index()
weather_df["datetime"] = [dt.datetime.fromtimestamp(weather_df["time"][i]) for i in range(0, len(weather_df))]
weather_df = weather_df.sort_values(by=["datetime"])

# take a look
weather_df.dtypes
weather_df["precipIntensity"]
weather_df["precipProbability"]
weather_df["precipType"].unique()

weather_df["rain"] = 0
weather_df.loc[weather_df["precipType"]=="rain", ["rain"]] = 1
weather_df["rain"].unique()

weather_df.columns
weather_df = weather_df[['datetime', 'apparentTemperature', 'cloudCover', 'dewPoint', 
    'humidity','ozone', 'pressure', 'temperature', 'uvIndex', 'visibility',
    'windBearing', 'windGust', 'windSpeed', 'rain']]

# build the label
weather_df["rain"] = weather_df["rain"].shift(periods=-1)
weather_df = weather_df.dropna()

# weather_df.to_pickle("model_training/data/weather_df_proc.pkl")
weather_df = pd.read_pickle("model_training/data/weather_df_proc.pkl")

# train a simple model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# prep the data
weather_df.columns
X = weather_df[['apparentTemperature', 'cloudCover', 'dewPoint', 'humidity','ozone', 'pressure', 'temperature', 'uvIndex', 'visibility','windBearing', 'windGust', 'windSpeed']]
y = weather_df['rain']

# split test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up a little grid
rf = RandomForestClassifier(n_jobs="-1")
tuned_parameters = {
    'n_estimators': [10, 20, 30, 40, 50, 100, 200, 500], 
    'max_depth': [8, 9, 10, 11, 12, 13, 14]}
clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,scoring="accuracy")
clf.fit(X_train, y_train)
# GridSearchCV(cv=5, error_score='raise-deprecating',
#              estimator=RandomForestClassifier(bootstrap=True, class_weight=None,
#                                               criterion='gini', max_depth=None,
#                                               max_features='auto',
#                                               max_leaf_nodes=None,
#                                               min_impurity_decrease=0.0,
#                                               min_impurity_split=None,
#                                               min_samples_leaf=1,
#                                               min_samples_split=2,
#                                               min_weight_fraction_leaf=0.0,
#                                               n_estimators='warn', n_jobs=None,
#                                               oob_score=False,
#                                               random_state=None, verbose=0,
#                                               warm_start=False),
#              iid='warn', n_jobs=None,
#              param_grid={'max_depth': [8, 9, 10, 11, 12, 13, 14],
#                          'n_estimators': [10, 20, 30, 40, 50, 100, 200, 500]},
#              pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
#              scoring='accuracy', verbose=0)
             
# save the model to disk
filename = "model_training/model/rf_rain.pkl"
pickle.dump(clf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

print("Best parameters set found on development set:")
print()
print(loaded_model.best_params_)
print()
print("Grid scores on development set:")
print()
means = loaded_model.cv_results_['mean_test_score']
stds = loaded_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
            % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, loaded_model.predict(X_test)
print(classification_report(y_true, y_pred))
print()
probs = loaded_model.predict_proba(X_test.iloc[0])
# write the model to lambda layers directory
filename = "layers/03_rain_model/models/rf_rain.pkl"
pickle.dump(clf, open(filename, 'wb'))