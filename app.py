from flask import Flask, jsonify
from flask_cors import CORS

from datetime import datetime
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

MODEL_FOLDER = "model/"


def random_forest():
    dataset = pd.read_csv('dataset/DATASET_TRAINING_TEKNIK_INDUSTRI.csv')

    X = dataset.drop(columns=['RESPONDEN', 'CLASS'])
    y = dataset['CLASS']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = RandomForestClassifier(
        n_estimators=500, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)

    now = datetime.now()

    filename = "random_forest_" + now.strftime("%Y%m%d%H%M%S")

    joblib.dump(classifier, os.path.join(MODEL_FOLDER, filename))

    print('done')
    # y_pred = classifier.predict(X_test)
    return filename


def predict():
    latest_file = ""

    dir = os.listdir(MODEL_FOLDER)

    for file in dir:
        latest_file = file

    model = joblib.load(os.path.join(MODEL_FOLDER, latest_file))

    print(model)

    # model.predict()


@app.route("/train")
def train():
    filename = random_forest()

    data = {
        "code": 200,
        "status": "ok",
        "data": {
            "filename": filename,
            "path": MODEL_FOLDER
        }
    }

    return jsonify(data), 200


@app.route("/")
def index():
    data = {
        "code": 200,
        "status": "ok",
        "data": {
            "message": "berhasil terkoneksi ke server!"
        }
    }

    return jsonify(data), 200


if __name__ == "__main__":
    app.run()
