import json
import os

from flask import Flask, request
from flask_cors import CORS

import LLM

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/get_functioncode", methods=["POST"])
def get_functioncode():
    request_body: dict[str, str] = request.json
    inputdata = str(request_body["inputdata"])
    instruction_code = LLM.get_function_openai(inputdata=inputdata)
    return json.dumps(instruction_code)


@app.route("/api/class_anomaly", methods=["POST"])
def classification_anomlay():
    request_body: dict[str, str] = request.json
    inputdata = str(request_body["inputdata"])
    error_class = LLM.classification_anomaly_openai(inputdata=inputdata)

    return json.dumps(error_class)


@app.route("/api/analyze_data", methods=["POST"])
def analyze_data():
    request_body: dict[str, str] = request.json
    inputdata = str(request_body["inputdata"])
    describe = LLM.analyze_data(inputdata=inputdata)
    return json.dumps(describe)


@app.route("/api/gptqa", methods=["POST"])
def gptqa():
    request_body: dict[str, str] = request.json
    query = str(request_body["query"])
    describe = LLM.gptqa(query=query)
    print(describe)
    return json.dumps(describe)


@app.route("/api/real_detection", methods=["POST"])
def real_detection():
    request_body: dict[str, str] = request.json
    inputdata = str(request_body["inputdata"])
    ori_log = str(request_body["ori_log"])
    describe = LLM.real_detection(inputdata=inputdata, ori_log=ori_log)
    print(describe)
    return json.dumps(describe)


if __name__ == "__main__":
    # dataset_path = "C:/Users/user/Desktop/2024HACK/ICSD_Cloud_Resource_Sample/"
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)