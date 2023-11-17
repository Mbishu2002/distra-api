from flask import Flask, request, render_template
from flask_restful import Api, Resource
import torch
from joblib import load
from flask_cors import CORS

app = Flask(__name__)
api = Api(app)
CORS(app)
model = load('/mode/distra.joblib')
model.eval()

class Predict(Resource):
    def post(self):
        try:
            data = request.get_json()
            input_data = torch.tensor(data['input_data'])

            with torch.no_grad():
                output = model(input_data)
                prediction = output.item()

            return {'prediction': prediction}

        except Exception as e:
            return {'error': str(e)}

api.add_resource(Predict, '/predict')

@app.route('/')
def documentation():
    return render_template('app.html')

if __name__ == '__main__':
    app.run(debug=True)
