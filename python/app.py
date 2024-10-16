from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load your model
with open('basic_model_intents.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data from the request
    user_message = data['message']  # Get the user message
    # Here, use your model to generate a response based on user_message
    # For demonstration, we'll just return a mock response
    response = model.predict([user_message])  # Adjust according to your model's input requirements
    return jsonify({'response': response[0]})  # Return the prediction as JSON

if __name__ == '__main__':
    app.run(debug=True)
