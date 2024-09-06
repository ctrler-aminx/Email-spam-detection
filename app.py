from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form['message']  # Get user input
        message_vectorized = vectorizer.transform([message])  # Convert text to numerical features
        prediction = model.predict(message_vectorized)[0]  # Predict spam or ham
        return render_template('index.html', result="Spam" if prediction == 1 else "Not Spam")
    return render_template('index.html', result="")

if __name__ == "__main__":
    app.run(debug=True)
