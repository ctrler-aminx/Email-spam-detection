Email Spam Detection
This project is a machine learning-based system that classifies emails as either spam or not spam using different models. It uses natural language processing (NLP) techniques to process email text and applies various machine learning classifiers to make predictions.

Three different classifiers are used:

Naive Bayes – A simple and fast probabilistic model.
Support Vector Machine (SVM) – A more accurate but slightly slower model.
Random Forest Classifier – An ensemble approach using decision trees.
Email text is cleaned by removing unnecessary characters, punctuation, and common words (stop words) that don’t add much meaning.

TF-IDF vectorization converts email text into numerical data for machine learning models.

Flask-based Web Interface is used to input email messages and get real-time classification.

The model is trained on a dataset (spam.csv) containing labeled emails marked as spam or not spam. The labels are represented as:

1 for spam
0 for not spam
The project includes a Flask web interface:

Users input an email message into a text box.
The message is preprocessed, cleaned, and converted into a numerical form.
The trained SVM model predicts whether the message is spam or not.
The result is displayed on the webpage.
Technologies used:

Python
Pandas, NumPy (Data Handling)
NLTK, Scikit-learn (Text Processing & ML Models)
Flask (Web Framework)
This project effectively detects spam emails using multiple classifiers and provides a simple Flask-based web interface for testing. The SVM model performed the best, achieving high accuracy.