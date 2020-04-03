from flask import Flask, render_template, url_for, request
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Loading Saved Model
    tfidf_vectorizer = pickle.load(open("pklFiles/tfidf_vectorizer.pkl", "rb"))
    svm_classifier = pickle.load(
        open("pklFiles/svm_classifier_for_tfidf_vectorizer.pkl", "rb"))


    if request.method == 'POST':
        user_headline = request.form['user_headline']
        user_headline = [user_headline]
        headline_count = tfidf_vectorizer.transform(user_headline)
        prediction = svm_classifier.predict(headline_count)
        return render_template('Result.html', prediction=prediction, headline=user_headline[0])


if __name__ == '__main__':
    app.run(debug=True)
