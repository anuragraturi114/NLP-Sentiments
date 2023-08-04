from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    newsgroups = fetch_20newsgroups(subset='all')
    categories = newsgroups.target_names

    train_data = newsgroups.data
    train_target = newsgroups.target

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_data)

    clf = MultinomialNB()
    clf.fit(X, train_target)

    text = [request.form['text']]
    X_text = vectorizer.transform(text)

    category = categories[clf.predict(X_text)[0]]

    return render_template('index.html', category=category)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
