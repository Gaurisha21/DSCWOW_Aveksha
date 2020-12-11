from flask import Flask, render_template, redirect
from tweet import getSubjectivity
from tweet import getPolarity
from tweet import getAnalysis
from tweet import clean
from tweet import scraptweets
from tweet import get_tweets
app = Flask(__name__)

def requestResults(name):
    tweets = get_tweets(name)
    return tweets

@app.route('/')
def hello():
    return render_template("tutorial\Twitter_Today.html")

@app.route('/get', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = str(request.form['search'])
        ans = list(requestResults(user))
    return render_template("tutorial\Twitter_Today.html", Top_Users=ans)

if __name__ == '__main__':
    app.run(debug = False)