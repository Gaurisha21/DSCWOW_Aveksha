from flask import Flask, render_template, redirect, request
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
    return render_template("Twitter_Today.html")

@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        
        user = str(request.form['city'])
        ans = list(requestResults(user))
        
        print(ans)
    return render_template("Twitter_Today.html", Top_Users=ans)

if __name__ == '__main__':
    app.run(debug = False)