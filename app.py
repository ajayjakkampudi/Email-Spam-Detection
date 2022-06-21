import imp


import pickle
from flask import Flask,render_template,request,jsonify

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)