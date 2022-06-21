import pickle
from unittest import result
from flask import Flask,render_template,request,jsonify
from sklearn.feature_extraction.text import CountVectorizer

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
data=pickle.load(open('data.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        C = CountVectorizer(max_features=1000)
        data.append(message)
        Q = C.fit_transform(data).toarray()
        prediction_value=model.predict(Q[-1].reshape(1,-1))
        result=''
        if prediction_value[0]==1:
            result='ham'
        elif prediction_value[0]==0:
            result='spam'
        return render_template('index.html',prediction='The message is {}'.format(result))

if __name__=="__main__":
    app.run(debug=True)