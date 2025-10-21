from flask import Flask, render_template, request
from main_ml_script import get_recommendations_from_topic

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reccomedations')
def recomment():
    student_topic = request.form['topic']
    recommendations = get_recommendations_from_topic(student_topic)
    return render_template('results.html', topic=student_topic, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)