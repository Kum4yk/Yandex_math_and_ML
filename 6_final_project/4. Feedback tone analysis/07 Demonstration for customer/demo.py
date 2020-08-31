from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import Length, DataRequired
from sentiment_classifier import SentimentClassifier
import time


app = Flask(__name__)
app.config['SECRET_KEY'] = "secret_key"
bootstrap = Bootstrap(app)
print("Preparing classifier")
start_time = time.time()
classifier = SentimentClassifier()
print("Classifier is ready")
print(time.time() - start_time, "seconds")


class NameForm(FlaskForm):
    name = TextAreaField("Отзыв", validators=[DataRequired(), Length(1, 5000)])
    submit = SubmitField('Оценить')


@app.route('/', methods=['GET', 'POST'])
def index():
    name = None
    form = NameForm()
    if form.validate_on_submit():
        text = form.name.data
        name = classifier.get_prediction_message(text)
    return render_template('index.html', form=form, name=name)


if __name__ == '__main__':
    app.run(debug=False, port=5050)
