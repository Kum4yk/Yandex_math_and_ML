import joblib
import numpy as np


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("data/pipeline_model.pkl")
        self.classes_dict = {0: "отрицательный", 1: "положительный", -1: "prediction error"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "не совсем"
        if probability < 0.75:
            return "скорее всего"
        if probability < 0.85:
            return "почти"
        else:
            return "однозначно"

    def predict_text(self, text: str):
        if not isinstance(text, list) or not isinstance(text, np.ndarray):
            text = [text]
        pred_class = self.model.predict(text)[0]
        prob_pred_class = self.model.predict_proba(text)[0][pred_class]
        return pred_class, prob_pred_class

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]

        percent = round(prediction_probability * 100, 2)
        word = self.get_probability_words(prediction_probability).capitalize()
        pred_class = self.classes_dict[class_prediction]

        return f"{word} уверен ({percent}%), что данный отзыв {pred_class}."


if __name__ == "__main__":
    check = SentimentClassifier()
    test_text = "очень плохо"
    print(check.predict_text(test_text))
    print(check.get_prediction_message(test_text))

