from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import logging

def evaluate_model(model, X_test, y_test, model_name, visualization_dir):
    """
    Evaluate a model and return accuracy, report, and predictions.
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        logging.info(f"{model_name} Accuracy: {accuracy:.4f}")
        return accuracy, report, y_pred
    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {str(e)}")
        raise