from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import logging


def evaluate_model(model, X_test, y_test, model_name, visualization_dir):
    """
    Evaluate a model and return accuracy, report, and predictions.
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        logging.info(f"{model_name} Accuracy: {accuracy:.4f}")

        # Log additional metrics
        if 'weighted avg' in report:
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1_score = report['weighted avg']['f1-score']
            logging.info(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")

        return accuracy, report, y_pred

    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {str(e)}")
        # Return default values in case of error
        return 0.0, {}, np.array([])