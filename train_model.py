from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import logging
from config import Config


def train_and_fine_tune_models(X_train, y_train):
    """
    Train and fine-tune models, returning best models and ensemble.
    """
    try:
        models = {
            'RandomForest': RandomForestClassifier(random_state=Config.RANDOM_STATE),
            'SVM': SVC(probability=True, random_state=Config.RANDOM_STATE),
            'NeuralNetwork': MLPClassifier(max_iter=1000, random_state=Config.RANDOM_STATE),
            'XGBoost': XGBClassifier(random_state=Config.RANDOM_STATE)
        }

        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10]
            },
            'SVM': {
                'C': [1, 10],
                'kernel': ['rbf']
            },
            'NeuralNetwork': {
                'hidden_layer_sizes': [(50,), (100,)],
                'learning_rate_init': [0.001, 0.01]
            },
            'XGBoost': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5]
            }
        }

        best_models = {}
        for name, model in models.items():
            try:
                grid_search = GridSearchCV(model, param_grids[name], cv=Config.CV_FOLDS, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                best_models[name] = grid_search.best_estimator_
                joblib.dump(best_models[name], f"{Config.MODELS_DIR}/{name}.pkl")
                logging.info(f"Trained and saved {name} model")
            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                continue

        # Ensemble model
        ensemble = VotingClassifier(estimators=[(name, model) for name, model in best_models.items()], voting='soft')
        ensemble.fit(X_train, y_train)
        joblib.dump(ensemble, f"{Config.MODELS_DIR}/Ensemble.pkl")
        logging.info("Trained and saved Ensemble model")

        return best_models, ensemble
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise