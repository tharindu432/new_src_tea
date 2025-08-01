from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import logging
import os
from config import Config

# Import XGBoost with error handling
try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")


def train_and_fine_tune_models(X_train, y_train):
    """
    Train and fine-tune models, returning best models and ensemble.
    """
    try:
        # Ensure models directory exists
        os.makedirs(Config.MODELS_DIR, exist_ok=True)

        # Define base models
        models = {
            'RandomForest': RandomForestClassifier(random_state=Config.RANDOM_STATE, n_jobs=-1),
            'SVM': SVC(probability=True, random_state=Config.RANDOM_STATE),
            'NeuralNetwork': MLPClassifier(max_iter=1000, random_state=Config.RANDOM_STATE, early_stopping=True)
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(random_state=Config.RANDOM_STATE, eval_metric='logloss')

        # Define parameter grids
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'NeuralNetwork': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001]
            }
        }

        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }

        best_models = {}
        failed_models = []

        # Train each model with grid search
        for name, model in models.items():
            try:
                logging.info(f"Training {name} model...")

                # Use fewer CV folds if dataset is small
                cv_folds = min(Config.CV_FOLDS, len(X_train) // 2)
                if cv_folds < 2:
                    cv_folds = 2

                grid_search = GridSearchCV(
                    model,
                    param_grids[name],
                    cv=cv_folds,
                    n_jobs=-1 if name != 'NeuralNetwork' else 1,  # Neural networks don't parallelize well
                    scoring='accuracy',
                    verbose=1
                )

                grid_search.fit(X_train, y_train)
                best_models[name] = grid_search.best_estimator_

                # Save model
                model_path = os.path.join(Config.MODELS_DIR, f"{name}.pkl")
                joblib.dump(best_models[name], model_path)

                logging.info(f"✓ {name} - Best score: {grid_search.best_score_:.4f}")
                logging.info(f"✓ {name} - Best params: {grid_search.best_params_}")
                logging.info(f"✓ Saved {name} model to {model_path}")

            except Exception as e:
                logging.error(f"✗ Error training {name}: {str(e)}")
                failed_models.append(name)
                continue

        if not best_models:
            raise ValueError("All models failed to train. Check your data and parameters.")

        if failed_models:
            logging.warning(f"Failed to train: {failed_models}")

        # Create ensemble from successfully trained models
        if len(best_models) >= 2:
            try:
                logging.info("Creating ensemble model...")
                ensemble_estimators = [(name, model) for name, model in best_models.items()]
                ensemble = VotingClassifier(estimators=ensemble_estimators, voting='soft', n_jobs=-1)
                ensemble.fit(X_train, y_train)

                # Save ensemble
                ensemble_path = os.path.join(Config.MODELS_DIR, "Ensemble.pkl")
                joblib.dump(ensemble, ensemble_path)
                logging.info(f"✓ Saved Ensemble model to {ensemble_path}")

            except Exception as e:
                logging.error(f"✗ Error creating ensemble: {str(e)}")
                ensemble = None
        else:
            logging.warning("Not enough models for ensemble (need at least 2)")
            ensemble = None

        logging.info(f"Successfully trained {len(best_models)} models")
        return best_models, ensemble

    except Exception as e:
        logging.error(f"Error in model training pipeline: {str(e)}")
        raise


def load_model(model_name):
    """
    Load a saved model from disk.
    """
    try:
        model_path = os.path.join(Config.MODELS_DIR, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        logging.info(f"Loaded {model_name} model from {model_path}")
        return model

    except Exception as e:
        logging.error(f"Error loading model {model_name}: {str(e)}")
        raise


def get_model_info(model_name):
    """
    Get information about a trained model.
    """
    try:
        model = load_model(model_name)

        info = {
            'model_type': type(model).__name__,
            'parameters': model.get_params() if hasattr(model, 'get_params') else 'Not available'
        }

        # Add model-specific info
        if hasattr(model, 'feature_importances_'):
            info['has_feature_importance'] = True
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_

        return info

    except Exception as e:
        logging.error(f"Error getting model info for {model_name}: {str(e)}")
        return None