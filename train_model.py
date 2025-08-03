import os

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import logging
import numpy as np
from config import Config


def create_model_pipelines():
    """
    Create model pipelines with preprocessing for better performance variation.
    """
    models = {
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=Config.RANDOM_STATE))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(probability=True, random_state=Config.RANDOM_STATE))
        ]),
        'NeuralNetwork': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', MLPClassifier(max_iter=2000, random_state=Config.RANDOM_STATE))
        ]),
        'XGBoost': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(random_state=Config.RANDOM_STATE))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=Config.RANDOM_STATE))
        ]),
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=Config.RANDOM_STATE, max_iter=1000))
        ]),
        'NaiveBayes': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GaussianNB())
        ])
    }

    return models


def get_parameter_grids():
    """
    Define parameter grids for hyperparameter tuning with varied complexity.
    """
    param_grids = {
        'RandomForest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto', 0.001, 0.01]
        },
        'NeuralNetwork': {
            'classifier__hidden_layer_sizes': [(50,), (100,), (50, 30), (100, 50)],
            'classifier__learning_rate_init': [0.001, 0.01, 0.1],
            'classifier__alpha': [0.0001, 0.001, 0.01]
        },
        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__subsample': [0.8, 0.9, 1.0]
        },
        'KNN': {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        },
        'GradientBoosting': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7],
            'classifier__learning_rate': [0.01, 0.1, 0.2]
        },
        'LogisticRegression': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['liblinear', 'saga']
        },
        'NaiveBayes': {
            # NaiveBayes has fewer hyperparameters
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }

    return param_grids


def evaluate_model_cross_validation(model, X, y, cv_folds=5):
    """
    Evaluate model using cross-validation to get realistic performance estimates.
    """
    try:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)

        return mean_score, std_score, cv_scores

    except Exception as e:
        logging.error(f"Error in cross-validation: {str(e)}")
        return 0.0, 0.0, []


def train_and_fine_tune_models(X_train, y_train):
    """
    Train and fine-tune multiple models with realistic performance variation.
    """
    try:
        logging.info(f"Starting model training with {X_train.shape[0]} samples and {X_train.shape[1]} features")
        logging.info(f"Class distribution: {np.bincount(y_train)}")

        # Create models
        models = create_model_pipelines()
        param_grids = get_parameter_grids()

        best_models = {}
        model_scores = {}

        # Train each model
        for name, model in models.items():
            try:
                logging.info(f"Training {name}...")

                # Perform grid search with cross-validation
                if name in param_grids:
                    grid_search = GridSearchCV(
                        model,
                        param_grids[name],
                        cv=Config.CV_FOLDS,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_score = grid_search.best_score_

                    logging.info(f"{name} best CV score: {best_score:.4f}")
                    logging.info(f"{name} best parameters: {grid_search.best_params_}")

                else:
                    # For models without parameter grid, just fit normally
                    best_model = model
                    best_model.fit(X_train, y_train)

                    # Get cross-validation score
                    mean_score, std_score, cv_scores = evaluate_model_cross_validation(
                        best_model, X_train, y_train, Config.CV_FOLDS
                    )
                    best_score = mean_score

                    logging.info(f"{name} CV score: {mean_score:.4f} ± {std_score:.4f}")

                # Store model and score
                best_models[name] = best_model
                model_scores[name] = best_score

                # Save individual model
                model_path = f"{Config.MODELS_DIR}/{name}.pkl"
                joblib.dump(best_model, model_path)
                logging.info(f"Saved {name} model to {model_path}")

            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
                continue

        if not best_models:
            raise ValueError("No models were successfully trained")

        # Create ensemble from top performing models
        logging.info("Creating ensemble model...")

        # Select top models for ensemble (avoid overfitting with too many models)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        top_models = dict(sorted_models[:5])  # Use top 5 models

        logging.info(f"Top models for ensemble: {list(top_models.keys())}")

        # Create voting classifier
        ensemble_estimators = [(name, best_models[name]) for name in top_models.keys()]
        ensemble = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft'  # Use soft voting for probability-based decisions
        )

        # Train ensemble
        ensemble.fit(X_train, y_train)

        # Evaluate ensemble
        ensemble_mean_score, ensemble_std_score, _ = evaluate_model_cross_validation(
            ensemble, X_train, y_train, Config.CV_FOLDS
        )

        logging.info(f"Ensemble CV score: {ensemble_mean_score:.4f} ± {ensemble_std_score:.4f}")

        # Save ensemble
        ensemble_path = f"{Config.MODELS_DIR}/Ensemble.pkl"
        joblib.dump(ensemble, ensemble_path)
        logging.info(f"Saved ensemble model to {ensemble_path}")

        # Log final model performance summary
        logging.info("=== Model Training Summary ===")
        for name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"{name}: {score:.4f}")
        logging.info(f"Ensemble: {ensemble_mean_score:.4f}")

        return best_models, ensemble

    except Exception as e:
        logging.error(f"Error in model training pipeline: {str(e)}")
        raise


def save_model_metadata(best_models, ensemble, X_train, y_train):
    """
    Save metadata about trained models for later analysis.
    """
    try:
        metadata = {
            'training_samples': X_train.shape[0],
            'feature_count': X_train.shape[1],
            'class_count': len(np.unique(y_train)),
            'models_trained': list(best_models.keys()),
            'ensemble_components': len(ensemble.estimators_) if hasattr(ensemble, 'estimators_') else 0
        }

        metadata_path = f"{Config.MODELS_DIR}/training_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        logging.info(f"Saved training metadata to {metadata_path}")

    except Exception as e:
        logging.warning(f"Failed to save model metadata: {str(e)}")


def load_trained_models():
    """
    Load all trained models from disk.
    """
    try:
        import glob
        model_files = glob.glob(f"{Config.MODELS_DIR}/*.pkl")
        models = {}

        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('.pkl', '')
            if model_name != 'training_metadata' and model_name != 'label_encoder':
                try:
                    models[model_name] = joblib.load(model_file)
                except Exception as e:
                    logging.warning(f"Failed to load model {model_name}: {str(e)}")

        logging.info(f"Loaded {len(models)} models: {list(models.keys())}")
        return models

    except Exception as e:
        logging.error(f"Error loading trained models: {str(e)}")
        return {}