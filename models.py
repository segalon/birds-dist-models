from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from catboost import CatBoostClassifier

from sklearn.utils.class_weight import compute_class_weight

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import r

pandas2ri.activate()

base = importr('base')
maxent = importr('dismo')


#DEBUG = True
DEBUG = False

SEED = 5


class CategoricalToString(BaseEstimator, TransformerMixin):
    def __init__(self, features_cat):
        self.features_cat = features_cat

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_cat = X[self.features_cat].astype(str)
        return X_cat


class ModelBird:
    def __init__(self, to_scale=True, to_ohe=False, drop_cats=[], cfg=None):
        self.to_scale = to_scale
        self.to_ohe = to_ohe
        self.drop_cats = drop_cats
        self.cfg = cfg
        self.features_cont = self.cfg['features_cont']
        self.features_cat = self.cfg['features_cat']
        self.threshold = self.cfg.get('threshold', 0.5)

        self.X_train = None
        self.y_train = None

        transformers = []

        if len(drop_cats) < len(self.features_cat):
            drop_cats = 'first'

        if self.to_scale:
            transformers.append((StandardScaler(), self.features_cont))

        if self.to_ohe:
            cat_to_str = CategoricalToString(self.features_cat)
            ohe = OneHotEncoder(handle_unknown='ignore', drop=drop_cats)
            transformers.append((make_pipeline(cat_to_str, ohe), self.features_cat))
        if len(transformers) == 0:
            self.transformer = None
        else:
            self.transformer = make_column_transformer(*transformers, remainder='passthrough')

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        pass


class ModelBirdLogisticRegression(ModelBird):
    def __init__(self, to_scale=True, to_ohe=False, cfg=None, drop_cats=[]):
        """
        Logistic regression model for bird species distribution.
        """
        if len(cfg['features_cat']) > 0:
            to_ohe = True
        self.to_ohe = to_ohe

        super().__init__(to_scale=to_scale, to_ohe=to_ohe, cfg=cfg, drop_cats=drop_cats)

    def fit(self, X, y):
        if DEBUG:
            X_orig = X.copy()
            print(X_orig.head())

        X = self.transformer.fit_transform(X)

        # Get column names from the transformer
        column_names = []
        for name, trans, columns in self.transformer.transformers_:
            if isinstance(trans, StandardScaler):
                column_names.extend(columns)
            elif isinstance(trans, Pipeline) and isinstance(trans.named_steps['onehotencoder'], OneHotEncoder):
                column_names.extend(trans.named_steps['onehotencoder'].get_feature_names_out(columns))

        # Create a dataframe with the correct column names
        X = pd.DataFrame(X, columns=column_names)

        X = sm.add_constant(X)

        self.model = Logit(y, X).fit()
        return self

    def predict_proba(self, X):
        """
        Predict the probability of bird presence using the logistic regression model.
        """
        X = self.transformer.transform(X)
        X = sm.add_constant(X)
        return self.model.predict(X)

    def predict(self, X):
        """
        Predict bird presence using the logistic regression model.
        """
        self.threshold = self.cfg.get('threshold', 0.5)
        return (self.predict_proba(X) > self.threshold).astype(int)


    def get_coefs_stats(self):
        """
        Get the coefficients and statistics of the logistic regression model.
        """
        return self.model

    def summary(self):
        """
        Get the summary of the logistic regression model.
        """
        return self.model.summary()


class ModelBirdCatBoost(ModelBird):
    def __init__(self, to_scale=False, to_ohe=False, cfg=None, drop_cats=[]):
        """
        Catboost model for bird species distribution.
        """
        # no need for scaling or one-hot encoding

        super().__init__(to_scale=to_scale, to_ohe=to_ohe, cfg=cfg, drop_cats=drop_cats)
        self.cat_features = self.cfg['features_cat']
        self.model = None
                            
    def fit(self, X, y):
        classes = np.unique(y)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y.ravel())

        class_weights = dict(zip(classes, weights))
        self.model = CatBoostClassifier(verbose=False,
                                        random_state=SEED,
                                        cat_features=self.cat_features,
                                        class_weights=class_weights)

        self.X_train = X
        self.y_train = y
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict bird presence using the catboost model.
        """
        threshold = self.cfg.get('threshold', 0.5)
        return (self.predict_proba(X) > threshold).astype(int)


    def predict_proba(self, X):
        """
        Predict the probability of bird presence using the catboost model.
        """
        return self.model.predict_proba(X)[:, 1]

    def summary(self):
        return self.model

    def plot_feature_importances(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_train)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, self.X_train, plot_type="dot", show=False)
        plt.tight_layout()
        return fig


class ModelBirdMaxEnt(ModelBird):
    def __init__(self, to_scale=True, to_ohe=True, cfg=None, drop_cats=[]):
        super().__init__(to_scale=to_scale, to_ohe=to_ohe, cfg=cfg, drop_cats=drop_cats)
        self.model_r = None
        self.fimps = None



    def fit(self, X, y):
        X = self.transformer.fit_transform(X)
        # Get column names from the transformer
        column_names = []
        for name, trans, columns in self.transformer.transformers_:
            if isinstance(trans, StandardScaler):
                column_names.extend(columns)
            elif isinstance(trans, Pipeline) and isinstance(trans.named_steps['onehotencoder'], OneHotEncoder):
                column_names.extend(trans.named_steps['onehotencoder'].get_feature_names_out(columns))

        self.X_with_colnames = pd.DataFrame(X, columns=column_names)

        # Create a dataframe with the correct column names
        X = pd.DataFrame(X)
        X_r = pandas2ri.py2rpy(X)
        y_r = robjects.IntVector(y)
        self.model = maxent.maxent(X_r, y_r)
        self.model_r = pandas2ri.py2rpy(self.model)
        return self


    def predict_proba(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.col_names)
        X = self.transformer.transform(X.copy())
        X = pd.DataFrame(X)
        X_r = pandas2ri.py2rpy(X)
        preds = robjects.r.predict(self.model_r, X_r, type="response")
        preds_py = np.array(preds)
        return preds_py


    def predict(self, X):
        """
        Predict bird presence using the maxent model.
        """
        threshold = self.cfg.get('threshold', 0.5)  
        preds = self.predict_proba(X)
        preds[preds >= threshold] = 1
        preds[preds < threshold] = 0
        return preds

    def summary(self):
        return self.model

    def plot_feature_importances(self):
        # todo get feature importances from maxent model itself
        fig, ax = plt.subplots()

        sns.barplot(x="importance", y="feature", data=self.df_fimp.sort_values(by="importance", ascending=False))

        plt.tight_layout()
        return fig
