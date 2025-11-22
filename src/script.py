import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_score


# FUNCIÓN 1: Preparar los datos 

def preparar_datos_explicit(df, features, target):
    """
    Convierte columnas a numérico, aplica imputación por mediana y separa X e y.
    """
    df2 = df.copy()

    # Convertir todas las columnas a tipo numérico
    for c in features + [target]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # Imputar valores faltantes en features
    imputer = SimpleImputer(strategy="median")
    df2[features] = imputer.fit_transform(df2[features])

    # Imputar faltantes en target
    df2[target] = df2[target].fillna(0)

    # X e y listos
    X = df2[features]
    y = df2[target]

    return X, y


# FUNCIÓN 2: Balanceo y división train/test

def balancear_y_dividir(X, y, test_size=0.20):
    """
    Calcula pesos balanceados para las clases y separa los datos en train/test.
    """
    # Calcular pesos de clases
    pesos = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )
    cw = {cls: peso for cls, peso in zip(np.unique(y), pesos)}

    # División train-test estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, cw
