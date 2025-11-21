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

import statsmodels.api as sm
##FUNCION QUE PREPARA DATOS PARA EL MODELO
def prepare_platform_data(df, platform):
    dfp = df.copy()  # Copia del DF para no modificar el original

    # Columnas comunes si existen
    common = [c for c in ['Track Score','Release Year','Num_Artists','Explicit Track'] if c in dfp.columns]

    # Selección de target y features según la plataforma
    if platform=='spotify':
        target = 'Spotify Popularity'
        features = [c for c in ['Spotify Playlist Count','Spotify Playlist Reach','Spotify Streams'] if c in dfp.columns] + common

    elif platform=='youtube':
        target = 'YouTube Views'
        features = [c for c in ['YouTube Likes'] if c in dfp.columns] + common

    elif platform=='tiktok':
        target = 'TikTok Views'
        features = [c for c in ['TikTok Likes','TikTok Posts'] if c in dfp.columns] + common

    elif platform=='shazam':
        target = 'Shazam Counts'
        features = [c for c in ['Track Score'] if c in dfp.columns] + common

    else:
        return None, None, None  # Plataforma no válida

    # Validar que el target exista
    if target not in dfp.columns:
        return None, None, None

    # Convertir columnas relevantes a numérico
    for c in features + [target]:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors='coerce')

    # Eliminar filas donde el target esté vacío
    dfp = dfp[dfp[target].notna()]

    # Preparar X e y con medianas para completar valores faltantes
    X = dfp[features].fillna(dfp[features].median())
    y = dfp[target].fillna(dfp[target].median())

    return X, y, features  # Devolver datos listos para el modelo

def evaluar_modelos(X, y, nombre_plataforma, random_state=42):

    # Diccionario de modelos con sus pipelines (escalado + modelo)
    models = {
        'LinearRegression': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
        'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
        'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.001, max_iter=10000))]),
        'RandomForest': Pipeline([('model', RandomForestRegressor(n_estimators=200, random_state=random_state))])
    }

    resultados = []  # Lista para almacenar métricas

    # Separación train-test (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Entrenar y evaluar cada modelo.
    for nombre, modelo in models.items():
        modelo.fit(X_train, y_train)        # Entrena el modelo
        pred = modelo.predict(X_test)       # Predicciones del modelo

        # Métricas de evaluación
        rmse = np.sqrt(mean_squared_error(y_test, pred))  # Error cuadrático medio
        mae  = mean_absolute_error(y_test, pred)          # Error absoluto medio
        r2   = r2_score(y_test, pred)                     # Coeficiente R2

        resultados.append([nombre, rmse, mae, r2])        # Guardar resultado

    # Convertir resultados a DataFrame
    df_resultados = pd.DataFrame(resultados, columns=["Modelo","RMSE","MAE","R2"])

    print(f"\n--- Plataforma: {nombre_plataforma} | filas usadas: {X.shape[0]} ---")
    print(df_resultados.sort_values("R2", ascending=False))   # Mostrar ordenados por mejor R2


    # Seleccionar el mejor modelo
    best = df_resultados.sort_values("R2", ascending=False).iloc[0]
    print(f"\nMejor modelo para {nombre_plataforma}: {best['Modelo']} (R2={best['R2']:.4f})")

    return df_resultados  # Devolver tabla con métricas
