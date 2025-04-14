import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar el dataset
archivo = 'AirPassengers.csv'
df = pd.read_csv(archivo)

# Verificar columnas
print("Columnas del archivo:", df.columns)

# 2. Convertir 'Month' a datetime y extraer año y mes numérico
df['Month'] = pd.to_datetime(df['Month'])
df['Year'] = df['Month'].dt.year
df['Month_num'] = df['Month'].dt.month

# 3. Variables predictoras y objetivo
X = df[['Year', 'Month_num']]
y = df['#Passengers']

# 4. División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenamiento del modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# 6. Predicciones
y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)

# 7. Evaluación
print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)
print("Error cuadrático medio (prueba):", mean_squared_error(y_test, y_pred_test))
print("R² (prueba):", r2_score(y_test, y_pred_test))

# 8. Gráfico de entrenamiento
plt.figure(figsize=(8, 6))
plt.scatter(X_train['Year'] + X_train['Month_num']/12, y_train, color='blue', label='Datos reales')
plt.plot(X_train['Year'] + X_train['Month_num']/12, y_pred_train, color='red', label='Predicción')
plt.title('Entrenamiento')
plt.xlabel('Año')
plt.ylabel('Pasajeros')
plt.legend()
plt.tight_layout()
plt.show()  # Espera a que se cierre antes de continuar

# 9. Gráfico de prueba
plt.figure(figsize=(8, 6))
plt.scatter(X_test['Year'] + X_test['Month_num']/12, y_test, color='green', label='Datos reales')
plt.plot(X_test['Year'] + X_test['Month_num']/12, y_pred_test, color='orange', label='Predicción')
plt.title('Prueba')
plt.xlabel('Año')
plt.ylabel('Pasajeros')
plt.legend()
plt.tight_layout()
plt.show()
