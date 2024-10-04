## Algoritmos paralelos y distribuidos:
## Guia05-Aprendizaje Estadístico
## Nombre: Efrain vitorino marin 



## Ejercicio 1: Principales Modelos Lineales

### 1. Regresión Bayesiana

La **Regresión Bayesiana** es un enfoque probabilístico de la regresión lineal. A diferencia de la regresión lineal clásica, que calcula estimaciones fijas para los parámetros, la regresión bayesiana utiliza distribuciones de probabilidad para modelar la incertidumbre en los parámetros.

#### Ejemplo de implementación:

```python
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generar un conjunto de datos para regresión
datosX, datosY = make_regresion(nSamples=100, nFeatures=2, noise=0.1, randomState=42)

# Dividir los datos en entrenamiento y prueba
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(datosX, datosY, testSize=0.2, randomState=42)

# Crear el modelo de Regresión Bayesiana
modeloBayesiano = BayesianRidge()

# Entrenar el modelo
modeloBayesiano.fit(xEntrenamiento, yEntrenamiento)

# Predecir en los datos de prueba
yPrediccion = modeloBayesiano.predict(xPrueba)

# Mostrar los coeficientes y la puntuación R²
print(f"Coeficientes: {modeloBayesiano.coef_}")
print(f"Intercepto: {modeloBayesiano.intercept_}")
print(f"Puntuacion R²: {modeloBayesiano.score(xPrueba, yPrueba)}")
```

### 2. Regresión Logística

La **Regresión Logística** es un modelo utilizado principalmente para problemas de clasificación binaria. En lugar de predecir valores continuos, la regresión logística predice probabilidades, lo que permite determinar la clase a la que pertenece una instancia.

#### Ejemplo de implementación:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
iris = load_iris()
datosX, datosY = iris.data, iris.target

# Dividir los datos en conjunto de entrenamiento y prueba
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(datosX, datosY, testSize=0.2, randomState=42)

# Crear el modelo de Regresión Logística
modeloLogistico = LogisticRegression(maxIter=200)

# Entrenar el modelo
modeloLogistico.fit(xEntrenamiento, yEntrenamiento)

# Predecir en los datos de prueba
yPrediccion = modeloLogistico.predict(xPrueba)

# Evaluar la precisión
precision = modeloLogistico.score(xPrueba, yPrueba)
print(f"Precision del modelo: {precision * 100:.2f}%")
```

### 3. Descenso de Gradiente Estocástico (SGD)

El **Descenso de Gradiente Estocástico (SGD)** es una técnica de optimización que ajusta los parámetros del modelo de forma iterativa utilizando una muestra aleatoria en cada paso.

#### Ejemplo de implementación:

```python
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generar un conjunto de datos para clasificación
datosX, datosY = makeClasificacion(nSamples=1000, nFeatures=20, randomState=42)

# Dividir los datos en conjunto de entrenamiento y prueba
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(datosX, datosY, testSize=0.2, randomState=42)

# Crear el modelo de SGD
modeloSGD = SGDClassifier(maxIter=1000, tol=1e-3)

# Entrenar el modelo
modeloSGD.fit(xEntrenamiento, yEntrenamiento)

# Predecir en los datos de prueba
yPrediccion = modeloSGD.predict(xPrueba)

# Evaluar la precisión del modelo
precision = modeloSGD.score(xPrueba, yPrueba)
print(f"Precision del modelo SGD: {precision * 100:.2f}%")
```

### 4. Perceptrón

El **Perceptrón** es un modelo simple de clasificación supervisada que se utiliza para problemas de clasificación binaria.

#### Ejemplo de implementación:

```python
from sklearn.linear_model import Perceptron

# Generar un conjunto de datos para clasificación
datosX, datosY = makeClasificacion(nSamples=1000, nFeatures=20, randomState=42)

# Dividir los datos en conjunto de entrenamiento y prueba
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(datosX, datosY, testSize=0.2, randomState=42)

# Crear el modelo de Perceptrón
modeloPerceptron = Perceptron(maxIter=1000, tol=1e-3)

# Entrenar el modelo
modeloPerceptron.fit(xEntrenamiento, yEntrenamiento)

# Predecir en los datos de prueba
yPrediccion = modeloPerceptron.predict(xPrueba)

# Evaluar la precisión del modelo
precision = modeloPerceptron.score(xPrueba, yPrueba)
print(f"Precision del modelo Perceptron: {precision * 100:.2f}%")
```

---


# Informe sobre Modelos Lineales y K-Nearest Neighbors (KNN)

Este informe explora algunos de los principales modelos lineales utilizando la biblioteca **Scikit-learn** y el algoritmo **K-Nearest Neighbors (KNN)**. Aquí se detallan las características, ventajas, desventajas, y ejemplos de implementación de cada uno de ellos.

## Ejercicio 1: Principales Modelos Lineales

### 1. Regresión Bayesiana

La **Regresión Bayesiana** es un enfoque probabilístico de la regresión lineal. A diferencia de la regresión lineal clásica, que calcula estimaciones fijas para los parámetros, la regresión bayesiana utiliza distribuciones de probabilidad para modelar la incertidumbre en los parámetros.

#### Ejemplo de implementación:

```python
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generar un conjunto de datos para regresión
datosX, datosY = make_regresion(nSamples=100, nFeatures=2, noise=0.1, randomState=42)

# Dividir los datos en entrenamiento y prueba
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(datosX, datosY, testSize=0.2, randomState=42)

# Crear el modelo de Regresión Bayesiana
modeloBayesiano = BayesianRidge()

# Entrenar el modelo
modeloBayesiano.fit(xEntrenamiento, yEntrenamiento)

# Predecir en los datos de prueba
yPrediccion = modeloBayesiano.predict(xPrueba)

# Mostrar los coeficientes y la puntuación R²
print(f"Coeficientes: {modeloBayesiano.coef_}")
print(f"Intercepto: {modeloBayesiano.intercept_}")
print(f"Puntuacion R²: {modeloBayesiano.score(xPrueba, yPrueba)}")
```

### 2. Regresión Logística

La **Regresión Logística** es un modelo utilizado principalmente para problemas de clasificación binaria. En lugar de predecir valores continuos, la regresión logística predice probabilidades, lo que permite determinar la clase a la que pertenece una instancia.

#### Ejemplo de implementación:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Cargar el conjunto de datos Iris
iris = load_iris()
datosX, datosY = iris.data, iris.target

# Dividir los datos en conjunto de entrenamiento y prueba
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(datosX, datosY, testSize=0.2, randomState=42)

# Crear el modelo de Regresión Logística
modeloLogistico = LogisticRegression(maxIter=200)

# Entrenar el modelo
modeloLogistico.fit(xEntrenamiento, yEntrenamiento)

# Predecir en los datos de prueba
yPrediccion = modeloLogistico.predict(xPrueba)

# Evaluar la precisión
precision = modeloLogistico.score(xPrueba, yPrueba)
print(f"Precision del modelo: {precision * 100:.2f}%")
```

### 3. Descenso de Gradiente Estocástico (SGD)

El **Descenso de Gradiente Estocástico (SGD)** es una técnica de optimización que ajusta los parámetros del modelo de forma iterativa utilizando una muestra aleatoria en cada paso.

#### Ejemplo de implementación:

```python
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generar un conjunto de datos para clasificación
datosX, datosY = makeClasificacion(nSamples=1000, nFeatures=20, randomState=42)

# Dividir los datos en conjunto de entrenamiento y prueba
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(datosX, datosY, testSize=0.2, randomState=42)

# Crear el modelo de SGD
modeloSGD = SGDClassifier(maxIter=1000, tol=1e-3)

# Entrenar el modelo
modeloSGD.fit(xEntrenamiento, yEntrenamiento)

# Predecir en los datos de prueba
yPrediccion = modeloSGD.predict(xPrueba)

# Evaluar la precisión del modelo
precision = modeloSGD.score(xPrueba, yPrueba)
print(f"Precision del modelo SGD: {precision * 100:.2f}%")
```

### 4. Perceptrón

El **Perceptrón** es un modelo simple de clasificación supervisada que se utiliza para problemas de clasificación binaria.

#### Ejemplo de implementación:

```python
from sklearn.linear_model import Perceptron

# Generar un conjunto de datos para clasificación
datosX, datosY = makeClasificacion(nSamples=1000, nFeatures=20, randomState=42)

# Dividir los datos en conjunto de entrenamiento y prueba
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(datosX, datosY, testSize=0.2, randomState=42)

# Crear el modelo de Perceptrón
modeloPerceptron = Perceptron(maxIter=1000, tol=1e-3)

# Entrenar el modelo
modeloPerceptron.fit(xEntrenamiento, yEntrenamiento)

# Predecir en los datos de prueba
yPrediccion = modeloPerceptron.predict(xPrueba)

# Evaluar la precisión del modelo
precision = modeloPerceptron.score(xPrueba, yPrueba)
print(f"Precision del modelo Perceptron: {precision * 100:.2f}%")
```

---

## Ejercicio 2: Algoritmo K-Nearest Neighbors (KNN)

El **K-Nearest Neighbors (KNN)** es un algoritmo supervisado de clasificación que clasifica un punto de datos basándose en los k vecinos más cercanos de ese punto.

#### Ejemplo de implementación:

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos Iris
iris = load_iris()
datosX, datosY = iris.data, iris.target

# Dividir los datos en conjunto de entrenamiento y prueba
xEntrenamiento, xPrueba, yEntrenamiento, yPrueba = train_test_split(datosX, datosY, testSize=0.2, randomState=42)

# Crear el modelo KNN con 3 vecinos
modeloKNN = KNeighborsClassifier(nNeighbors=3)

# Entrenar el modelo
modeloKNN.fit(xEntrenamiento, yEntrenamiento)

# Predecir en los datos de prueba
yPrediccion = modeloKNN.predict(xPrueba)

# Evaluar la precisión del modelo
precision = modeloKNN.score(xPrueba, yPrueba)
print(f"Precision del modelo KNN: {precision * 100:.2f}%")
```

## Conclusión

Este informe explora los modelos lineales más utilizados, como la **Regresión Bayesiana**, **Regresión Logística**, **SGD**, y **Perceptrón**, así como el algoritmo de clasificación no paramétrico **K-Nearest Neighbors (KNN)**. Todos estos modelos tienen sus propias ventajas y desventajas, y son adecuados para distintos tipos de problemas.
