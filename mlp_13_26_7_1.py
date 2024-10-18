import numpy as np

from graficar import graficar_error_tiempo as graficar_et
from data import y_train_scaled, y_test_scaled
from data import X_train_scaled, X_test_scaled

def derivada_relu(arr):
    return np.where(arr > 0, 1, 0)

def calcular_ECM(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


X_train = X_train_scaled
y_train = y_train_scaled

X_test = X_test_scaled
y_test = y_test_scaled

#Definir una 13+1 26+1 7+1 1

#Pesos 109304
np.random.seed(2**4)
W1 = np.random.rand(27, 14)*0.1
W2 = np.random.rand(8, 27)*0.1
W3 = np.random.rand(1, 8) *0.1

epocas = 200
error_test = np.empty((0,))
error_tiempo = np.empty((0,))
bias = 1
tasa_aprendizaje = 0.01
regularizacion = "L2"
lambda_reg = 0.0001

for epoca in range(epocas):

    y_pred_train=np.empty((0,))

    #ENTRENAMIENTO
    for muestra in range(len(y_train)):

        patron = X_train[muestra][:]
        patron = np.append(patron,bias)
        objetivo = y_train[muestra]

        #Propagación hacia adelante
        suma1 = np.dot(W1,patron)
        suma1[-1] = bias
        z1 = np.maximum(0,suma1) #RELU
        z1[-1] = bias

        suma2 = np.dot(W2,z1)
        suma2[-1] = bias
        z2 = np.maximum(0,suma2) #RELU
        z2[-1] = bias

        suma3 = np.dot(W3,z2)
        z3 = np.maximum(0,suma3)
        y_pred_train = np.append(y_pred_train,z3)

        #Calculo del error
        error = (z3-objetivo)**2

        #Retropropagación
        delta3 = (objetivo - z3) * derivada_relu(z3)
        delta2 = delta3 * W3 * derivada_relu(z2)
        delta1 = np.dot(delta2,W2) * derivada_relu(z1)

        #Ajuste de pesos
        delta1_mayor = (delta1 * z1 * tasa_aprendizaje)
        W1 = W1 + delta1_mayor.T
        delta2_mayor = (delta2 * z2 * tasa_aprendizaje)
        W2 = W2 + delta2_mayor.T
        delta3_mayor = (delta3 * z3 * tasa_aprendizaje)
        W3 = W3 + delta3_mayor.T

    if (regularizacion == "L2"):
        ecm = calcular_ECM(y_train, y_pred_train) + (lambda_reg / (2 * len(y_train))) * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
        error_tiempo = np.append(error_tiempo, ecm)
        delta1_mayor = (delta1 * z1 * tasa_aprendizaje)
        W1 = W1 - (W1*lambda_reg) - delta1_mayor.T 
        delta2_mayor = (delta2 * z2 * tasa_aprendizaje)
        W2 = W2 - (W2*lambda_reg) - delta2_mayor.T 
        delta3_mayor = (delta3 * z3 * tasa_aprendizaje)
        W3 = W3 - (W3*lambda_reg) - delta3_mayor.T 
    else:
        error_tiempo = np.append(error_tiempo, calcular_ECM(y_train,y_pred_train))
    
    y_pred_test = np.empty((0,))

    #PRUEBA
    for muestra in range(len(y_test)):

        patron = X_test[muestra][:]
        patron = np.append(patron,bias)
        objetivo = y_test[muestra]

        #Propagación hacia adelante
        suma1 = np.dot(W1,patron)
        suma1[-1] = bias
        z1 = np.maximum(0,suma1) #RELU
        z1[-1] = bias

        suma2 = np.dot(W2,z1)
        suma2[-1] = bias
        z2 = np.maximum(0,suma2) #RELU
        z2[-1] = bias

        suma3 = np.dot(W3,z2)
        z3 = np.maximum(0,suma3)
        y_pred_test = np.append(y_pred_test,z3)

        #Calculo del error
        error = (z3-objetivo)**2

    error_test = np.append(error_test, calcular_ECM(y_test,y_pred_test))    

graficar_et(error_tiempo,error_test)


