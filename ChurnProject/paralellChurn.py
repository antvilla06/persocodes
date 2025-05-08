import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import tensorflow as tf
from timeit import default_timer as timer
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pyodbc
import concurrent.futures

total_start = timer()

# Obtener los clientes
connection_string = "Driver={SQL Server};Server=convertidordb.lamarina.mx;Database=BODESA_DWH;Trusted_Connection=yes;"
connection = pyodbc.connect(connection_string)
cursor = connection.cursor()

with open('cleint_last3months.sql', 'r') as data:
    compradores = data.read()

query_start = timer()
cursor.execute(compradores)
comprador = [row.iIdCliente for row in cursor.fetchall()]
query_end = timer()
print(f"SQL Query Execution Time: {query_end - query_start:.4f} seconds")
print(comprador)

ncli = 100  # Número de clientes
pcli = 0

def count(vec1, vec2):
    if np.sum(vec1) == 0 and np.sum(vec2) == 0:
        return 1
    elif np.sum(vec1) > 0 and np.sum(vec2) == 0:
        return 0
    elif np.sum(vec1) > 0 and np.sum(vec2) > 0:
        return 1
    else:
        return 0

def process_client(cliente):
    try:
        # Crear conexión local (no se puede compartir entre procesos)
        conn = pyodbc.connect(connection_string)
        with open('client_weekall.sql', 'r') as file:
            query = file.read()
        df = pd.read_sql(query, conn, params=(cliente,))
        conn.close()

        sequence = df['TotalTickets'].to_numpy()
        sequence[sequence > 0] = 1

        window_size = 8
        X, y = [], []

        for i in range(len(sequence) - window_size - 24 + 1):
            X.append(sequence[i:i + window_size])
            y.append(sequence[i + window_size:i + window_size + 12])

        if len(X) == 0:
            return None

        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(128, activation='relu', return_sequences=True, input_shape=(window_size, 1)),
            LSTM(63, activation='relu', return_sequences=False),
            Dense(32, activation='relu'),
            Dense(12)
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X, y, epochs=5, batch_size=16, validation_split=0.09, verbose=0)

        input_sequence = np.array(sequence[-12 - window_size:-12]).reshape((1, window_size, 1))
        predicted = model.predict(input_sequence)
        predicted_binary = (predicted > 0.07).astype(int)

        actual = sequence[-12:]
        correct = count(predicted_binary[0], actual)

        TP = int((np.sum(predicted_binary[0]) > 0) and (np.sum(actual) > 0))
        TN = int((np.sum(predicted_binary[0]) == 0) and (np.sum(actual) == 0))
        FP = int((np.sum(predicted_binary[0]) > 0) and (np.sum(actual) == 0))
        FN = int((np.sum(predicted_binary[0]) == 0) and (np.sum(actual) > 0))

        return (correct, TP, TN, FP, FN)

    except Exception as e:
        print(f"Error con cliente {cliente}: {e}")
        return None

# Ejecutar procesamiento paralelo
results = []
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(process_client, comprador[pcli:ncli]))

# Consolidar resultados
TP = TN = FP = FN = k = 0
for res in results:
    if res is not None:
        correct, tp, tn, fp, fn = res
        k += correct
        TP += tp
        TN += tn
        FP += fp
        FN += fn

p = k / len([r for r in results if r is not None])
print('Accuracy =', p)

# Matriz de confusión
total_cm = np.array([[TN, FP], [FN, TP]])
labels = np.array([[f"TN: {TN}\n{TN/(TN+FP+FN+TP):.2%}", f"FP: {FP}\n{FP/(TN+FP+FN+TP):.2%}"],
                   [f"FN: {FN}\n{FN/(TN+FP+FN+TP):.2%}", f"TP: {TP}\n{TP/(TN+FP+FN+TP):.2%}"]])

plt.figure(figsize=(6, 5))
sns.heatmap(total_cm, annot=labels, fmt='', cmap='hot', cbar=True,
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"],
            linewidths=1, linecolor='black', square=True)
plt.title("Total Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

total_end = timer()
print(f"Total Execution Time: {total_end - total_start:.4f} seconds")