import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from timeit import default_timer as timer
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


connection_string = "Driver={SQL Server};Server=convertidordb.lamarina.mx;Database=BODESA_DWH;Trusted_Connection=yes;"
connection = pyodbc.connect(connection_string)

cursor = connection.cursor()
with open('cleintes_freq4.sql', 'r') as data:
    compradores = data.read()

cursor.execute(compradores)
comprador = list(set([row.iIdCliente for row in cursor.fetchall()]))
#print(comprador[5:40])
k = 0
ncli = 90 # Number of clients
pcli = 0

TP, TN, FP, FN = 0, 0, 0, 0

def count(vec1, vec2):
    if np.sum(vec1) == 0 and np.sum(vec2) == 0:
        return 1
    elif np.sum(vec1) > 0 and np.sum(vec2) == 0:
        return 0
    elif np.sum(vec1) > 0 and np.sum(vec2) > 0:
        return 1
    else:
        return 0

for cliente in comprador[pcli:ncli]:
    with open('client_weekall.sql', 'r') as file:
        query = file.read()

    df = pd.read_sql(query, connection, params=(cliente,))
    random = df['TotalTickets'].to_numpy()
    random[random > 0] = 1

    plt.figure(figsize=(10, 6))
    plt.plot(df['TotalTickets'], marker='o', linestyle='-', label=f'Client {cliente}')
    plt.title(f'Total Tickets Over Time for Client {cliente}', fontsize=16)
    plt.ylabel('Total Tickets', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.close()

    window_size = 8
    M=len(random)-window_size+1
    sequence=random
    

    X, y = [], []
    
    for i in range(len(sequence) -window_size-12+1):
        X.append(sequence[i:i+window_size])
        y.append(sequence[i+window_size:i+window_size+12])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(128, activation='relu',return_sequences = True, input_shape=(window_size, 1)),
        LSTM(63, activation = 'relu', return_sequences = False),
        Dense(32, activation='relu'),
        Dense(12)
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=16, validation_split=0.09, verbose=0)

    input_sequence = np.array(sequence[len(sequence)-12-window_size:len(sequence)-12]).reshape((1, window_size, 1))
    predicted = model.predict(input_sequence)
    predicted_binary = (predicted > 0.18).astype(int)
    print(random[len(random)-12:])
    print(random)
    print("Predicted next 4 numbers:", predicted_binary[0])
    k += count(predicted_binary[0], random[len(random)-12:])

    # Update confusion matrix counts
    TP += int((np.sum(predicted_binary[0]) > 0) and (np.sum(random[len(random)-12:]) > 0))
    TN += int((np.sum(predicted_binary[0]) == 0) and (np.sum(random[len(random)-12:]) == 0))
    FP += int((np.sum(predicted_binary[0]) > 0) and (np.sum(random[len(random)-12:]) == 0))
    FN += int((np.sum(predicted_binary[0]) == 0) and (np.sum(random[len(random)-12:]) > 0))

p = k / len(comprador[pcli:ncli])
print('Accuracy =', p)

# Create and plot confusion matrix
total_cm = np.array([[TN, FP], [FN, TP]])
labels = np.array([[f"TN: {TN}\n{TN/(TN+FP+FN+TP):.2%}", f"FP: {FP}\n{FP/(TN+FP+FN+TP):.2%}"],
                   [f"FN: {FN}\n{FN/(TN+FP+FN+TP):.2%}", f"TP: {TP}\n{TP/(TN+FP+FN+TP):.2%}"]])

plt.figure(figsize=(6, 5))
sns.heatmap(total_cm, annot=labels, fmt='', cmap='hot', cbar=True, xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["Actual Negative", "Actual Positive"], linewidths=1, linecolor='black', square=True)
plt.title("Total Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()