import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

train_0PR = pd.read_csv('gtcc_gdcc_4.csv', header=None)
train_1PR = pd.read_csv('gtcc_gdcc_5.csv', header=None)
train_2PR = pd.read_csv('gtcc_gdcc_6.csv', header=None)

test_0PR = pd.read_csv('gtcc_gdcc_1.csv', header=None)
test_1PR = pd.read_csv('gtcc_gdcc_2.csv', header=None)
test_2PR = pd.read_csv('gtcc_gdcc_3.csv', header=None)

X_train = np.vstack((train_0PR.values, train_1PR.values, train_2PR.values))
y_train = np.array([0] * len(train_0PR) + [1] * len(train_1PR) + [2] * len(train_2PR))

X_test = np.vstack((test_0PR.values, test_1PR.values, test_2PR.values))
y_test = np.array([0] * len(test_0PR) + [1] * len(test_1PR) + [2] * len(test_2PR))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model = Sequential([
    LSTM(64, input_shape=(1, 39), return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_reshaped, y_train_cat, epochs=50, batch_size=32, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['0PR', '1PR', '2PR']))

def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

def compute_tdcf_audio(y_true, y_scores, p_bonafide=0.95, c_miss_asv=1, c_fa_asv=10, c_miss_cm=1, c_fa_cm=10):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    p_spoof = 1 - p_bonafide
    
    p_miss_asv = 0.05
    p_fa_asv = 0.01
    
    c_asv = p_bonafide * p_miss_asv * c_miss_asv + p_spoof * p_fa_asv * c_fa_asv
    
    tdcf_norm = []
    for i in range(len(thresholds)):
        p_miss_cm = fnr[i]
        p_fa_cm = fpr[i]
        c_cm = p_bonafide * p_miss_cm * c_miss_cm + p_spoof * p_fa_cm * c_fa_cm
        tdcf = c_cm + p_bonafide * (1 - p_miss_cm) * p_miss_asv * c_miss_asv + p_spoof * (1 - p_fa_cm) * p_fa_asv * c_fa_asv
        tdcf_norm.append(tdcf / c_asv)
    
    min_tdcf = min(tdcf_norm)
    return min_tdcf

y_test_binary = (y_test != 0).astype(int)
y_scores = 1 - y_pred[:, 0]  

eer = compute_eer(y_test_binary, y_scores)
print(f"EER (%): {eer * 100:.2f}%")

min_tdcf = compute_tdcf_audio(y_test_binary, y_scores)
print(f"Min t-DCF: {min_tdcf:.4f}")