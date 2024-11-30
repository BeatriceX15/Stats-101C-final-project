import numpy as np
import pandas as pd
import csv
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV



strings = []
attitude = []
comments = []

# ======================== PREPROCESSING =====================
csv_file_path = '/Users/chaoslxa/Desktop/STATS101C/cleaned2.csv'
df = pd.read_csv(csv_file_path, header=0)
#print(df)

positive = '/Users/chaoslxa/Desktop/STATS101C/positive-words.txt'
negative = '/Users/chaoslxa/Desktop/STATS101C/negative-words.txt'
allwordspath = '/Users/chaoslxa/Desktop/STATS101C/all-words.txt'

#csv_file_path = '../cleaned2.csv'
#df = pd.read_csv(csv_file_path, header=0)

#positive = '../Dataset/positive-words.txt'
#negative = '../Dataset/negative-words.txt'
#all_words_path = '../Dataset/all-words.txt'

def read_words_from_file(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:
        words = file.read().splitlines()
    return words

negative_words = read_words_from_file(negative)
positive_words = read_words_from_file(positive)
sentiment_dictionary = {'positive': positive_words, 'negative': negative_words}

all_words = read_words_from_file(all_words_path)
word_dictionary = {}
for i, word in enumerate(all_words):
    word_dictionary.update({word: 0})

with open(csv_file_path, 'r') as file:
    csv = csv.reader(file)
    next(csv)
    for line in csv:
        comments.append(line[0])
        words = line[0].split()
        strings.append(words)
        attitude.append(line[1])
        text_tokens = line[0].lower().split()

        for token in text_tokens:
            if token in word_dictionary:
                word_dictionary[token] += 1

keys_to_delete = [key for key, value in word_dictionary.items() if value == 0]
for key in keys_to_delete:
    del word_dictionary[key]


# ========== Sentiment Count ==========
print(attitude.count('0'))
print(attitude.count('1'))


# ======================== PCA =====================
vectorizer = CountVectorizer(vocabulary=word_dictionary.keys())
word_count_matrix = vectorizer.fit_transform(comments)
# Convert to DataFrame for easier viewing and manipulation
word_count_df = pd.DataFrame(word_count_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# PCA Cumulative Variance Explained by Each Component
pca = PCA().fit(word_count_matrix.toarray())
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= 0.95) + 1
plt.plot(cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Get the explained variance of each component
explained_variance = pca.explained_variance_ratio_
# Select the top 500 components
top_500_components = np.cumsum(explained_variance[:1000])
# To get the cumulative variance explained by the top 500 components
cumulative_variance_top_500 = np.cumsum(top_500_components)
plt.plot(top_500_components)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()


scaler = StandardScaler()
standardized_data = scaler.fit_transform(word_count_matrix.toarray())
pca = PCA(n_components=500)
pca_result = pca.fit_transform(standardized_data)

X = pca_result
y = attitude
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# ======================== MODELING =====================
# ========== Logistic Regression ==========
# Grid Search Cross Validation for Logistic Regression with L2 Regularization
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 10], 'penalty': ['l2'], }
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

# Logistic Regression with optimal hyperparameters
logreg = LogisticRegression(penalty='l2', C=0.001)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)


# ========== KNN ==========
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

knn_classifier = KNeighborsClassifier(n_neighbors=15)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

knn_classifier = KNeighborsClassifier(n_neighbors=25)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# Grid Search Cross Validation for KNN
knn_classifier = KNeighborsClassifier()
param_grid = {'n_neighbors': [5, 7, 9, 15, 35, 151, 201, 251, 301]}
grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

# KNN using optimal hyperparameter
knn_classifier = KNeighborsClassifier(n_neighbors=151)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# ========== SVM ==========
from sklearn import svm
from sklearn.metrics import accuracy_score

svm_model = svm.SVC(kernel='sigmoid', probability=True)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# ========== LDA ==========
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# ========== QDA ==========
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
qda_classifier = QuadraticDiscriminantAnalysis()
qda_classifier.fit(X_train, y_train)
y_pred = qda_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# ========== Random Forest ==========
# Random Search CV for Random Forest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state=999)
param_dist = {'n_estimators': [1, 5, 10, 25, 50, 100, 200], 'max_depth': [1, 5, 10, 30, 50, 90, 100, 150],
              'max_features': ["sqrt", "log2"], 'min_samples_leaf': [1, 2, 5, 10, 25, 50],
              'min_samples_split': [2, 7, 10, 20, 50]}
random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=5)
random_search.fit(X_train, y_train)
print("Best Hyperparameters:", random_search.best_params_)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

# Random Forest using optimal hyperparameters (12/30/23)
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=200, min_samples_split=7, min_samples_leaf=25,
                max_features='log2', max_depth=150, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



# ======================== ROC CURVES =====================
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

mapping = {'1': 1, '0': 0}
y_test = [mapping[item] for item in y_test]


# ========== Logistic Regression ==========
y_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ========== KNN ==========
y_prob = knn_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ========== SVM ==========
y_prob = svm_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ========== LDA ==========
y_prob = lda.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ========== QDA ==========
y_prob = qda_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# ========== Random Forest ==========
y_prob = rf_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



# ================================= NEURAL NETWORK =================================
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, GlobalMaxPool1D, Bidirectional
from keras.models import Model
from keras.utils import to_categorical

comments = pd.read_csv("../cleaned2.csv")
positive_lexicon = set(open("../Dataset/positive-words.txt").read().splitlines())
negative_lexicon = set(open("../Dataset/negative-words.txt").read().splitlines())

# Tokenization and Padding
max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(comments['review'])
tokenized_comments = tokenizer.texts_to_sequences(comments['review'])
maxlen = 400
X = pad_sequences(tokenized_comments, maxlen=maxlen)

# Use the sentiment columns as labels
y = to_categorical(comments['sentiment'], num_classes=2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ========== LSTM Model 1 ==========
inp = Input(shape=(maxlen,))
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.2)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2, activation="softmax")(x)  # Use softmax for multi-class classification
model = Model(inputs=inp, outputs=x)
model.compile(loss='categorical_crossentropy',  # Use categorical crossentropy for multi-class classification
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

batch_size = 32
epochs = 5
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
score = model.evaluate(X_test, y_test)
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')

# ========== LSTM Model 2 ==========
inp = Input(shape=(maxlen,))
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(LSTM(60, return_sequences=True, name='lstm_layer1'))(x)  # Added Bidirectional LSTM
x = Bidirectional(LSTM(30, return_sequences=True, name='lstm_layer2'))(x)  # Added another Bidirectional LSTM
x = GlobalMaxPool1D()(x)
x = Dropout(0.2)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2, activation="softmax")(x)
model2 = Model(inputs=inp, outputs=x)
model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model2.summary())

batch_size = 32
epochs = 5
history = model2.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
score = model2.evaluate(X_test, y_test)
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')

# ========== LSTM Model 3 ==========
from keras.layers import Conv1D, MaxPooling1D

# Updated Model with Convolutional Layer
inp = Input(shape=(maxlen,))
embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = Conv1D(128, 5, activation='relu')(x)  # Added 1D Convolutional layer
x = MaxPooling1D(5)(x)
x = Bidirectional(LSTM(60, return_sequences=True, name='lstm_layer1'))(x)
x = Bidirectional(LSTM(30, return_sequences=True, name='lstm_layer2'))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.2)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(2, activation="softmax")(x)
model3 = Model(inputs=inp, outputs=x)
model3.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model3.summary())

batch_size = 32
epochs = 5
history = model3.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)
score = model3.evaluate(X_test, y_test)
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')


# Model 3 ROC Curve
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Predict probabilities on the test set
y_prob = model3.predict(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test[:, 1], y_prob)  # Assuming y_test is a 2D array
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()