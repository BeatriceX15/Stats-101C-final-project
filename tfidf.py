from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import csv
import numpy as np

strings = []
attitude = []
csv_file_path = '../cleaned2.csv'
all_words_path = '../Dataset/all-words.txt'
df = pd.read_csv(csv_file_path, header=0)


# Define a function to read a file and return a list of words
def read_words_from_file(file_path):
    with open(file_path, 'r', encoding='latin-1') as file:  # using 'latin-1' encoding to avoid encoding issues
        words = file.read().splitlines()  # Read and split lines to get individual words
    return words

all_words = read_words_from_file(all_words_path)
all_words = np.unique(all_words)


def vectorize_line(line, dictionary):
    # Tokenize the line into words and convert to lowercase
    tokens = line.lower().split()
    # Create a vector of zeros with the same length as the number of unique words in the dictionary
    vector = [0] * len(all_words)
    # Create a mapping of words to their indices in the vector
    word_to_index = {word: i for i, word in enumerate(all_words)}

    # Iterate over words in the line
    for word in tokens:
        # If the word is in the dictionary, set the corresponding value in the vector to 1
        if word in word_to_index:
            vector[word_to_index[word]] = 1

    return vector

vectorized_line = []
counter = 0
reviews = []

with open(csv_file_path, 'r') as file:
    # Create a CSV reader
    csv = csv.reader(file)
    next(csv)
    for line in csv:
        words = line[0].split()
        strings.append(list(words))
        attitude.append(line[1])
        reviews.append(line[0])
        vl = vectorize_line(line[0], all_words)
        counter += sum(vl)
        vectorized_line.append(list(vl))

vectorized_matrix = np.array(vectorized_line).reshape((50000, 6786))
print(counter)
print(vectorized_matrix.shape)




# ======================== TFIDF REDUCTION ========================
result = [sum(col) for col in zip(*vectorized_matrix)]
print(result)

zero_indices = [index for index, value in enumerate(result) if value == 0]

print("Indices of elements with value 0:", zero_indices)

new_matrix = np.delete(vectorized_matrix, zero_indices, axis=1)
print(new_matrix.shape)

words_used = [value for index, value in enumerate(all_words) if index not in zero_indices]
tv=TfidfVectorizer(min_df=0.0,max_df=1.0,use_idf=True,ngram_range=(1,3),vocabulary=words_used)
tv_ft = tv.fit_transform(reviews)
print(tv_ft)
print('Tfidf:',tv_ft.shape)

tfidf = tv_ft.max(axis=0)
tfidf = tfidf.todense().tolist()
tfidf = tfidf[0]
print(tfidf)
print(len(tfidf))



# ======================== Model Evaluation with TFIDF ========================

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# ======================== TFIDF 0.2-0.7 ========================
indices_outside_range = [index for index, value in enumerate(tfidf) if value < 0.2 or value > 0.7]
reduced_matrix = np.delete(new_matrix, indices_outside_range, axis=1)
print(reduced_matrix.shape)

X = reduced_matrix
y = attitude
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



# ======================== TFIDF 0.5-0.9 ========================
indices_outside_range = [index for index, value in enumerate(tfidf) if value < 0.5 or value > 0.9]
reduced_matrix = np.delete(new_matrix, indices_outside_range, axis=1)
print(reduced_matrix.shape)

X = reduced_matrix
y = attitude
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)


# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Random Forest
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
