import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical

# Load data
data = pd.read_csv('sms_spam.csv')  # Replace 'sms_spam.csv' with your dataset file

# Prepare data
X = data['text']
y = data['label']

# Tokenization
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X)

# Convert labels to one-hot encoding
y = pd.get_dummies(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Embedding(max_words, 128, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(196, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 5
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

# Evaluate the model
score, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Test accuracy:", accuracy)

# Example usage
sms_text = ["Congratulations! You've won a free cruise. Call now to claim your prize!",
            "Hey, just wanted to check in and see how you're doing. Let's catch up soon!"]
sequences = tokenizer.texts_to_sequences(sms_text)
padded_sequences = pad_sequences(sequences, maxlen=X.shape[1])
predictions = model.predict(padded_sequences)
print("Predictions:")
for i, text in enumerate(sms_text):
    print(text, "->", "Spam" if predictions[i][1] > 0.5 else "Not Spam")
