def AutoEncoder(input_size, hidden_size, output_size):
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(input_size,), activation='relu'))
    model.add(Dense(output_size, activation='sigmoid'))
    return model
