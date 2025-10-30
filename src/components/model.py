from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_ann_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)), # THis represents the input layer and 64 neurons in the hidden layer and activation function is relu
        Dropout(0.3), # This represents the dropout layer means drop 30% of the neurons in the hidden layer
        Dense(32, activation='relu'), # This represents the hidden layer 2 and 32 neurons are present in this hidden layer
        Dense(1, activation='sigmoid')  # For binary classification sigmoid activation function is used and this is the output layer with 1 neuron
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
