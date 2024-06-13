from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense, InputLayer, Activation

model = Sequential([
                    InputLayer(input_shape=(160, 160, 3)),
                    Flatten(),
                    Dense(units = 256),
                    BatchNormalization(),
                    Activation('relu'),

                    Dense(units=256),
                    BatchNormalization(),
                    Activation('relu'),

                    Dense(units = 256),
                    BatchNormalization(),
                    Activation('relu'),

                   Dense(units=2, activation = 'softmax')
                   ])




model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
