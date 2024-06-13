from tensorflow.keras import layers 
from tensorflow.keras.applications import ResNet50V2 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 

def build_model():
    densenet = ResNet50V2(
                        weights='imagenet',
                        include_top=False,
                        input_shape=(160,160,3)
                        )
    model = Sequential([densenet,

                        layers.GlobalAveragePooling2D(),
                        layers.Dense(1024, activation = 'relu'),
                        layers.BatchNormalization(),
                         layers.Dense(256, activation = 'relu'),
                        layers.BatchNormalization(),


                        layers.Dense(2, activation='softmax')
                        ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                 )
    return model
