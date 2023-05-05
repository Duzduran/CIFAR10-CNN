from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np

# classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# Load CIFAR10 data
(x_train_all, y_train_all), (x_test_all, y_test_all) = keras.datasets.cifar10.load_data()

# Filter images to only include animals and vehicles
vehicle_indices = np.where((y_train_all == 0) | (y_train_all == 1) | (y_train_all == 8) | (y_train_all == 9))[0]
animal_indices = np.where(
    (y_train_all == 2) | (y_train_all == 3) | (y_train_all == 4) | (y_train_all == 5) | (y_train_all == 6) | (
                y_train_all == 7))[0]

x_train_vehicle = x_train_all[vehicle_indices]
y_train_vehicle = y_train_all[vehicle_indices]

x_train_animal = x_train_all[animal_indices]
y_train_animal = y_train_all[animal_indices]

vehicle_indices = np.where((y_test_all == 0) | (y_test_all == 1) | (y_test_all == 8) | (y_test_all == 9))[0]
animal_indices = np.where(
    (y_test_all == 2) | (y_test_all == 3) | (y_test_all == 4) | (y_test_all == 5) | (y_test_all == 6) | (
                y_test_all == 7))[0]

x_test_vehicle = x_test_all[vehicle_indices]
y_test_vehicle = y_test_all[vehicle_indices]

x_test_animal = x_test_all[animal_indices]
y_test_animal = y_test_all[animal_indices]

# Assign labels 0 and 1 to animals and vehicles, respectively
y_train_vehicle = np.zeros_like(y_train_vehicle)
y_train_animal = np.ones_like(y_train_animal)

y_test_vehicle = np.zeros_like(y_test_vehicle)
y_test_animal = np.ones_like(y_test_animal)

# Concatenate animal and vehicle data
x_train = np.concatenate([x_train_vehicle, x_train_animal], axis=0)
y_train = np.concatenate([y_train_vehicle, y_train_animal], axis=0)

x_test = np.concatenate([x_test_vehicle, x_test_animal], axis=0)
y_test = np.concatenate([y_test_vehicle, y_test_animal], axis=0)

# Split data into training and testing sets with a ratio of 3:7
x_train, x_test, y_train, y_test = train_test_split(
    np.concatenate([x_train, x_test]),
    np.concatenate([y_train, y_test]),
    test_size=0.7,
    random_state=42,
)
classes = ["animal", "vehicle"]

x_train = x_train / 255
x_test = x_test / 255

cnn1 = keras.models.Sequential([
    # cnn
    keras.layers.Conv2D(filters=32, activation='relu', kernel_size=(3, 3), input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(2, 2),

    # dense
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

cnn2 = keras.models.Sequential([
    # cnn
    keras.layers.Conv2D(filters=32, activation='relu', kernel_size=(3, 3), input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(filters=32, activation='relu', kernel_size=(3, 3)),
    keras.layers.MaxPooling2D(2, 2),
    # dense
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
cnn3 = keras.models.Sequential([
    # cnn
    keras.layers.Conv2D(filters=32, activation='relu', kernel_size=(3, 3), input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(filters=32, activation='relu', kernel_size=(3, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(filters=32, activation='relu', kernel_size=(3, 3)),
    keras.layers.MaxPooling2D(2, 2),
    # dense
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
cnn1.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
cnn1.fit(x_train, y_train, epochs=5)

cnn2.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
cnn2.fit(x_train, y_train, epochs=5)
cnn3.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
cnn3.fit(x_train, y_train, epochs=5)

# predictions
y_pred1 = cnn1.predict(x_test)
y_pred_classes1 = [np.argmax(element) for element in y_pred1]

y_pred2 = cnn2.predict(x_test)
y_pred_classes2 = [np.argmax(element) for element in y_pred2]

y_pred3 = cnn3.predict(x_test)
y_pred_classes3 = [np.argmax(element) for element in y_pred3]
print("Classification Report(1 layers): \n", classification_report(y_test, y_pred_classes1))
print("Classification Report(2 layers): \n", classification_report(y_test, y_pred_classes2))
print("Classification Report(3 layers): \n", classification_report(y_test, y_pred_classes3))