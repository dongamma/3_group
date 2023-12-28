# 과제#4

상태: 시작 전

![Untitled](%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6#4%2072dec06566f74e60bf50d0881eae5e22/Untitled.png)

![Untitled](%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6#4%2072dec06566f74e60bf50d0881eae5e22/Untitled%201.png)

```python
model = keras.models.Sequential([
    keras.layers.Conv2D(32, 7, activation="relu", padding="same",
                        input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax"),
])
```

![Untitled](%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6#4%2072dec06566f74e60bf50d0881eae5e22/Untitled%202.png)

![Untitled](%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6#4%2072dec06566f74e60bf50d0881eae5e22/Untitled%203.png)

![Untitled](%E1%84%80%E1%85%AA%E1%84%8C%E1%85%A6#4%2072dec06566f74e60bf50d0881eae5e22/Untitled%204.png)

0

```python
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]
```

```python
model = keras.models.Sequential([
    keras.layers.Conv2D(32, 7, activation="relu", padding="same",
                        input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax"),
])
```

```python
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10]
y_pred = model.predict(X_new)
```

```python
model.summary()
```

```python
model.evaluate(X_test, y_test)
```