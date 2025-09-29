# ASL Alphabet Classification

This project focuses on building a **Convolutional Neural Network (CNN)** to classify **American Sign Language (ASL) alphabet images**. The dataset used contains images of ASL hand gestures representing the letters of the English alphabet.

---

## 📂 Dataset

* **Source**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* **Structure**:

  * `asl_alphabet_train/` → Training images for each letter.
  * `asl_alphabet_test/` → Test images (one sample per class).

Images were resized to a fixed input size (`input_size = (64, 64, 3)`) for consistency.

---

## 🧠 Model Architecture

The CNN was designed with **regularization, dropout, and batch normalization** to improve generalization and prevent overfitting:

* **Conv2D + BatchNormalization + MaxPooling** blocks with increasing filters: 32 → 64 → 128 → 256
* **Dropout layers** (0.25–0.4) to reduce overfitting
* **GlobalAveragePooling2D** instead of flattening → reduces parameters
* **Dense layer (512 units)** with L2 regularization
* **Output layer**: `Dense(num_classes, activation='softmax')`

### Key Parameters:

* Optimizer: **Adam**
* Loss: **categorical_crossentropy**
* Metrics: **accuracy**
* Regularization: **L2 (weight_decay = 1e-4)**

---

## ⚡ Training Setup

Callbacks were used to optimize training:

* **EarlyStopping** → stop if no improvement in `val_loss` for 10 epochs.
* **ReduceLROnPlateau** → reduce learning rate when validation loss plateaus.
* **ModelCheckpoint** → save best model (`best_model.h5`) based on `val_accuracy`.

```python
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=True
)
```

---

## 📊 Results

* The model shows strong performance on validation data, with accuracy improving over epochs.
* Confusion matrix was plotted to visualize per-class performance.
* Model is saved as **`best_model.h5`** for deployment or inference.

---

## 🚀 Future Work

* Use **data augmentation** (rotation, shift, zoom, flip) to further improve generalization.
* Experiment with **pre-trained models** (Transfer Learning: ResNet50, MobileNet, EfficientNet).
* Deploy the model as an **interactive ASL recognition app**.

---

## ✅ Summary

This project demonstrates how a well-regularized CNN can effectively classify ASL alphabet hand signs. With further fine-tuning and augmentation, it can be deployed in real-world applications to help people communicate using sign language.
