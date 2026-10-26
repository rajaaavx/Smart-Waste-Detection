from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("saved/best_finetuned.keras")
test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255).flow_from_directory(
    "dataset/test", target_size=(224,224), class_mode='categorical', shuffle=False
)

pred = model.predict(test)
y_true = test.classes
y_pred = np.argmax(pred, axis=1)

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_true, y_pred))
