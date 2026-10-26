import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("saved/best_finetuned.keras")

img = image.load_img("broken-eggshells.jpg", target_size=(224,224)) 
x = image.img_to_array(img)/255.
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
class_names = ["Hazardous", "Non-Recyclable", "Organic", "Recyclable"]
print("\nPrediction:", class_names[np.argmax(pred)])
