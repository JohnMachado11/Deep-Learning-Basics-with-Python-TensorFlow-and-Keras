import cv2
import tensorflow as tf
import matplotlib.pyplot as plt


CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value
IMG_SIZE = 50  # 50 in txt-based

def prepare(filepath):

    # Read the image in color
    color_image = cv2.imread(filepath)
    
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

    return new_array, color_image

# model = tf.keras.models.load_model("64x3-CNN.h5")
model = tf.keras.models.load_model("300-epochs.h5")

# Paths to the images
# image_paths = ["dog.png", "cat.png", "dudu.jpg", "potter.jpg"]
image_paths = ["dudu.jpg", "potter.jpg", "layla.jpeg", "maple.jpeg", "nelly.jpeg"]


# Prepare and predict each image
predictions = []
original_images = []
for path in image_paths:
    image, original_image = prepare(path)
    prediction = model.predict(image)
    pred_class = CATEGORIES[int(prediction[0][0])]

    predictions.append(pred_class)
    original_images.append((original_image, path))

# Plotting the results in a larger size
plt.figure(figsize=(15, 10))
for i in range(len(original_images)):
    plt.subplot(2, 3, i + 1)
    original_image, filename = original_images[i]
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    plt.title(f"Predicted: {predictions[i]}\n{filename}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# prediction = model.predict(prepare("dog.png"))  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
# print("Dog Prediction: ")
# print(CATEGORIES[int(prediction[0][0])])

# prediction = model.predict(prepare("cat.png"))
# print("Cat Prediction: ")
# print(CATEGORIES[int(prediction[0][0])])