
# 1. Download Venice Lagoon Test Image
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
import matplotlib.pyplot as plt
import numpy as np
import os

config = SHConfig()
config.instance_id = "769210e6-ece3-4735-840c-0284735f62be"
config.sh_client_id = "c9d6bac7-bcb5-423f-8d54-3f1ea2b4db36"
config.sh_client_secret = "EEnLfpjryLHz5nuvqABeCXyboGCbkrz9"

location_name = "Venice_Lagoon_Test"
bbox_coords = [12.3, 45.3, 12.4, 45.4]
time_range = ('2023-11-01', '2023-12-01')
bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
size = bbox_to_dimensions(bbox, resolution=10)

request = SentinelHubRequest(
    evalscript="""
    function setup() {
        return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3 }
        };
    }
    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """,
    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L1C, time_interval=time_range)],
    responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
    bbox=bbox,
    size=size,
    config=config
)
img = request.get_data()[0]
os.makedirs(f"test_images/{location_name}", exist_ok=True)
test_image_path = f"test_images/{location_name}/test_rgb.png"
plt.imsave(test_image_path, img)
print(f"Test image saved at: {test_image_path}")


# 2. Distributed Dataset Collection

from pyspark.sql import SparkSession
import cv2

spark = SparkSession.builder.appName("OilSpillDatasetCollector").getOrCreate()
sc = spark.sparkContext

locations = [
    ("Gulf_of_Mexico", [-90.0, 28.0, -89.9, 28.1]),
    ("Port_of_LA", [-118.3, 33.7, -118.2, 33.8]),
    ("Mumbai_Coast", [72.8, 18.9, 72.9, 19.0]),
    ("Niger_Delta", [6.0, 4.8, 6.1, 4.9]),
    ("Persian_Gulf", [50.0, 26.5, 50.1, 26.6]),
    ("Singapore_Strait", [103.7, 1.2, 103.8, 1.3]),
    ("Rotterdam_Port", [4.3, 51.9, 4.4, 52.0]),
    ("Alaska_Coast", [-150.0, 60.9, -149.9, 61.0])
]
locations_rdd = sc.parallelize(locations)

def fetch_and_tile(location):
    from sentinelhub import SentinelHubRequest, MimeType, DataCollection, BBox, bbox_to_dimensions, CRS, SHConfig
    import numpy as np
    import os
    import cv2

    PATCH_SIZE = 64
    THRESHOLD = 0.10
    name, bbox_coords = location
    print(f"🛰 Processing {name}")
    config = SHConfig()
    config.instance_id = "769210e6-ece3-4735-840c-0284735f62be"
    config.sh_client_id = "c9d6bac7-bcb5-423f-8d54-3f1ea2b4db36"
    config.sh_client_secret = "EEnLfpjryLHz5nuvqABeCXyboGCbkrz9"

    try:
        bbox = BBox(bbox=bbox_coords, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=10)

        request = SentinelHubRequest(
            evalscript="""
            function setup() {
                return {
                    input: ["B04", "B03", "B02"],
                    output: { bands: 3 }
                };
            }
            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
            """,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=('2023-10-01', '2023-12-31'),
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=bbox,
            size=size,
            config=config
        )

        img = request.get_data()[0]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

        h, w = mask.shape
        for i in range(0, h, PATCH_SIZE):
            for j in range(0, w, PATCH_SIZE):
                img_patch = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                mask_patch = mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                if img_patch.shape[0] < PATCH_SIZE or img_patch.shape[1] < PATCH_SIZE:
                    continue

                oil_pixels = (mask_patch > 127).sum()
                label = "oil_spill" if oil_pixels > (PATCH_SIZE * PATCH_SIZE * THRESHOLD) else "no_spill"
                save_path = f"dataset/{label}/"
                os.makedirs(save_path, exist_ok=True)
                filename = f"{name}{i}{j}.png"
                cv2.imwrite(os.path.join(save_path, filename), img_patch)

        return f" {name} processed."
    except Exception as e:
        return f"{name} failed: {str(e)}"

results = locations_rdd.map(fetch_and_tile).collect()
for res in results:
    print(res)


# 3. CNN Model Training

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import torch

def load_image_label(path_label_tuple):
    path, label = path_label_tuple
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    return (img, label)

base_path = "dataset"
oil_paths = [(os.path.join(base_path, "oil_spill", f), 1) for f in os.listdir(os.path.join(base_path, "oil_spill"))]
no_paths = [(os.path.join(base_path, "no_spill", f), 0) for f in os.listdir(os.path.join(base_path, "no_spill"))]
data = oil_paths + no_paths
rdd = sc.parallelize(data)
img_label = rdd.map(load_image_label).collect()

X = np.array([x[0] for x in img_label])
y = to_categorical(np.array([x[1] for x in img_label]), num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

import tensorflow as tf
from tensorflow.keras import backend as K
def dice_loss(y_true, y_pred, smooth=1.0):
    y_pred = tf.keras.activations.sigmoid(y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
model.save("my_model.h5")
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")


# 4. Predict on New Image

from tensorflow.keras.models import load_model

model = load_model("my_model.h5", custom_objects={'dice_loss': dice_loss})

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

img = preprocess_image("test_images/Venice_Lagoon_Test/test_rgb.png")
prediction = model.predict(img)

class_idx = np.argmax(prediction)
class_label = "Oil Spill" if class_idx == 1 else "No Spill"
print(f"Prediction: {class_label} (Confidence: {prediction[0][class_idx]:.2f})")