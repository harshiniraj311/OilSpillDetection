from pyspark.sql import SparkSession
import os

# Initialize Spark
spark = SparkSession.builder.appName("OilSpillDetection").getOrCreate()
sc = spark.sparkContext

# List of locations with bounding boxes
locations = [
    ("Gulf_of_Mexico", [-90.0, 28.0, -89.9, 28.1]),
    ("Port_of_LA", [-118.3, 33.7, -118.2, 33.8]),
    ("Mumbai_Coast", [72.8, 18.9, 72.9, 19.0])
]

locations_rdd = sc.parallelize(locations)

# Function to detect oil spill in a given location
def detect_oil_spill_distributed(location):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions

    location_name, bbox_coords = location
    print(f"Processing: {location_name}")

    # Sentinel Hub Configuration
    config = SHConfig()
    config.instance_id = "769210e6-ece3-4735-840c-0284735f62be"  # Fill in
    config.sh_client_id = "c9d6bac7-bcb5-423f-8d54-3f1ea2b4db36"  # Fill in
    config.sh_client_secret = "EEnLfpjryLHz5nuvqABeCXyboGCbkrz9"  # Fill in


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
            responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
            bbox=bbox,
            size=size,
            config=config
        )

        # Fetch image
        img = request.get_data()[0] / 255.0

        # Convert to grayscale and apply thresholding
        gray = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        result = (img * 255).astype('uint8').copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        # Save results
        output_dir = f"results/{location_name}"
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/mask.png", mask)
        cv2.imwrite(f"{output_dir}/detection.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        return f"{location_name} processed and saved."

    except Exception as e:
        return f"{location_name} failed: {str(e)}"

# Run detection across all locations in parallel
results = locations_rdd.map(detect_oil_spill_distributed).collect()

# Output results
for r in results:
    print(r)
