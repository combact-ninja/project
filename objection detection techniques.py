import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
import numpy as np
import cv2

# Load a pretrained Vision Transformer and modify it for denoising (deblurring)
class DeblurringTransformer(nn.Module):
    def __init__(self, model_name='google/vit-base-patch16-224'):
        super(DeblurringTransformer, self).__init__()
        # Load pretrained Vision Transformer
        self.vit = ViTForImageClassification.from_pretrained(model_name)
        
        # Modify the classifier layer to output the same number of channels as input
        self.vit.classifier = nn.Conv2d(in_channels=768, out_channels=3, kernel_size=1)

    def forward(self, x):
        # Pass through the Vision Transformer model
        x = self.vit(x).logits
        return x

# Function to apply Gaussian blur to an image
def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Replace with your image dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize model, loss function, and optimizer
model = DeblurringTransformer()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(5):  # Just a few epochs for demonstration
    model.train()
    running_loss = 0.0
    for i, (images, _) in enumerate(train_loader):
        # Apply Gaussian Blur to the images
        blurred_images = torch.tensor([apply_gaussian_blur(img.numpy().transpose(1, 2, 0)).transpose(2, 0, 1) for img in images])

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(blurred_images.float())

        # Calculate loss (difference between sharp and blurred image)
        loss = criterion(outputs, images)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print loss
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Epoch [{epoch + 1}], Batch [{i + 1}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print("Training Complete.")

# Example of testing the model with a blurred image
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load a test image
test_image = cv2.imread('test_image.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
blurred_test_image = apply_gaussian_blur(test_image)

# Convert the image to a tensor
test_tensor = torch.tensor(blurred_test_image).permute(2, 0, 1).unsqueeze(0).float()

# Deblur using the model
model.eval()
with torch.no_grad():
    deblurred_output = model(test_tensor)
    deblurred_image = deblurred_output.squeeze(0).permute(1, 2, 0).numpy()

# Display the blurred and deblurred images
cv2.imshow('Blurred Image', blurred_test_image)
cv2.imshow('Deblurred Image', deblurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()









import cv2
import numpy as np

def hybrid_sift_object_detection(query_image_path, target_image_path):
    """
    Hybrid SIFT-based object detection and localization using SIFT, feature matching, and RANSAC.
    
    Args:
    - query_image_path: Path to the query image.
    - target_image_path: Path to the target image.
    
    Returns:
    - None (Displays images with detected keypoints and matches)
    """
    # Load the images
    query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded
    if query_image is None or target_image is None:
        print("Error: Could not load images.")
        return

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors with SIFT
    query_keypoints, query_descriptors = sift.detectAndCompute(query_image, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(target_image, None)

    # Initialize FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN parameters for KD-Tree
    search_params = dict(checks=50)  # Number of times the tree(s) in the index should be recursively traversed
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using KNN
    matches = flann.knnMatch(query_descriptors, target_descriptors, k=2)

    # Apply ratio test to filter out poor matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Lowe's ratio test
            good_matches.append(m)

    # Draw matches between query image and target image
    result_image = cv2.drawMatches(query_image, query_keypoints, target_image, target_keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # If there are enough good matches, apply RANSAC for homography
    if len(good_matches) > 10:
        # Extract location of good matches
        query_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        target_pts = np.float32([target_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute Homography using RANSAC
        H, mask = cv2.findHomography(query_pts, target_pts, cv2.RANSAC, 5.0)

        # Draw bounding box in the target image if Homography is found
        h, w = query_image.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        if H is not None:
            # Project the bounding box
            dst = cv2.perspectiveTransform(pts, H)
            target_image_with_box = cv2.polylines(target_image.copy(), [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # Show images
            cv2.imshow("Query Image", query_image)
            cv2.imshow("Target Image with Detected Object", target_image_with_box)
        else:
            print("Homography could not be computed.")
    else:
        print("Not enough good matches were found - {}/{}".format(len(good_matches), 10))

    # Show the matched keypoints
    cv2.imshow("Matched Keypoints", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example Usage
query_image_path = 'query_image.jpg'  # Replace with path to your query image
target_image_path = 'target_image.jpg'  # Replace with path to your target image
hybrid_sift_object_detection(query_image_path, target_image_path)




import cv2
import numpy as np

def rolling_shutter_distance_estimation(video_path, camera_speed, frame_rate, sensor_height, focal_length):
    """
    Estimate distance based on rolling shutter effect from a video.
    
    Args:
    - video_path: Path to the input video file.
    - camera_speed: Speed of the camera (m/s).
    - frame_rate: Frame rate of the video (fps).
    - sensor_height: Height of the camera sensor (mm).
    - focal_length: Focal length of the camera (mm).
    
    Returns:
    - distances: List of estimated distances for each frame.
    """
    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Rolling shutter parameters
    rolling_shutter_time = 1 / frame_rate  # Time it takes to scan the entire frame
    pixel_height = sensor_height / cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Height per pixel in mm
    
    distances = []
    
    # Read frames from the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect edges (you can use any other detection method)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours in the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Assume the largest contour is our object of interest (adjust based on your needs)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Measure the vertical distortion
            distortion = abs(h - (rolling_shutter_time * camera_speed * focal_length / sensor_height))
            
            # Estimate distance using distortion
            distance = (focal_length * sensor_height) / (distortion * pixel_height)
            distances.append(distance)
            
            # Draw bounding box on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with distance estimation
        cv2.imshow("Rolling Shutter Distance Estimation", frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()
    
    return distances

# Parameters
video_path = 'rolling_shutter_video.mp4'  # Replace with your video file path
camera_speed = 10.0  # m/s (example speed)
frame_rate = 30.0  # frames per second
sensor_height = 4.3  # mm (example value for a mobile camera)
focal_length = 4.0  # mm (example focal length)

# Call the function to estimate distance
distances = rolling_shutter_distance_estimation(video_path, camera_speed, frame_rate, sensor_height, focal_length)
print("Estimated distances:", distances)


#################-----------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image captured with a rolling shutter camera
image_path = 'rolling_shutter_image.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use edge detection to find the distorted object's edges
edges = cv2.Canny(gray_image, 50, 150)

# Use Hough Line Transform to detect skewed lines in the distorted image
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

# Draw the detected lines on the image to visualize them
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the original image with detected lines
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines in Distorted Image")
plt.show()

# Calculate skew angle for distance estimation
def calculate_skew_angle(lines):
    """
    Calculate the average skew angle of detected lines due to the rolling shutter effect.
    
    Args:
        lines (numpy array): Array of detected lines in the image.
        
    Returns:
        float: Average skew angle in degrees.
    """
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi  # Calculate the angle of each line in degrees
            angles.append(angle)
    
    # Calculate the average angle
    return np.mean(angles) if angles else 0

# Calculate the skew angle from the detected lines
skew_angle = calculate_skew_angle(lines)

# Estimate distance from skew angle
def estimate_distance_from_skew(skew_angle, object_speed, camera_readout_time):
    """
    Estimates the distance to an object using the skew angle caused by the rolling shutter effect.
    
    Args:
        skew_angle (float): Average skew angle in degrees.
        object_speed (float): Speed of the moving object in m/s.
        camera_readout_time (float): Time it takes to read the entire sensor in seconds.
    
    Returns:
        float: Estimated distance to the object in meters.
    """
    skew_radians = np.deg2rad(skew_angle)  # Convert angle to radians
    # Estimate distance using the known speed, skew angle, and readout time
    estimated_distance = (object_speed * camera_readout_time) / np.tan(skew_radians)
    return estimated_distance

# Constants
object_speed = 2  # Object speed in meters per second (m/s)
camera_readout_time = 0.05  # Camera readout time in seconds

# Estimate the distance using the skew angle
estimated_distance = estimate_distance_from_skew(skew_angle, object_speed, camera_readout_time)
print(f"Estimated Distance to Object: {estimated_distance:.2f} meters")

