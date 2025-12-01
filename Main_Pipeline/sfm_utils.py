import os
import cv2
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt

def load_images_and_paths(output_dir):
    resized_paths = []
    images = []

    # fetch the final paths of all the resized images
    for f in os.listdir(output_dir):
        ext = os.path.splitext(f)[1]
        if (ext.lower() in [".jpg", ".jpeg", ".png"]):
            full_path = os.path.join(output_dir, f)
            resized_paths.append(full_path)
            # print(f"Found File: {full_path}")

    # sort the paths according to the file names
    resized_paths = natsorted(resized_paths)

    # open each image path and store the resultant images in a list
    for img_path in resized_paths:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        else:
            print(f"Warning: Could not load image {img_path}")

    return resized_paths, images

def compute_features(image):
    gray_scaled = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Increase nfeatures to allow SIFT to detect more keypoints
    sift = cv2.SIFT_create(
        nfeatures=10000,
        contrastThreshold=0.01,
        edgeThreshold=5,
        sigma=1.2
    )

    keypoints, descriptors = sift.detectAndCompute(gray_scaled, None)
    return keypoints, descriptors

def compute_matches(descriptor1, descriptor2, ratio=0.65):
    # FLANN parameters
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    
    # Check if descriptors are available and have the correct data type
    if descriptor1 is None or descriptor2 is None:
        return []
        
    descriptor1 = descriptor1.astype(np.float32)
    descriptor2 = descriptor2.astype(np.float32)

    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        # identify matches (k=2 for ratio test)
        matches_found = flann_matcher.knnMatch(descriptor1, descriptor2, k=2)
    except cv2.error as e:
        print(f"FLANN matching error: {e}")
        return []
    
    # apply lowes ratio test using ratio threshold
    final_matches = []
    for m, n in matches_found:
        if m.distance < ratio * n.distance:
            final_matches.append(m)
            
    return final_matches

def get_aligned_points(kp1, kp2, matches):
    img1idx = []
    img2idx = []
    pts1 = []
    pts2 = []
    
    for m in matches:
        img1idx.append(m.queryIdx)
        img2idx.append(m.trainIdx)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

    img1idx = np.array(img1idx)
    img2idx = np.array(img2idx)
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    return pts1, pts2, img1idx, img2idx

def visualize_matches(img1, kp1, img2, kp2, matches, title="Feature Matches"):
    matches_visualized = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(20, 8))
    plt.imshow(matches_visualized)
    plt.title(title)
    plt.show()

def create_intrinsic_matrix(img_sample):
    h, w = img_sample.shape[:2]

    # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # Assuming fx = fy = w (simple pinhole model, typically a good starting point)
    K = np.array([[w, 0, w/2],
                  [0, w, h/2],
                  [0, 0, 1]], dtype=np.float64)
    return K