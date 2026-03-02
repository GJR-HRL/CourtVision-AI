"""
Homography Matrix (H): This is a 3x3 matrix that defines the mapping between corresponding points in two images. It is used to transform homogeneous coordinates.
Eg. If there is a person in both images with different angel with aleast 4 points marked on him. If he has a ball in his hand and we know the coordinates of that ball in one image , we can find that same ball from second image taking ball is same position as first image. 

Homogeneous coordinates are a mathematical system where points are represented by an extra dimension, allowing geometric transformations like translation, rotation, and scaling to be expressed as simple matrix multiplications
Eg. (1,2) can be represented as (1,2,1)
"""
import numpy as np
import cv2 

class Homography:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        """
         transforms the keypoints from court into top-view with cv2.findHomography()
         Args: 
             source(np.array) : keypoints from court 
             target(np.array) : keypoints from Top-View image
        """

        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")
        
        # Expected input for cv2.findHomography() function
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        
        # cv2.findHomography(): This function is used to calculate the homography matrix. It requires at least four pairs of corresponding points (source points and destination points) to estimate the matrix. It can also employ robust methods like RANSAC to handle outliers in the point matches.

        # self.m carries the matrix used in tranforming the points 
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
            Using the function of cv2.findHomography() we find the four points from court video and transform them into top view.
        """
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")
        
        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.m)
        return points.reshape(-1, 2).astype(np.float32)




