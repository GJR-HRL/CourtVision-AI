"""
A utility module providing functions for drawing shapes on video frames.

This module includes functions to draw triangles and ellipses on frames, which can be used
to represent various annotations such as player positions or ball locations in sports analysis.
"""

import cv2
import numpy as np
import sys 
sys.path.append('../')
from utils import get_center_of_bbox,get_bbox_width

def draw_ellipse(frame,bbox,color,track_id=None):
    """
    Draws an ellipse and an optional rectangle with a track ID on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the ellipse.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the ellipse in BGR format.
        track_id (int, optional): The track ID to display inside a rectangle. Defaults to None.

    Returns:
        numpy.ndarray: The frame with the ellipse and optional track ID drawn on it.
    """

    # x1 = top left x coordinate
    # y1 = top left y coordinate
    # x2 = bottom right x coordinate
    # y2 = bottom right y coordinate

    y2 = int(bbox[3])
    x_center,_ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    # ellipse center starts from center(x_center) of foot(y2) , axes is width and height(35% of width) 
    cv2.ellipse(
        frame,
        center=(x_center,y2),
        axes=(int(width), int(0.35*width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color= color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    if track_id is not None:
        rectangle_width = 40
        rectangle_height=20
        # We cut the rectangle in half and take x_center steps towards left so rectangle at center
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15 

        cv2.rectangle(frame,
                      (int(x1_rect),int(y1_rect)),
                      (int(x2_rect),int(y2_rect)),
                      color,
                      cv2.FILLED)
        
        # possition of text inside the rectangle
        x1_text = x1_rect + 12

        # generally we wont label items more then 99 but still it would shifts the label towards left if happens 
        if track_id > 99:
            x1_text -=10

        cv2.putText(
            frame,
            f"{track_id}",
            (int(x1_text), int(y1_rect+15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,0),
            2
        )


    return frame


def draw_triangle(frame,bbox,color):
    """
     Draws a filled triangle on the given frame at the specified bounding box location.

    Args:
        frame (numpy.ndarray): The frame on which to draw the triangle.
        bbox (tuple): A tuple representing the bounding box (x, y, width, height).
        color (tuple): The color of the triangle in BGR format.

    Returns:
        numpy.ndarray: The frame with the triangle drawn on it.
    """
    # y1 = top left y coordinate
    y = int(bbox[1])
    x,_ = get_center_of_bbox(bbox)

    # Start at x and y
    #
    # x-10,y-20     x+10,y-20
    #      
    #           x,y
    #
    triangle_point = np.ndarray([
        [x,y],
        [x-10, y-20],
        [x+10,y-20],
    ])

    cv2.drawContours(frame,[triangle_point],0,color,cv2.FILLED)
    #Boundaries
    cv2.drawContours(frame,[triangle_point],0,color,2)

    return frame



    

    