import numpy as np
import cv2

class TeamBallControllerPlotter: 
    def __init__(self):
        pass


    def get_team_ball_control(self, player_assignment, ball_aquisition):
        """
        Calculate which team has ball control for each frame.

        Args:
            player_assignment (list): A list of dictionaries indicating team assignments for each player in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.

        Returns:
            numpy.ndarray: An array indicating which team has ball control for each frame
                (1 for Team 1, 2 for Team 2, -1 for no control).
        """

        team_ball_control = []

        # ZIP is basically a frame synchronizer.
        # It guarantees:
        # frame0 of list A pairs with frame0 of list B
        # frame1 of list A pairs with frame1 of list B

        # | Frame | player_assignment[frame] | ball_aquisition[frame] |
        # | ----- | ------------------------ | ---------------------- |
        # | 0     | {2:1, 4:2}               | 4                      |
        # | 1     | {3:2, 6:1}               | -1                     |
        # | 2     | {1:1, 8:2}               | 8                      |




        for player_assignment_frame , ball_aquisition_frame in zip(player_assignment , ball_aquisition):
            if ball_aquisition_frame == -1:
                team_ball_control.append(-1)
                continue

            if ball_aquisition_frame not in player_assignment_frame:
                team_ball_control.append(-1)
                continue

            # finding the player which has the ball from detected players in that frame
            if player_assignment_frame[ball_aquisition_frame] == 1:
                team_ball_control.append(1)
                continue

            else : 
                team_ball_control.append(2)

        # Converting to numpy array so manipulations are easy
        team_ball_control = np.array(team_ball_control)
        return team_ball_control

    
    def plot_frame(self,frame , frame_num , team_ball_control):
        """
        Draw a semi-transparent overlay of team ball control percentages on a single frame.

        Args:
            frame (numpy.ndarray): The current video frame on which the overlay will be drawn.
            frame_num (int): The index of the current frame.
            team_ball_control (numpy.ndarray): An array indicating which team has ball control for each frame.

        Returns:
            numpy.ndarray: The frame with the semi-transparent overlay and statistics.
        """
        # Plot Semi transparent rectangle
        overlay = frame.copy()
        font_scale  = 0.7 
        font_thickness = 2 

        # Overlay position 
        frame_height , frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.60) 
        rect_y1 = int(frame_height * 0.75)
        rect_x2 = int(frame_width * 0.99)  
        rect_y2 = int(frame_height * 0.90)

        # text positions 
        text_x = int(frame_width * 0.63)  
        text_y1 = int(frame_height * 0.80)  
        text_y2 = int(frame_height * 0.88)


        cv2.rectangle(overlay, (rect_x1,rect_y1),(rect_x2,rect_y2), (255,255,255), -1)
        alpha = 0.8 
        # adding transparent bg 
        cv2.addWeighted(overlay, alpha, frame,1 - alpha , 0 , frame)

        # getting all the ball control till the current frame  
        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # filters and sums the array which has value of 1
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        # finding controll percentage 
        team_1 = team_1_num_frames/(team_ball_control_till_frame.shape[0])
        team_2 = team_2_num_frames/(team_ball_control_till_frame.shape[0])

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), font_thickness)

        return frame


    def plot(self,video_frames , player_assignment, ball_aquisition):
        """
        Plot team ball control statistics on a list of video frames.

        Args:
            video_frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            player_assignment (list): A list of dictionaries indicating team assignments for each player in the corresponding frame.
            ball_aquisition (list): A list indicating which player has possession of the ball in each frame.

        Returns:
            list: A list of frames with team ball control statistics drawn on them.
        """
        team_ball_control = self.get_team_ball_control(player_assignment,ball_aquisition)

        output_video_frames = []
        for frame_num , frame in enumerate(video_frames):
            frame_drawn= self.plot_frame(frame,frame_num,team_ball_control)
            output_video_frames.append(frame_drawn)
        
        return output_video_frames




        