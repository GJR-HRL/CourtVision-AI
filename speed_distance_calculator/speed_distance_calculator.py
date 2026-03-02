import os
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import measure_distance



# why are we using top view to calculate the distances ?
# Reason: The camera view creates WRONG distances.
# Because your video is taken from an angle, not from the top.
# Imagine this:
#   A player moves 5 pixels at the bottom of the screen → maybe 0.2 m
#   A player moves 5 pixels at the top far away → maybe 1.5 m 


# Tactical view removes all camera distortion
# top-view (bird’s-eye view) gives:
#  Every pixel represents the same distance
#  Movement in all areas is treated equally


class SpeedAndDistanceCalculator():
    """
    Calculates the speed and distance with the help of top view points positions.
    """
    def __init__(self, 
                 width_in_pixels,
                 height_in_pixels,
                 width_in_meters,
                 height_in_meters,
                 ):
        
        """
            width_in_pixels:
                width of top-view court image 
            height_in_pixels:
                height of top-view court image

            width_in_meters:
                width of court in meters 
            height_in_meters:
                height of court in meters 
        """
        
        self.width_in_pixels = width_in_pixels
        self.height_in_pixels= height_in_pixels

        self.width_in_meters = width_in_meters
        self.height_in_meters= height_in_meters

    def calculate_distance(self,
                            tactical_player_positions
                            ):
        """
            calculates the distance between the players
            Args:
                tactical_player_positions(list):
                    list of positions of the top view players in pixels
        """ 
        previous_players_position = {}
        output_distances =[]

        for frame_number, tactical_player_position_frame in enumerate(tactical_player_positions):
            output_distances.append({})

            # For every new frame, We check : Where was this player before? Where is he now?

            # Example:Frame 10 → (200px, 300px)
            #         Frame 11 → (210px, 295px)

            # Convert pixel movement → meter movement
            # “If 300 pixels = 15 meters, then 1 pixel = 0.05 meters.”

            # using Euclidean distance we calculate the distance in meters

            # This gives how much the player moved in meters between frames.
            for player_id, current_player_position in tactical_player_position_frame.items():
        
                if player_id in previous_players_position:
                    previous_position = previous_players_position[player_id]
                    # Convert pixel movement → meter movement
                    meter_distance = self.calculate_meter_distance(previous_position, current_player_position)
                    output_distances[frame_number][player_id] = meter_distance

                previous_players_position[player_id]=current_player_position
        
        return output_distances

    def calculate_meter_distance(self,previous_pixel_position, current_pixel_position):
         # using width_in_pixels,height_in_pixels and width_in_meters,height_in_meters Calculate the meter distance betweent current position and previous position
         previous_pixel_x, previous_pixel_y = previous_pixel_position
         current_pixel_x, current_pixel_y = current_pixel_position

        # Cross mulitplication between propotions of pixel and meter values 
        #     x                           pixel_x
        #    ---                  =        ---
        #  width in meters             width in pixels

        # X = pixel_x * width in meters / width in pixels


         previous_meter_x = previous_pixel_x * self.width_in_meters / self.width_in_pixels
         previous_meter_y = previous_pixel_y * self.height_in_meters / self.height_in_pixels

         current_meter_x = current_pixel_x * self.width_in_meters / self.width_in_pixels
         current_meter_y = current_pixel_y * self.height_in_meters / self.height_in_pixels

         meter_distance =measure_distance((current_meter_x,current_meter_y),
                                          (previous_meter_x,previous_meter_y)
                                          )
        # penalize the output as distance might be over estimated.
        # Suppose pixel distance = 5px.
        # Homography may convert this to:
        # Correct world distance ≈ 0.30 m
        # But because of slight camera tilt, it outputs ≈ 0.75 m (overestimate)
        # So you apply:
        # meter_distance = meter_distance * 0.4
        # This shrinks exaggerated world distances.

        # Why overestimation happens more than underestimation
        # Due to projection geometry:A 1 pixel change near the bottom of the frame = small meter change.
        # A 1 pixel change near the top of the frame = large meter change.
         meter_distance = meter_distance*0.4
         return meter_distance

    def calculate_speed(self, distances, fps=30):
        """
        Calculate player speeds based on distances covered over the last 5 frames.
        
        Args:
            distances (list): List of dictionaries containing distance per player per frame,
                            as output by calculate_distance method.
            fps (float): Frames per second of the video, used to calculate elapsed time.
            
        Returns:
            list: List of dictionaries where each dictionary maps player_id to their
                speed in km/h at that frame.
        """

        # Speed = Distance ÷ Time
        # so we calculate total distance between 5 frames here and time of the 5 frames and converting it into km/hr.


        speeds = []
        window_size = 5  # Look at last 5 frames for speed calculation
        
        # Calculate speed for each frame and player
        for frame_idx in range(len(distances)):
            speeds.append({})
            # For each player in current frame
            for player_id in distances[frame_idx].keys():
                # look the frames detected mentioned in window_size among 3 times window_size in original distance predictions
                start_frame = max(0, frame_idx - (window_size * 3) + 1)
                
                total_distance = 0
                frames_present = 0
                last_frame_present = None
                
                # Calculate total distance in the window
                for i in range(start_frame, frame_idx + 1):
                    if player_id in distances[i]:
                        if last_frame_present is not None:
                            total_distance += distances[i][player_id]
                            frames_present += 1
                        last_frame_present = i
                
                # Calculate speed only if player was present in at least two frames
                if frames_present >= window_size:
                    # Calculate time in hours (convert frames to hours)
                    # Convert number of frames → time
                    # If you used 5 frames: time_in_seconds = frames_present / fps 
                    # 5 frames at 30 fps = 5 / 30 = 0.166 seconds
                    time_in_seconds = frames_present / fps

                    # convert seconds in to hours
                    time_in_hours = time_in_seconds / 3600
                    
                    # Calculate speed in km/h
                    if time_in_hours > 0:
                        speed_kmh = (total_distance / 1000) / time_in_hours
                        speeds[frame_idx][player_id] = speed_kmh
                    else:
                        speeds[frame_idx][player_id] = 0
                else:
                    speeds[frame_idx][player_id] = 0
        
        return speeds