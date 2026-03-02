from PIL import Image
import cv2
from transformers import CLIPProcessor , CLIPModel

import sys
sys.path.append('../')
from utils import read_stub, save_stub


# Zero - shot Classifier takes in the image as input and classifies into classes which it has not seen with the help of training data.


# so it can take our image and string classes and give the classification based on its training data.
class TeamAssigner:
    """
    A class that assigns players to teams based on their jersey colors using visual analysis.

    The class uses a pre-trained vision model to classify players into teams based on their
    appearance. It maintains a consistent team assignment for each player across frames.

    Attributes:
        team_colors (dict): Dictionary storing team color information.
        player_team_dict (dict): Dictionary mapping player IDs to their team assignments.
        team_1_class_name (str): Description of Team 1's jersey appearance.
        team_2_class_name (str): Description of Team 2's jersey appearance.
    """
    def __init__(self, team_1_class_name = "white shirt",team_2_class_name = "dark blue shirt" ):
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.player_team_dict = {}
        self.team_colors = {}


    def load_model(self):
        """
        Loads the pre-trained vision model for jersey color classification.
        """
        # CLIPModel takes the processed inputs and performs the core task of generating embeddings and computing their similarity.
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")

        # CLIPProcessor handles the necessary pre-processing steps for both modalities (image and text).
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")



    def get_player_color(self,frame,bbox):
        """
        Analyzes the jersey color of a player within the given bounding box.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            bbox (tuple): Bounding box coordinates of the player.

        Returns:
            str: The classified jersey color/description.
        """

        # x1 , y1 , x2 ,y2
        # takes slices of frame from y1 till y2 and x1 till x2
        image = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]):int(bbox[2])]  

        # Convert to PIL image as our CLIP requires PIL
        rgb_image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        classes = [self.team_1_class_name, self.team_2_class_name]

        # Does Preprocessing
        inputs = self.processor(text=classes, images=pil_image, return_tensors="pt", padding=True)

        # In function call, ** unpacks a dictionary, treating its key-value pairs as keyword arguments for the function.
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        class_name = classes[probs.argmax(dim=1)[0]]

        return class_name
    

    def get_player_team(self, frame , player_bbox , player_id):
        """
        Gets the team assignment for a player, using cached results if available.

        Args:
            frame (numpy.ndarray): The video frame containing the player.
            player_bbox (tuple): Bounding box coordinates of the player.
            player_id (int): Unique identifier for the player.

        Returns:
            int: Team ID (1 or 2) assigned to the player.
        """

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame,player_bbox)

        team_id = 2 
        if player_color == self.team_1_class_name:
            team_id=1

        self.player_team_dict[player_id] = team_id
        return team_id



    def get_player_teams_across_frames(self, video_frames, player_tracks , read_from_stub=False , stub_path = None):
        """
        Processes all video frames to assign teams to players, with optional caching.

        Args:
            video_frames (list): List of video frames to process.
            player_tracks (list): List of player tracking information for each frame.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries mapping player IDs to team assignments for each frame.
        """

        player_assignment = read_stub(read_from_stub, stub_path)

        # Checks if all the frames where present.
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment
            
        self.load_model()

        player_assignment = []

        # Loop through the entire video and return the final team assignment for each players
        for frame_num , player_tracks in enumerate(player_tracks):
            player_assignment.append({})


            # We will recalculate after 50 frames
            if frame_num % 50 == 0:
                self.player_team_dict = {}


            for player_id, track in player_tracks.items():
                team = self.get_player_team(video_frames[frame_num], track['bbox'], player_id)

                player_assignment[frame_num][player_id] = team

        
        save_stub(stub_path,player_assignment)

        return player_assignment

            

                