from ultralytics import YOLO 
import supervision as sv
import sys 
sys.path.append('../')
from utils import read_stub , save_stub


class PlayerTracker:
    """
    
    """


    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self,frames):
        """
            Detects the players from each frame without tracking and returns the detections as list 
        """ 
        batch_size = 20 
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self,frames,read_from_stub=False,stub_path=None):
        """
        We use only this method to get the final player tracks 
        It uses detect_frames to detect the players and supervision for tracking

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
            where each dictionary maps player IDs to their bounding box coordinates.

        """

        # check if we have already found the tracks for the video
        tracks = read_stub(read_from_stub,stub_path)
        
        # If the we find the tracks then check the len of the video frame and tracks , if they are same return tracks if not then we might have path of another video so we recalculate the tracks again and save to the given location.

        if tracks is not None: 
            if len(tracks) == len(frames):
                return tracks

        detections = self.detect_frames(frames)
        tracks = []

        for frame_num , detection in enumerate(detections):
            cls_names = detections.names

            # inverts the names of each key and value
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Converts YOLO output into supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            detection_with_tracks =  self.tracker.update_with_detections(detection_supervision)

            tracks.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                # eg tracks 
                # [
                # {12: {'bbox': [12, 12, 12]}}, 
                # {2: {'bbox': [12, 12, 12]}}, 
                # {5: {'bbox': [12, 12, 12]}}
                # ]

                if cls_id == cls_names_inv['Player']:
                    tracks[frame_num][track_id] = {"bbox":bbox}
                
            
            save_stub(stub_path,tracks)
            return tracks