from utils import read_video,save_video
from ball_aquisition import BallAquisitionDetector
from tracker import PlayerTracker,BallTracker
from team_assigner import TeamAssigner
from court_key_detector import CourtKeyDetector
from pass_interception_detection import PassAndInterceptionDetector
from top_view_converter import TopViewConverter
from speed_distance_calculator import SpeedAndDistanceCalculator
from plotters import (
    PlayerTracksPlotter,
    BallTracksPlotter,
    TeamBallControllerPlotter,
    PassInterceptionPlotter,
    CourtKeypointPlotter,
    TopViewPlotter,
    SpeedAndDistancePlotter
)


def main():
    # Read Video 
    video_frames = read_video("training-input-videos/video_1.mp4")


    # Initialize Trackers
    player_tracker = PlayerTracker("models/player_detector.pt")
    ball_tracker = BallTracker("models/ball_detector.pt")


    # Initialise court key trackers 
    court_keypoints_detector = CourtKeyDetector("models/court_keypoint_detector.pt")
    

    # Run Trackers
    player_tracks = player_tracker.get_object_tracks(video_frames, read_from_stub=True,stub_path="stubs/player_track_stubs.pkl")
    ball_tracks = ball_tracker.get_object_tracks(video_frames, read_from_stub=True,stub_path="stubs/ball_track_stubs.pkl")


    # Court key points
    court_keypoints_per_frame = court_keypoints_detector.get_court_keypoints(video_frames,read_from_stub=True,stub_path="stub/court_keypoints_stubs.pkl")


    # Remove wrong ball Detections 
    ball_tracks = ball_tracker.remove_wrone_detections(ball_tracks)

    # Interpolate Ball Tracks
    ball_tracks =  ball_tracker.interpolate_ball_position(ball_tracks)


    # Assign Players Team
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,player_tracks,read_from_stub=True,stub_path="stubs/player_team_assignment.pkl")


    # Ball Acquisition
    ball_aquisition_detector = BallAquisitionDetector()
    ball_aquisition =  ball_aquisition_detector.detect_ball_possession(player_tracks,ball_tracks)

    
    # Detect Passes
    pass_and_interception_detector = PassAndInterceptionDetector()
    passes = pass_and_interception_detector.detect_passes(ball_aquisition,player_assignment)
    interceptions = pass_and_interception_detector.detect_interceptions(ball_aquisition,player_assignment)


    # Top View
    top_view_converter = TopViewConverter(court_image_path="./images/basketball_court.png")
    court_keypoints_per_frame = top_view_converter.validate_keypoints(court_keypoints_per_frame)

    top_view_positions = top_view_converter.transform_players_to_top_view(court_keypoints_per_frame,player_tracks)


    # Calculate Speed and Distance of each player
    speed_and_distance_calculator = SpeedAndDistanceCalculator(
        top_view_converter.width,
        top_view_converter.height,
        top_view_converter.actual_width_in_meters,
        top_view_converter.actual_height_in_meters
    )
    
    player_distances_per_frame = speed_and_distance_calculator.calculate_distance(top_view_positions)
    player_speed_per_frame = speed_and_distance_calculator.calculate_speed(player_distances_per_frame)


    # Initialize Plotters
    player_tracks_plotter = PlayerTracksPlotter()
    ball_tracks_plotter = BallTracksPlotter()
    team_ball_controller_plotter=TeamBallControllerPlotter()
    pass_and_interception_plotter=PassInterceptionPlotter()
    court_keypoint_plotter = CourtKeypointPlotter()
    top_view_plotter =TopViewPlotter()
    speed_distance_plotter = SpeedAndDistancePlotter()


    # Plotting object tracks
    # Plots each player ellipse
    output_video_frames = player_tracks_plotter.plot(video_frames,player_tracks,)

    # Plots ball 
    output_video_frames = ball_tracks_plotter.plot(output_video_frames, ball_tracks)

    # Plot Team Ball Control
    output_video_frames = team_ball_controller_plotter.plot(output_video_frames,player_assignment,ball_aquisition)

    # Plot Passes and Interceptions
    output_video_frames = pass_and_interception_plotter.draw(output_video_frames,passes,interceptions)

    # Plot Court keypoints
    output_video_frames = court_keypoint_plotter.plot(output_video_frames, court_keypoints_per_frame)


    # Plot Top view of player
    output_video_frames = top_view_plotter.plot(output_video_frames,
                                                top_view_converter.court_image_path,
                                                top_view_converter.width,
                                                top_view_converter.height,             
                                                top_view_converter.key_points,
                                                top_view_positions,
                                                player_assignment,
                                                ball_aquisition
                                                )


    output_video_frames = speed_distance_plotter.plot(output_video_frames,
                                                         player_tracks,
                                                         player_distances_per_frame,
                                                         player_speed_per_frame
                                                         )


    # Save Video
    save_video(output_video_frames,"output_video/output_video.avi")
                     

if __name__ == "__main__":
    main()