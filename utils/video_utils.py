import cv2
import os 



def read_video(video_path):
    """
    Read all the frames from video 
    read() method reads each frame and returns ret(boolean) , numpy array of the frame 

    ret = 0 if there was error reading frame,end of the video , can't capture video from the source

    args: 
        video_path (str) : path to the input video
    returns:
        list of video frames as numpy array
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret , frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    return frames


def save_video(output_video_frames, output_video_path):

    # if folder does not exists , creates it
    if not os.path.exists(os.path.dirname(output_video_path)):
        os.makedirs(os.path.dirname(output_video_path))

    # VideoWriter_fourcc specifies the 4 digit codec code of video
    # (*'XVID') unpacks the character ('X', 'V', 'I', 'D')
    # this four digits represents the video codec of AVI format developed by microsoft

    # If a frame has a shape of (1080, 1920, 3) (Height, Width, Channels), the expression (output_video_frames[0].shape[1], output_video_frames[0].shape[0]) evaluates to (1920, 1080).
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path,fourcc,24,(output_video_frames[0].shape[1],output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()

    


