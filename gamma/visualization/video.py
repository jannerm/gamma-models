import os
import math
from IPython.display import display, HTML
from base64 import b64encode
import skvideo.io

def display_video(images):
    ## write video to disk as mp4
    savepath = os.path.join(os.getcwd(), 'video.mp4')
    save_video(savepath, images)

    ## open mp4
    mp4 = open(savepath,'rb').read()

    ## embed mp4 in html
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML("""
        <div class="center">
        <video controls autoplay loop width=800 controls>
            <source src="%s" type="video/mp4">
        </video>
        </div>
        """ % data_url))

def mkdir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_video(filename, video_frames, duration=5, video_format='mp4'):
    """
        (very lightly) adapted from
            https://github.com/rail-berkeley/
            softlearning/blob/48393e5e645ff2f39d7dadb17956b6a75edee900/
            softlearning/utils/video.py
    """
    fps = int(math.ceil(len(video_frames) / duration))
    assert fps == int(fps), fps
    mkdir(filename)

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            '-r': str(int(fps)),
        },
        outputdict={
            '-f': video_format,
            '-pix_fmt': 'yuv420p', # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        }
    )