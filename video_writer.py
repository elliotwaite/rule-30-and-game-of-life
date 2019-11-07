import glob
import os
import pathlib
import shutil
import subprocess
import tempfile

from cv2 import VideoWriter
from cv2 import VideoWriter_fourcc
import imageio
import numpy as np

# Set `FFMPEG_PATH` to the local path of your installation of ffmpeg.
# The conda version doesn't support the libx264 encoder, so I use the
# Homebrew version: https://formulae.brew.sh/formula/ffmpeg
FFMPEG_PATH = glob.glob('/usr/local/Cellar/ffmpeg/*/bin/ffmpeg')[-1]


class Writer:
  """A class for creating video files from frames of Numpy arrays.

  Args
    fps: (int) The frames per second of the video.
    high_quality: (bool) If true, the quality of the output video will be
        higher, but it will take longer to render (about twice as long).
        The lower quality writer uses OpenCV's VideoWriter.
        The higher quality writer writes all the frames to PNG image files,
        then combines those image files into a video using FFmpeg.
  """
  def __init__(self, fps, high_quality=True):
    self.writer = (HighQualityWriter(fps) if high_quality else
                   LowQualityWriter(fps))

  def add_frame(self, frame):
    """Adds a frame to the video.

    Args:
      frame: (uint8 Numpy array of shape: (video height, video width, 3))
          The RGB data of the frame to add to the video. All frames must have
          the same width and height.
    """
    self.writer.add_frame(frame)

  def write(self, output_path):
    """Writes the video file to the output path.

    Args:
      output_path: (string) The path where the video file will be saved to.
    """
    self.writer.write(output_path)


class LowQualityWriter:
  def __init__(self, fps=30):
    self.fps = fps
    self.tmp_dir = None
    self.tmp_video_path = None
    self.video_writer = None

  def _initialize_video(self, frame):
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.tmp_video_path = os.path.join(self.tmp_dir.name, 'video.avi')
    fourcc = VideoWriter_fourcc(*'MJPG')
    height, width, _ = frame.shape
    self.video_writer = VideoWriter(
        self.tmp_video_path, fourcc, float(self.fps), (width, height))

  def add_frame(self, frame):
    if self.tmp_dir is None:
      self._initialize_video(frame)

    self.video_writer.write(np.flip(frame, axis=2))

  def write(self, output_path):
    self.video_writer.release()
    abs_output_path = pathlib.Path(output_path).with_suffix('.avi').absolute()
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    shutil.move(self.tmp_video_path, abs_output_path)
    self.tmp_dir.cleanup()
    self.tmp_dir = None
    print(f'Video written to: {abs_output_path}')


class HighQualityWriter:
  def __init__(self, fps=30):
    self.fps = fps
    self.tmp_dir = None
    self.cur_frame = 0

  def _initialize_video(self):
    self.tmp_dir = tempfile.TemporaryDirectory()
    self.cur_frame = 0

  def add_frame(self, frame):
    if self.tmp_dir is None:
      self._initialize_video()

    frame_path = os.path.join(self.tmp_dir.name, f'{self.cur_frame}.png')
    imageio.imwrite(frame_path, frame)
    self.cur_frame += 1

  def write(self, output_path):
    abs_tmp_dir_path = pathlib.Path(self.tmp_dir.name).absolute()
    abs_output_path = pathlib.Path(output_path).absolute()
    os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
    subprocess.call([FFMPEG_PATH,
                     '-framerate', f'{self.fps}',  # Frames per second.
                     '-i', f'{abs_tmp_dir_path}/%d.png',  # Input file pattern.
                     '-vcodec', 'libx264',  # Codec.

                     # Ensures players can decode the H.264 format.
                     '-pix_fmt', 'yuv420p',

                     # Video quality, lower is better, but zero (lossless)
                     # doesn't work.
                     '-crf', '1',

                     '-y',  # Overwrite output files without asking.
                     abs_output_path  # Output path.
                     ])
    self.tmp_dir.cleanup()
    self.tmp_dir = None
    print(f'Video written to: {abs_output_path}')
