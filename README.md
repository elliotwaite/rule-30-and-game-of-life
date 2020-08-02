# Rule 30 and Game of Life

This code generates a 2D animation of a 1D cellular automaton, Rule 30 (or other rules), being fed as input to a 2D cellular automaton, Conway’s Game of Life.

Video demo of Rule 30: https://youtu.be/IK7nBOLYzdE

[<img src="https://img.youtube.com/vi/IK7nBOLYzdE/hqdefault.jpg">](https://www.youtube.com/watch?v=IK7nBOLYzdE)

Video demo of Rule 110: https://youtu.be/P2uhhAXd7PI

[<img src="https://img.youtube.com/vi/P2uhhAXd7PI/hqdefault.jpg">](https://www.youtube.com/watch?v=P2uhhAXd7PI)

## Requirements

The following Python packages are required (I use a combination of Conda and Pip):
```
conda install colour imageio numpy opencv scipy tqdm

pip install imutils
```
A version of ffmpeg that support the libx264 encoder is also required. I use the Homebrew version.

To install [Homebrew](https://brew.sh/) (if you don't already have it installed):
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```
To install ffmpeg:
```
brew install ffmpeg
```
Note: If you want use a version of ffmpeg other than the Homebrew version, you'll have to change the `FFMPEG_PATH` value in `video_writer.py` to the path where your ffmpeg executable file is installed.

## Running the Code

Just run:
```
python rule_30_and_game_of_life.py
```

To change the settings for the output video, just edit the constants at the top of the `rule_30_and_game_of_life.py` file.

For example:
```
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160
SECS = int(60 * 3.5)  # 3 mins 30 secs.
PIXEL_SIZE = 6
OUTPUT_PATH = 'videos/youtube-3m-30s-6px.mp4'

FPS = 30  # Frames per second.
HIGH_QUALITY = True
```

If you set `HIGH_QUALITY = False`, a slightly lower quality `.avi` video will be generated, but it will take less time to render, usually about half the time of the high-quality version. This is can be useful for generating preview versions when still experimenting with different settings.

The low quality renderer uses OpenCV's VideoWriter. The high quality renderer writes all the frames to PNG image files, then combines those image files into a video using FFmpeg.

## License

[MIT](LICENSE)
