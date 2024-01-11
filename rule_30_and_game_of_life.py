import colour
import cv2
import imutils
import numpy as np
from scipy import signal
import tqdm

import video_writer

# # Instagram.
# VIDEO_WIDTH = 1080
# VIDEO_HEIGHT = 1350
# SECS = 60
# PIXEL_SIZE = 5
# OUTPUT_PATH = 'videos/instagram-60s-5px.mp4'


# # Twitter.
# VIDEO_WIDTH = 1024
# VIDEO_HEIGHT = 1024
# SECS = 140  # 2 mins 20 secs.
# PIXEL_SIZE = 4
# OUTPUT_PATH = 'videos/twitter-2m-20s-4px.mp4'

# YouTube.
VIDEO_WIDTH = 3840
VIDEO_HEIGHT = 2160
SECS = int(60 * 3.5)  # 3 mins 30 secs.
PIXEL_SIZE = 6
OUTPUT_PATH = 'videos/youtube-3m-30s-6px.mp4'

# YouTube channel art.
# VIDEO_WIDTH = 2880
# VIDEO_HEIGHT = 1800
# SECS = 60
# PIXEL_SIZE = 4
# OUTPUT_PATH = 'videos/youtube-channel-art-2.mp4'

FPS = 60  # Frames per second.
HIGH_QUALITY = True

STATE_WIDTH = VIDEO_WIDTH // PIXEL_SIZE
STATE_HEIGHT = VIDEO_HEIGHT // PIXEL_SIZE
NUM_FRAMES = SECS * FPS

# `RULE` specifies which cellular automaton rule to use.
RULE = 30

# `X_OFFSET` specifies how far from the center to place the initial first pixel.
X_OFFSET = 0

# These settings can be used for rule 110, which only grows to the left, so we
# offset the starting pixel to be close to the right edge of the screen.
# RULE = 110
# X_OFFSET = VIDEO_WIDTH // PIXEL_SIZE // 2 - 1 - 60 * 4

# The Game of Life state wraps across the left and right edges of the state,
# and dies out at the top of the state (all values of the top row are zero).
# By adding padding to the state, you extend the state beyond the edges of the
# visible window, essentially hiding the wrapping and/or dying out aspects of
# the state.
GOL_STATE_WIDTH_PADDING = VIDEO_WIDTH
GOL_STATE_HEIGHT_PADDING = VIDEO_HEIGHT


class Rule30AndGameOfLife:
  def __init__(self, width, height,
               gol_percentage=0.5,
               num_frames=NUM_FRAMES):
    self.width = width
    self.height = height

    self.gol_height = int(height * gol_percentage)
    self.gol_state_width = self.width + GOL_STATE_WIDTH_PADDING * 2
    self.gol_state_height = self.gol_height + GOL_STATE_HEIGHT_PADDING

    self.gol_state = np.zeros((self.gol_state_height, self.gol_state_width),
                              np.uint8)

    self.row_padding = num_frames // 2
    self.row_width = self.gol_state_width + self.row_padding * 2
    self.row = np.zeros(self.row_width, np.uint8)
    self.row[self.row_width // 2 + X_OFFSET] = 1

    self.rows_height = self.height - self.gol_height
    self.rows = np.concatenate((
        np.zeros((self.rows_height - 1, self.gol_state_width), np.uint8),
        self.row[None, self.row_padding:-self.row_padding]
    ))

    self.row_neighbors = np.array([1, 2, 4], dtype=np.uint8)
    self.gol_neighbors = np.array([[1, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 1]], dtype=np.uint8)
    self.rule = RULE
    self.rule_kernel = None
    self.update_rule_kernel()

    hex_colors = [
        '#711c91',
        '#ea00d9',
        '#0abdc6',
        '#133e7c',
        '#091833',
        '#000103'
    ]
    color_decay_times = [2 * 8 ** i for i in range(len(hex_colors) - 1)]
    assert len(hex_colors) == len(color_decay_times) + 1
    color_list = [colour.Color('white')]
    for i in range(len(hex_colors) - 1):
      color_list += list(colour.Color(hex_colors[i]).range_to(
          colour.Color(hex_colors[i + 1]), color_decay_times[i]))
    color_list += [colour.Color('black')]
    rgb_list = [c.rgb for c in color_list]

    self.colors = (np.array(rgb_list, np.float64) * 255).astype(np.uint8)

    self.decay = np.full((self.height, self.width), len(self.colors) - 1,
                         np.int_)

    self.rgb = None

    self.update_decay()
    self.update_rgb()

  def step(self):
    self.update_rows_and_gol_state()
    self.update_decay()
    self.update_rgb()

  def update_rule_kernel(self):
    self.rule_kernel = np.array([int(x) for x in f'{self.rule:08b}'[::-1]],
                                np.uint8)

  def update_rows_and_gol_state(self):
    # Update `rows` (the state of the 2D cellular automaton).
    rule_index = signal.convolve2d(self.row[None, :],
                                   self.row_neighbors[None, :],
                                   mode='same', boundary='wrap')
    self.row = self.rule_kernel[rule_index[0]]
    transfer_row = self.rows[:1]
    self.rows = np.concatenate((
        self.rows[1:],
        self.row[None, self.row_padding:-self.row_padding]
    ))

    # Update `gol_state` (the state of the 3D cellular automaton).
    num_neighbors = signal.convolve2d(self.gol_state, self.gol_neighbors,
                                      mode='same', boundary='wrap')
    self.gol_state = np.logical_or(num_neighbors == 3,
                                   np.logical_and(num_neighbors == 2,
                                                  self.gol_state)
                                   ).astype(np.uint8)

    self.gol_state = np.concatenate((
        np.zeros((1, self.gol_state_width), np.uint8),
        self.gol_state[1:-1],
        transfer_row
    ))

  def update_decay(self):
    visible_state = np.concatenate(
        (self.gol_state[-self.gol_height:,
         GOL_STATE_WIDTH_PADDING:-GOL_STATE_WIDTH_PADDING],
         self.rows[:, GOL_STATE_WIDTH_PADDING:-GOL_STATE_WIDTH_PADDING]),
        axis=0)
    self.decay += 1
    self.decay = np.clip(self.decay, None, len(self.colors) - 1)
    self.decay *= 1 - visible_state

  def update_rgb(self):
    self.rgb = self.colors[self.decay]


def main():
  writer = video_writer.Writer(fps=FPS, high_quality=HIGH_QUALITY)

  animation = Rule30AndGameOfLife(STATE_WIDTH, STATE_HEIGHT)

  for _ in tqdm.trange(NUM_FRAMES):
    small_frame = animation.rgb
    enlarged_frame = imutils.resize(small_frame, VIDEO_WIDTH, VIDEO_HEIGHT,
                                    cv2.INTER_NEAREST)
    writer.add_frame(enlarged_frame)
    animation.step()

  writer.write(OUTPUT_PATH)


if __name__ == '__main__':
  main()
