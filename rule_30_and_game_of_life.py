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

FPS = 30  # Frames per second.
HIGH_QUALITY = True

STATE_WIDTH = VIDEO_WIDTH // PIXEL_SIZE
STATE_HEIGHT = VIDEO_HEIGHT // PIXEL_SIZE
NUM_FRAMES = SECS * FPS


class Rule30AndGameOfLife:
  def __init__(self, width, height,
               gol_percentage=0.5,
               num_frames=NUM_FRAMES):
    self.width = width
    self.height = height

    self.gol_height = int(height * gol_percentage)
    self.gol_state = np.zeros((self.height, self.width), np.uint8)

    self.row_padding = num_frames // 2
    self.row_width = self.width + self.row_padding * 2
    self.row = np.zeros(self.row_width, np.uint8)
    self.row[self.row_width // 2] = 1

    self.rows_height = self.height - self.gol_height
    self.rows = np.concatenate((
        np.zeros((self.rows_height - 1, self.width), np.uint8),
        self.row[None, self.row_padding:-self.row_padding]
    ))

    self.row_neighbors = np.array([1, 2, 4], dtype=np.uint8)
    self.gol_neighbors = np.array([[1, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 1]], dtype=np.uint8)
    self.rule = 30
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

    self.colors = (np.array(rgb_list, np.float) * 255).astype(np.uint8)

    self.decay = np.full((self.height, self.width), len(self.colors) - 1,
                         np.int)

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
    # Update rows.
    rule_index = signal.convolve2d(self.row[None, :],
                                   self.row_neighbors[None, :],
                                   mode='same', boundary='wrap')
    self.row = self.rule_kernel[rule_index[0]]
    transfer_row = self.rows[:1]
    self.rows = np.concatenate((
        self.rows[1:],
        self.row[None, self.row_padding:-self.row_padding]
    ))

    # Update gol_state.
    num_neighbors = signal.convolve2d(self.gol_state, self.gol_neighbors,
                                      mode='same', boundary='wrap')
    self.gol_state = np.logical_or(num_neighbors == 3,
                                   np.logical_and(num_neighbors == 2,
                                                  self.gol_state)
                                   ).astype(np.uint8)

    self.gol_state = np.concatenate((
        np.zeros((1, self.width), np.uint8),
        self.gol_state[1:-1],
        transfer_row
    ))

  def update_decay(self):
    full_state = np.concatenate(
        (self.gol_state[-self.gol_height:], self.rows), axis=0)
    self.decay += 1
    self.decay = np.clip(self.decay, None, len(self.colors) - 1)
    self.decay *= 1 - full_state

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
