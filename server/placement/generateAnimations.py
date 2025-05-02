import os
import json
from Board import Board
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

# def list_files_in_dir(dir_path: str) -> list[str]:
#     try:
#         files = os.listdir(dir_path)
#         return files
#     except:
#         print("Not a directory.")
#         pass

# files = list_files_in_dir('frames/inital_test')

# for file in files:

#     with open(f'frames/inital_test/{file}', "r") as f:
#         frame_data = json.load(f)

#     frames = [
#         (Board.from_dict(frame["board"]), frame["lines_cleared"])
#         for frame in frame_data
#     ]

#     fig, ax = plt.subplots()
#     gray_map = plt.cm.get_cmap('gray')

#     im = ax.imshow(frames[0][0].board, cmap=gray_map.reversed(), vmin=0, vmax=1)

#     def frameUpdate(i):
#         board = frames[i][0].board
#         im.set_data(board)
#         ax.set_title(f'Frame {i}')
#         ax.set_ylabel(f'Lines Cleared: {frames[i][1]}', rotation=0)

#     ani = FuncAnimation(fig, frameUpdate, frames=len(frames), interval=200)
#     ani.save(f'gif/inital_test/{file}_animation.gif', dpi=100, writer='pillow')
#     plt.close()


with open(f'0_frames.json', "r") as f:
    frame_data = json.load(f)

frames = [
    (Board.from_dict(frame["board"]), frame["lines_cleared"])
    for frame in frame_data
]

fig, ax = plt.subplots()
gray_map = plt.cm.get_cmap('gray')

im = ax.imshow(frames[0][0].board, cmap=gray_map.reversed(), vmin=0, vmax=1)

def frameUpdate(i):
    board = frames[i][0].board
    im.set_data(board)
    ax.set_title(f'Frame {i}')
    ax.set_ylabel(f'Lines Cleared: {frames[i][1]}', rotation=0)

ani = FuncAnimation(fig, frameUpdate, frames=len(frames), interval=200)
ani.save(f'0_animation.gif', dpi=100, writer='pillow')
plt.close()


