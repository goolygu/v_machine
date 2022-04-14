import cProfile
import glob
import gzip
import io
import os
import pickle
import pstats
import queue
import random
import time
import tkinter as tk
from functools import lru_cache
from multiprocessing import Process, Queue
from pstats import SortKey

import numpy as np
import sounddevice as sd
from PIL import Image, ImageEnhance, ImageTk


def get_key_frame_array(key_frame_buffer):
    return np.asarray(Image.open(key_frame_buffer), dtype=np.uint8)


class GUI:
    def __init__(
        self,
        video_dir,
        enable_profile=False,
        max_fps=30,
    ):
        self.loading_stage = 0
        self.video_dir = video_dir
        self.all_video_paths = sorted(glob.glob(f"{video_dir}/*.mtd"))
        self.current_vid_idx = 0
        print("loading video")
        mtd = self.load_video(self.all_video_paths[self.current_vid_idx])
        print("finish loading")
        self.mtd = None
        self.current_img_idx = None
        self.current_frame = None
        self.mtd_shape = None

        self.setup_mtd_video(mtd)
        self.tk = tk.Tk()

        self.image_size = (400, 400)

        self.canvas = tk.Canvas(self.tk, width=700, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.configure(background="black")
        self.canvas.configure(highlightbackground="black")

        self.instruction = tk.Label(
            self.canvas,
            text="F11: toggle fullscreen, → next video, ← last video, ↑ increase change, ↓ decrease change.\n\nVisual Loop Machine by Li Yang Ku",
            fg="white",
            bg="black",
        )
        self.instruction_label = self.canvas.create_window(
            350, 550, window=self.instruction
        )

        orig_img = Image.fromarray(self.current_frame)
        img = orig_img.resize(self.image_size, Image.HAMMING)
        self.photoImg = ImageTk.PhotoImage(img)
        self.x_loc = 350
        self.y_loc = 300
        self.img_container = self.canvas.create_image(
            self.x_loc, self.y_loc, anchor=tk.CENTER, image=self.photoImg
        )

        self.state = False
        self.tk.bind("<F11>", self.toggle_fullscreen)
        self.tk.bind("<Escape>", self.end_fullscreen)
        self.tk.bind("<Key-Up>", self.up)
        self.tk.bind("<Key-Down>", self.down)
        self.tk.bind("<Key-Right>", self.right)
        self.tk.bind("<Key-Left>", self.left)
        self.tk.update_idletasks()
        self.tk.update()
        self.next = False
        self.full_size_img = False
        self.dim_1_dir = 0
        self.dim_0_dir = 0
        self.threshold = 0.9
        self.enable_profile = enable_profile
        if enable_profile:
            self.pr = cProfile.Profile()
        self.load_next = False
        self.load_previous = False
        self.brightness = 1
        self.pause = False
        self.max_fps = max_fps
        self._previous_t = time.time()
        self._estimated_image_time = 0

    def clear_mtd_memory(self):
        if hasattr(self, "mtd"):
            del self.mtd

        if hasattr(self, "key_frames"):
            del self.key_frames

    def setup_mtd_video(self, mtd):
        self.mtd = mtd
        first_idx = list(self.mtd.key_frames.keys())[0]
        self.current_img_idx = list(first_idx)
        self.current_frame = get_key_frame_array(self.mtd.key_frames[first_idx])
        self.mtd_shape = (mtd.diff_array_shape[0], mtd.diff_array_shape[1])

    @lru_cache(maxsize=500)
    def get_key_frame(self, mtd, img_idx):

        if img_idx in mtd.key_frames:
            return get_key_frame_array(mtd.key_frames[img_idx])
        return None

    @lru_cache(maxsize=500)
    def get_diff_image(self, mtd, dim0, dim1, dir):
        if mtd.diff_array[dim0][dim1][dir] is None:
            return None
        diff_image = (
            np.asarray(Image.open(mtd.diff_array[dim0][dim1][dir]), dtype=np.int16)
            - 128
        ) * 2
        return diff_image

    def load_video(self, path, q=None):
        fp = gzip.open(path, "rb")  # 91 94 95 97
        print(f"loading file {path}")
        mtd = pickle.load(fp)
        print(f"finish loading file")
        fp.close()
        if q is not None:
            q.put(mtd)
            print(f"finish put")
        return mtd

    def down(self, event=None):
        self.threshold += 0.1
        self.threshold = min(1.1, self.threshold)
        print(f"threshold {self.threshold}")

    def up(self, event=None):
        self.threshold -= 0.1
        self.threshold = max(0.5, self.threshold)
        print(f"threshold {self.threshold}")

    def right(self, event=None):
        if not self.load_previous:
            self.load_next = True

    def left(self, event=None):
        if not self.load_next:
            self.load_previous = True

    def toggle_fullscreen(self, event=None):
        if self.state is True:
            self.end_fullscreen()
            self.instruction_label = self.canvas.create_window(
                350, 550, window=self.instruction
            )
        else:
            self.canvas.delete(self.instruction_label)
            self.start_full_screen()

    def start_full_screen(self, event=None):
        canvas_size = [self.canvas.winfo_width(), self.canvas.winfo_height()]
        self.state = True
        self.tk.attributes("-fullscreen", True)
        self.full_size_img = True
        while True:
            self.tk.update()
            # check if canvas size changed before changing image size
            if canvas_size != [self.canvas.winfo_width(), self.canvas.winfo_height()]:
                size = min([self.canvas.winfo_width(), self.canvas.winfo_height()])
                self.x_loc = int(self.canvas.winfo_width() / 2)
                self.y_loc = int(self.canvas.winfo_height() / 2)
                self.image_size = (size, size)
                self.canvas.coords(self.img_container, self.x_loc, self.y_loc)

                break
        return

    def end_fullscreen(self, event=None):
        self.image_size = (400, 400)
        self.state = False
        self.tk.attributes("-fullscreen", False)
        self.full_size_img = False
        self.x_loc = 350
        self.y_loc = 300
        self.canvas.coords(self.img_container, self.x_loc, self.y_loc)
        if self.enable_profile:
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
        return

    def load_next_video(self, previous=False):
        if self.loading_stage == 0:
            self.loading_stage = 1

            if previous:
                self.current_vid_idx -= 1
            else:
                self.current_vid_idx += 1
            self.current_vid_idx = self.current_vid_idx % len(self.all_video_paths)
            print("loading video")
            self.q = Queue()
            p = Process(
                target=self.load_video,
                args=(self.all_video_paths[self.current_vid_idx], self.q),
            )
            p.start()
        elif self.loading_stage == 1:
            try:
                self.brightness = max(self.brightness - 0.02, 0)
                if self.brightness == 0:
                    mtd = self.q.get(block=False)
                    self.setup_mtd_video(mtd)
                    self.loading_stage = 2
            except queue.Empty:
                time.sleep(0.1)
                pass
        elif self.loading_stage == 2:
            self.brightness = min(self.brightness + 0.02, 1)
            if self.brightness == 1:
                self.load_next = False
                self.load_previous = False
                self.loading_stage = 0

    def set_next_image(self, dim_1_dir=0, dim_0_dir=0):
        self.dim_1_dir = dim_1_dir
        self.dim_0_dir = dim_0_dir
        self.next = True

    def update(self, force=False):
        if self.next is True or force:
            if self.load_next:
                self.load_next_video()
            elif self.load_previous:
                self.load_next_video(previous=True)
            if self.pause:
                return False
            if self.full_size_img and self.enable_profile:
                self.pr.enable()

            # First check if target img idx is key frame
            target_img_idx = self.current_img_idx.copy()

            if self.dim_1_dir == 1:
                if target_img_idx[1] == self.mtd_shape[1] - 1:
                    self.dim_0_dir = 1
                else:
                    target_img_idx[1] += 1
            elif self.dim_1_dir == -1:
                if target_img_idx[1] == 0:
                    self.dim_0_dir = 1
                else:
                    target_img_idx[1] -= 1

            if self.dim_0_dir == 1:
                target_img_idx[0] = (target_img_idx[0] + 1) % self.mtd_shape[0]

            keyframe = self.get_key_frame(self.mtd, tuple(target_img_idx))

            if keyframe is not None:
                self.current_frame = keyframe
            # Use difference to generate current frame
            else:
                # first move in dimension 1
                if self.dim_1_dir == 1:
                    if not self.current_img_idx[1] == self.mtd_shape[1] - 1:
                        next_img_idx = self.current_img_idx.copy()
                        next_img_idx[1] += 1
                        keyframe = self.get_key_frame(self.mtd, tuple(next_img_idx))
                        if keyframe is not None:
                            self.current_frame = keyframe
                        else:
                            diff_image = self.get_diff_image(
                                mtd=self.mtd,
                                dim0=self.current_img_idx[0],
                                dim1=self.current_img_idx[1],
                                dir=1,
                            )
                            self.current_frame = (
                                np.clip(
                                    self.current_frame.astype(np.int16) + diff_image,
                                    0,
                                    255,
                                )
                            ).astype(np.uint8)
                        self.current_img_idx = next_img_idx

                elif self.dim_1_dir == -1:
                    if not self.current_img_idx[1] == 0:
                        next_img_idx = self.current_img_idx.copy()
                        next_img_idx[1] -= 1
                        keyframe = self.get_key_frame(self.mtd, tuple(next_img_idx))
                        if keyframe is not None:
                            self.current_frame = keyframe
                        else:
                            diff_image = self.get_diff_image(
                                mtd=self.mtd,
                                dim0=self.current_img_idx[0],
                                dim1=self.current_img_idx[1] - 1,
                                dir=1,
                            )
                            self.current_frame = (
                                np.clip(
                                    self.current_frame.astype(np.int16) - diff_image,
                                    0,
                                    255,
                                )
                            ).astype(np.uint8)
                        self.current_img_idx = next_img_idx
                if self.dim_0_dir == 1:
                    # move in dimension 0
                    diff_image = self.get_diff_image(
                        mtd=self.mtd,
                        dim0=self.current_img_idx[0],
                        dim1=self.current_img_idx[1],
                        dir=0,
                    )
                    self.current_frame = (
                        np.clip(
                            self.current_frame.astype(np.int16) + diff_image, 0, 255
                        )
                    ).astype(np.uint8)

            self.current_img_idx = target_img_idx

            orig_img = Image.fromarray(self.current_frame)
            if self.brightness != 1:
                enhancer = ImageEnhance.Brightness(orig_img)
                orig_img = enhancer.enhance(self.brightness)

            img = orig_img.resize(self.image_size, Image.NEAREST)
            self.photoImg = ImageTk.PhotoImage(img)

            self.canvas.itemconfig(self.img_container, image=self.photoImg)

            current_t = time.time()
            elapsed = current_t - self._previous_t
            sleep_time = max(1 / self.max_fps - elapsed - self._estimated_image_time, 0)
            time.sleep(sleep_time)

            t_start = time.time()
            self.tk.update_idletasks()
            self._estimated_image_time = time.time() - t_start
            self._previous_t = time.time()

            self.next = False
            if self.full_size_img and self.enable_profile:
                self.pr.disable()
            return True
        else:
            return False


class SoundMonitor:
    def __init__(self, gui: GUI):
        self.gui = gui
        self.last_n = []
        self.callback_count = 0
        self.now = None

    def callback(self, indata, frames, ctime, status):
        amplitude = np.mean(abs(indata))
        n_average = np.mean(self.last_n)
        dim_0_dir = 0
        if amplitude < (n_average * (self.gui.threshold - 0.01)) or amplitude < 0.02:
            dim_1_dir = 1
        elif amplitude > n_average * (self.gui.threshold + 0.01):
            dim_1_dir = -1
        else:
            dim_1_dir = 0

        if 5 * amplitude > random.random() * self.gui.threshold:
            dim_0_dir = 1

        self.gui.set_next_image(dim_1_dir=dim_1_dir, dim_0_dir=dim_0_dir)
        self.last_n.append(amplitude)
        self.last_n = self.last_n[-100:]

        self.callback_count += 1
        if self.callback_count % 100 == 0:
            if self.now is not None:
                elapsed = time.time() - self.now
                print(f"sound callback per second {100 / elapsed}")
            self.now = time.time()

    def run(self):
        self.stream = sd.InputStream(
            samplerate=16000, channels=2, callback=self.callback, blocksize=300
        )
        self.stream.start()

    def stop(self):
        self.stream.stop()


if __name__ == "__main__":
    max_fps = 30
    file_dir = os.path.dirname(__file__)
    video_dir = os.path.join(file_dir, "../../mtd_videos")
    gui = GUI(video_dir=video_dir)
    sm = SoundMonitor(gui)
    sm.run()

    count = 0

    previous_10_t = None
    while True:

        updated = gui.update()
        if updated:
            count += 1

            if count % 10 == 0:
                gui.tk.update()
                count = 0
                if previous_10_t is None:
                    previous_10_t = time.time()
                else:
                    current_10_t = time.time()
                    elapse_sum = current_10_t - previous_10_t
                    previous_10_t = current_10_t
                    print(f"frame per second {10 / elapse_sum}")

        else:
            time.sleep(0.001)
