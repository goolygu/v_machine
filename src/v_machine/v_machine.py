import cProfile
import glob
import gzip
import io
import os
import pickle
import pstats
import queue
import random
import sys
import time
from functools import lru_cache
from multiprocessing import Process, Queue
from pstats import SortKey
import numpy as np
import sounddevice as sd
from PIL import Image, ImageEnhance
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.Qt import Qt
from PyQt5 import QtGui


def get_key_frame_array(key_frame_buffer):
    return np.asarray(Image.open(key_frame_buffer), dtype=np.uint8)


class GUI(QWidget):
    def __init__(
        self,
        video_dir,
        enable_profile=False,
        max_fps=30,
    ):
        super().__init__()
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
        self.setStyleSheet("background-color: black;")
        self.setWindowTitle("Visual Loop Machine by Li Yang Ku")
        self.setup_mtd_video(mtd)

        self.image_size = (512, 512)
        self.resize(550, 610)

        self.canvas = QLabel(self)
        self.canvas.resize(*self.image_size)
        self.default_img_loc = [19, 19]
        self.canvas.move(*self.default_img_loc)

        self.instruction_loc = [50, 550]
        self.instruction = QLabel(self)
        self.instruction.setAlignment(Qt.AlignCenter)
        self.instruction.setText(
            "→ next video, ← last video, ↑ increase change, ↓ decrease change.\nSpace: toggle fullscreen, Esc: exit fullscreen"
        )
        self.instruction.setStyleSheet("color: white;")
        self.instruction.move(*self.instruction_loc)
        self.instruction.show()

        orig_img = Image.fromarray(self.current_frame)
        img = orig_img.resize(self.image_size, Image.HAMMING)
        qim = QtGui.QImage(
            img.tobytes("raw", "RGB"), img.width, img.height, QtGui.QImage.Format_RGB888
        )
        self.canvas.setPixmap(QtGui.QPixmap.fromImage(qim))

        self.fullscreen_state = False
        self.next = False
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
        self.previous_canvas_size = None

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key_Space:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key_Right:
            self.right()
        elif event.key() == Qt.Key_Left:
            self.left()
        elif event.key() == Qt.Key_Up:
            self.up()
        elif event.key() == Qt.Key_Down:
            self.down()
        elif event.key() == Qt.Key_Escape:
            self.end_fullscreen()

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
            print(f"no diff for {dim0}, {dim1}, {dir}")
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
        return mtd

    def down(self):
        self.threshold += 0.1
        self.threshold = min(1.1, self.threshold)
        print(f"threshold {self.threshold}")

    def up(self):
        self.threshold -= 0.1
        self.threshold = max(0.5, self.threshold)
        print(f"threshold {self.threshold}")

    def right(self):
        if not self.load_previous:
            self.load_next = True

    def left(self):
        if not self.load_next:
            self.load_previous = True

    def toggle_fullscreen(self):
        if self.fullscreen_state is True:
            self.end_fullscreen()
        else:
            self.start_full_screen()

    def start_full_screen(self):
        self.instruction.hide()
        self.fullscreen_state = True
        self.showFullScreen()
        screen = QApplication.primaryScreen()
        full_width = screen.size().width()
        full_height = screen.size().height()
        size = min([full_width, full_height])
        self.image_size = (size, size)
        self.canvas.resize(*self.image_size)
        self.canvas.move((full_width - size) / 2, (full_height - size) / 2)
        return

    def end_fullscreen(self):
        self.image_size = (512, 512)
        self.fullscreen_state = False
        self.showNormal()
        self.canvas.resize(*self.image_size)

        self.canvas.move(*self.default_img_loc)
        self.instruction.show()

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

    def update(self, dim_1_dir=0, dim_0_dir=0):
        if self.load_next:
            self.load_next_video()
        elif self.load_previous:
            self.load_next_video(previous=True)
        if self.pause:
            return False
        if self.fullscreen_state and self.enable_profile:
            self.pr.enable()

        # First check if target img idx is key frame
        target_img_idx = self.current_img_idx.copy()

        if dim_1_dir == 1:
            if target_img_idx[1] == self.mtd_shape[1] - 1:
                dim_0_dir = 1
            else:
                target_img_idx[1] += 1
        elif dim_1_dir == -1:
            if target_img_idx[1] == 0:
                dim_0_dir = 1
            else:
                target_img_idx[1] -= 1

        if dim_0_dir == 1:
            target_img_idx[0] = (target_img_idx[0] + 1) % self.mtd_shape[0]

        keyframe = self.get_key_frame(self.mtd, tuple(target_img_idx))

        if keyframe is not None:
            self.current_frame = keyframe
        # Use difference to generate current frame
        else:
            # first move in dimension 1
            if dim_1_dir == 1:
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

            elif dim_1_dir == -1:
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
            if dim_0_dir == 1:
                # move in dimension 0
                diff_image = self.get_diff_image(
                    mtd=self.mtd,
                    dim0=self.current_img_idx[0],
                    dim1=self.current_img_idx[1],
                    dir=0,
                )
                self.current_frame = (
                    np.clip(self.current_frame.astype(np.int16) + diff_image, 0, 255)
                ).astype(np.uint8)

        self.current_img_idx = target_img_idx

        orig_img = Image.fromarray(self.current_frame)
        if self.brightness != 1:
            enhancer = ImageEnhance.Brightness(orig_img)
            orig_img = enhancer.enhance(self.brightness)

        img = orig_img.resize(self.image_size, Image.NEAREST)

        qim = QtGui.QImage(
            img.tobytes("raw", "RGB"), img.width, img.height, QtGui.QImage.Format_RGB888
        )

        current_t = time.time()
        elapsed = current_t - self._previous_t
        sleep_time = max(1 / self.max_fps - elapsed - self._estimated_image_time, 0)
        time.sleep(sleep_time)

        t_start = time.time()
        self.canvas.setPixmap(QtGui.QPixmap.fromImage(qim))
        self._estimated_image_time = time.time() - t_start
        self._previous_t = time.time()

        self.next = False
        if self.fullscreen_state and self.enable_profile:
            self.pr.disable()


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

        self.gui.update(dim_1_dir=dim_1_dir, dim_0_dir=dim_0_dir)
        self.last_n.append(amplitude)
        self.last_n = self.last_n[-200:]

        self.callback_count += 1
        if self.callback_count % 100 == 0:
            if self.now is not None:
                elapsed = time.time() - self.now
                print(f"frame per second {100 / elapsed}")
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
    app = QApplication(sys.argv)
    icon_dir = os.path.join(file_dir, "../../v_machine_icon.gif")
    app.setWindowIcon(QtGui.QIcon(icon_dir))
    gui = GUI(video_dir=video_dir)
    sm = SoundMonitor(gui)
    sm.run()
    gui.show()
    app.exec_()
