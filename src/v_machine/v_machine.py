import cProfile
import glob
import gzip
import io
import os
import pickle
import pstats
import random
import sys
import time
import threading
from functools import lru_cache
import numpy as np
import sounddevice as sd
from PIL import Image, ImageEnhance
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QMessageBox
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6 import QtGui
from PyQt6.QtGui import QGuiApplication
from type import mtd_video

sys.modules["mtd_video"] = mtd_video


def get_key_frame_array(key_frame_buffer):
    return np.asarray(Image.open(key_frame_buffer), dtype=np.uint8)


def load_video(path, idx, mtd_dict, lru_list, loading_indices, max_num_mtd=4) -> mtd_video.MTDVideo:
    fp = gzip.open(path, "rb")
    print(f"loading file {path}")
    mtd = pickle.load(fp)
    print(f"finish loading file")
    fp.close()
    mtd_dict[idx] = mtd
    if len(mtd_dict) > max_num_mtd:
        assert lru_list[0] != idx
        mtd_dict.pop(lru_list[0])
    loading_indices.remove(idx)


def pil_to_pixmap(im):
    im = im.convert("RGBA")
    data = im.tobytes("raw", "RGBA")
    qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format.Format_RGBA8888)
    pixmap = QtGui.QPixmap.fromImage(qim)
    return pixmap


class VideoNotReady(Exception):
    pass


class GUI(QWidget):

    signal = pyqtSignal(int, int)

    def __init__(
        self,
        video_dir,
        enable_profile=False,
        max_fps=30,
    ):
        super().__init__()
        self.loading_indices = []
        self.mtd_idx_lru_list = []
        self.mtd_dict = {}
        self.loading_stage = 0
        self.video_dir = video_dir
        self.all_video_paths = sorted(glob.glob(f"{video_dir}/*.mtd"))
        if len(self.all_video_paths) == 0:
            print(f"No MTD videos at {video_dir}")
            return
        self.current_vid_idx = 0
        print(f"loading video from {video_dir}")
        self.pre_load_next_video(self.current_vid_idx)

        print("finish loading")
        self.mtd = None
        self.current_img_idx = None
        self.current_frame = None
        self.mtd_shape = None
        self.setStyleSheet("background-color: black;")
        self.setWindowTitle("Visual Loop Machine")
        mtd = None
        while mtd is None:
            try:
                mtd = self.get_mtd_video(self.current_vid_idx)
            except VideoNotReady:
                time.sleep(0.5)

        self.setup_mtd_video(mtd)
        self.pre_load_next_video(self.current_vid_idx + 1)

        self.image_size = (512, 512)
        self.resize(550, 660)

        self.canvas = QLabel(self)
        self.canvas.resize(*self.image_size)
        self.default_img_loc = [19, 19]
        self.canvas.move(*self.default_img_loc)

        self.instruction_loc = [50, 545]
        self.instruction = QLabel(self)

        self.instruction.setText(
            "→ next video, ← last video, ↑ increase change, ↓ decrease change\nSpace: toggle fullscreen, Esc: exit fullscreen\n\nInput source:\n\nFullscreen setting:"
        )
        self.instruction.setStyleSheet("color: white;")
        self.instruction.move(*self.instruction_loc)
        self.instruction.show()

        self.combobox_loc = [150, 585]
        self.combobox = QComboBox(self)
        self.combobox.move(*self.combobox_loc)
        self.combobox.resize(350, 30)
        self.combobox.setStyleSheet(
            "color: #75F5CA; selection-color: white; selection-background-color: #4DCDA2"
        )
        self.combobox.keyPressEvent = self.keyPressEvent
        self.combobox.show()

        self.combobox_2_loc = [200, 618]
        self.combobox_2 = QComboBox(self)
        self.combobox_2.move(*self.combobox_2_loc)
        self.combobox_2.resize(150, 30)
        self.combobox_2.setStyleSheet(
            "color: #75F5CA; selection-color: white; selection-background-color: #4DCDA2"
        )
        self.combobox_2.addItem("default")
        self.combobox_2.addItem("mirror")
        self.combobox_2.addItem("rescale")
        self.combobox_2.keyPressEvent = self.keyPressEvent
        self.combobox_2.show()

        orig_img = Image.fromarray(self.current_frame)

        img = orig_img.resize(self.image_size, Image.Resampling.NEAREST)

        self.canvas.setPixmap(pil_to_pixmap(img))

        self.fullscreen_state = False
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
        self.previous_canvas_size = None

        self.sound_monitor = None
        self.cidx_mapping = {}
        self._update_count = 0
        self._now = None
        self.mirror = False


    def set_sound_monitor(self, sound_monitor):
        self.sound_monitor = sound_monitor
        self.cidx_mapping = {}
        for i, device in enumerate(self.sound_monitor.devices):
            if device["max_input_channels"] > 0:
                self.cidx_mapping[self.combobox.count()] = i
                self.combobox.addItem(device["name"])

        default_idx = self.combobox.count() - 1
        self.combobox.setCurrentIndex(default_idx)
        self.sound_monitor.current_device_id = self.cidx_mapping[default_idx]
        self.combobox.currentIndexChanged.connect(self.select_sound_device)
        self.combobox.keyPressEvent = self.keyPressEvent

    def select_sound_device(self, index):
        cindex = self.combobox.currentIndex()
        device_id = self.cidx_mapping[cindex]
        if device_id != self.sound_monitor.current_device_id:
            print(f"switch to device {self.sound_monitor.devices[device_id]['name']}")
            self.sound_monitor.close()
            self.sound_monitor.run(device_id)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Right:
            self.right()
        elif event.key() == Qt.Key.Key_Left:
            self.left()
        elif event.key() == Qt.Key.Key_Up:
            self.up()
        elif event.key() == Qt.Key.Key_Down:
            self.down()
        elif event.key() == Qt.Key.Key_Escape:
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
        self.combobox.hide()
        self.combobox_2.hide()
        self.fullscreen_state = True
        self.showFullScreen()

        screen = QGuiApplication.screenAt(self.mapToGlobal(QPoint(0, 0)))

        full_width = screen.size().width()
        full_height = screen.size().height()
        size = min([full_width, full_height])
        if self.combobox_2.currentIndex() == 1:
            self.mirror = True
            self.image_size = (full_width, full_height)
            self.canvas.resize(*self.image_size)
        elif self.combobox_2.currentIndex() == 2:
            self.image_size = (full_width, full_height)
            self.canvas.resize(*self.image_size)
        elif self.combobox_2.currentIndex() == 0:
            self.image_size = (size, size)
            self.canvas.resize(*self.image_size)
            self.canvas.move(int((full_width - size) / 2), int((full_height - size) / 2))
        return

    def end_fullscreen(self):
        self.image_size = (512, 512)
        self.fullscreen_state = False
        self.mirror = False
        self.showNormal()
        self.canvas.resize(*self.image_size)

        self.canvas.move(*self.default_img_loc)
        self.instruction.show()
        self.combobox.show()
        self.combobox_2.show()

        if self.enable_profile:
            s = io.StringIO()
            ps = pstats.Stats(self.pr, stream=s).sort_stats("cumulative")
            ps.print_stats()
            print(s.getvalue())
        return

    def pre_load_next_video(self, next_vid_idx):
        if next_vid_idx in self.loading_indices:
            return

        if next_vid_idx in self.mtd_dict:
            return

        next_vid_idx = next_vid_idx % len(self.all_video_paths)
        print("loading video")

        self.loading_indices.append(next_vid_idx)

        new_lru_list = []
        for idx in self.mtd_idx_lru_list[::-1]:
            if idx in self.mtd_dict and idx not in new_lru_list:
                new_lru_list.append(idx)
        for idx in self.mtd_dict.keys():
            if idx not in new_lru_list:
                new_lru_list.append(idx)

        self.mtd_idx_lru_list = new_lru_list[::-1]

        x = threading.Thread(target=load_video, args=(self.all_video_paths[next_vid_idx], next_vid_idx, self.mtd_dict, self.mtd_idx_lru_list, self.loading_indices))
        x.start()

    def get_mtd_video(self, idx):

        if idx in self.mtd_dict:
            self.mtd_idx_lru_list.append(idx)
            return self.mtd_dict[idx]

        self.pre_load_next_video(idx)
        raise VideoNotReady

    def load_next_video(self, previous=False, smooth_transition=True, transition_speed=0.2):
        if self.loading_stage == 0:
            self.loading_stage = 1
            if previous:
                self.current_vid_idx -= 1
            else:
                self.current_vid_idx += 1

            self.current_vid_idx = self.current_vid_idx % len(self.all_video_paths)

        elif self.loading_stage == 1:
            if smooth_transition:
                self.brightness = max(self.brightness - transition_speed, 0)
            if self.brightness == 0 or not smooth_transition:
                try:
                    mtd = self.get_mtd_video(self.current_vid_idx)
                    self.setup_mtd_video(mtd)
                    self.loading_stage = 2
                except VideoNotReady:
                    pass

        elif self.loading_stage == 2:
            if smooth_transition:
                self.brightness = min(self.brightness + transition_speed, 1)
            if self.brightness == 1 or not smooth_transition:
                self.load_next = False
                self.load_previous = False
                self.loading_stage = 0
                if not previous:
                    self.pre_load_next_video((self.current_vid_idx + 1) % len(self.all_video_paths))
                else:
                    self.pre_load_next_video((self.current_vid_idx - 1) % len(self.all_video_paths))

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
        if self.mirror:
            ratio = self.image_size[0] / self.image_size[1]
            print(ratio)
            if ratio == 1:
                orig_img = Image.fromarray(self.current_frame)
            elif ratio > 0:
                frame_size = self.current_frame.shape
                new_frame_size = [frame_size[0], int(frame_size[1] * ratio), frame_size[2]]
                margin = (new_frame_size[1] - frame_size[1]) // 2
                expand_frame = np.zeros(new_frame_size, dtype=self.current_frame.dtype)
                expand_frame[:, margin:margin+frame_size[1]] = self.current_frame
                expand_frame[:, :margin] = self.current_frame[:, :margin][:, ::-1]
                expand_frame[:, margin+frame_size[1]:margin*2+frame_size[1]] = self.current_frame[:, -margin:][:, ::-1]
                orig_img = Image.fromarray(expand_frame)
            else:
                frame_size = self.current_frame.shape
                new_frame_size = [int(frame_size[0] / ratio), frame_size[1], frame_size[2]]
                margin = (new_frame_size[0] - frame_size[0]) // 2
                expand_frame = np.zeros(new_frame_size, dtype=self.current_frame.dtype)
                expand_frame[margin:margin+frame_size[0]] = self.current_frame
                expand_frame[:margin] = self.current_frame[:margin][::-1]
                expand_frame[margin+frame_size[0]:margin*2+frame_size[0]] = self.current_frame[-margin:][::-1]
                orig_img = Image.fromarray(expand_frame)
        else:
            orig_img = Image.fromarray(self.current_frame)
        if self.brightness != 1:
            enhancer = ImageEnhance.Brightness(orig_img)
            orig_img = enhancer.enhance(self.brightness)

        img = orig_img.resize(self.image_size, Image.Resampling.NEAREST)

        self.canvas.setPixmap(pil_to_pixmap(img))

        self._update_count += 1

        if self._update_count % 100 == 0:
            if self._now is not None:
                elapsed = time.time() - self._now
                print(f"frame per second {100 / elapsed}")
            self._now = time.time()
            self._update_count = 0

        if self.fullscreen_state and self.enable_profile:
            self.pr.disable()


class SoundMonitor:
    def __init__(self, gui: GUI, max_fps: int):
        self.gui = gui
        self.signal = gui.signal
        self.last_n = [0]
        self.now = None
        self.devices = sd.query_devices()
        self.current_device_id = None
        self.max_fps = max_fps

    def callback(self, indata, frames, ctime, status):
        amplitude = np.mean(abs(indata))
        n_average = np.mean(self.last_n)
        dim_0_dir = 0
        if amplitude < (n_average * (self.gui.threshold - 0.01)) or amplitude < 0.001:
            dim_1_dir = 1
        elif amplitude > n_average * (self.gui.threshold + 0.01):
            dim_1_dir = -1
        else:
            dim_1_dir = 0

        if 10 * amplitude > random.random() * self.gui.threshold:
            dim_0_dir = 1

        self.signal.emit(dim_1_dir, dim_0_dir)
        self.last_n.append(amplitude)
        self.last_n = self.last_n[-200:]

    def run(self, device_id):
        self.current_device_id = device_id
        device = self.devices[self.current_device_id]
        samplerate = device["default_samplerate"]
        channels = device["max_input_channels"]
        blocksize = int(samplerate // self.max_fps)
        self.stream = sd.InputStream(
            samplerate=samplerate,
            channels=channels,
            callback=self.callback,
            blocksize=blocksize,
            device=device["name"],
            latency="low"
        )
        self.stream.start()

    def close(self):
        self.stream.close()


def get_video_directroy(file_dir: str, app_store: bool = False):
    if app_store:
        user_name = os.getenv('USER')
        video_dir = "/Users/" + user_name + "/Movies/mtd_videos"
    else:
        default_video_dir = os.path.join(file_dir, "../../mtd_videos")
        if os.path.isdir(default_video_dir):
            return default_video_dir
        else:
            home_dir = os.path.expanduser("~")
            if sys.platform == "linux" or sys.platform == "linux2":
                video_dir = os.path.join(home_dir, "Videos/mtd_videos/")
            elif sys.platform == "darwin":
                video_dir = os.path.join(home_dir, "Movies/mtd_videos/")
            elif sys.platform == "win32":
                video_dir = os.path.join(home_dir, "Videos/mtd_videos/")

        if not os.path.isdir(video_dir):
            os.makedirs(video_dir)
    num_videos = 0
    if os.path.isdir(video_dir):
        num_videos = len(glob.glob(f"{video_dir}/*.mtd"))
    if num_videos == 0:
        msg = QMessageBox()
        msg.setText(
            f"No MTD videos in directory {video_dir}. MTD Videos can be downloaded from "
            f'<a href="https://visualloopmachine.liyangku.com/download-mtd-videos">'
            f" here.</a> Loading low resolution demo videos instead."
        )
        msg.exec()
        video_dir = os.path.join(sys._MEIPASS, "mtd_videos")

    return video_dir


def get_icon_directory(file_dir: str):
    default_icon_dir = os.path.join(file_dir, "../../v_machine_icon.gif")
    if os.path.exists(default_icon_dir):
        return default_icon_dir
    icon_dir = os.path.join(sys._MEIPASS, "files/v_machine_icon.gif")
    return icon_dir


if __name__ == "__main__":
    max_fps = 30
    file_dir = os.path.dirname(__file__)
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(get_icon_directory(file_dir)))
    video_dir = get_video_directroy(file_dir)
    gui = GUI(video_dir=video_dir, max_fps=max_fps)
    sm = SoundMonitor(gui, max_fps=max_fps)
    sm.signal.connect(gui.update)
    gui.set_sound_monitor(sm)
    sm.run(sm.current_device_id)
    gui.show()
    app.exec()
    sm.close()
