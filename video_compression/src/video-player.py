import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                           QPushButton, QVBoxLayout, QHBoxLayout, QSlider,
                           QMenuBar, QMenu, QAction, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import pygame
import os

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimedia Player")
        
        # 视频相关变量
        self.video_path = None
        self.rgb_reader = None
        self.frame_count = 0
        self.current_frame = 0
        self.fps = 30
        self.playing = False
        self.video_size = (640, 480)  # 默认视频尺寸
        
        # 音频相关变量
        self.audio_path = None
        pygame.mixer.init()
        
        # 定时器用于视频播放
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 创建菜单栏和工具栏
        self.create_menu_bar()
        
        # 初始化UI
        self.init_ui()
        
        # 设置初始状态
        self.update_ui_state(False)
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('File')
        
        # 打开视频文件动作
        open_video_action = QAction('Open Video', self)
        open_video_action.setShortcut('Ctrl+O')
        open_video_action.triggered.connect(self.open_video_file)
        
        # 打开音频文件动作
        open_audio_action = QAction('Open Audio', self)
        open_audio_action.setShortcut('Ctrl+A')
        open_audio_action.triggered.connect(self.open_audio_file)
        
        # 分辨率选择菜单
        resolution_menu = menubar.addMenu('Resolution')
        
        # 添加分辨率选项
        resolutions = [
            ('540p', 960, 540),
            ('1080p', 1920, 1080)
        ]
        
        for name, width, height in resolutions:
            action = QAction(name, self)
            action.setCheckable(True)
            if (width, height) == self.video_size:
                action.setChecked(True)
            action.triggered.connect(lambda checked, w=width, h=height: self.change_resolution(w, h))
            resolution_menu.addAction(action)
        
        # 退出动作
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        # 添加动作到菜单
        file_menu.addAction(open_video_action)
        file_menu.addAction(open_audio_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu('View')
        
        # 重置视图动作
        reset_view_action = QAction('Reset View', self)
        reset_view_action.setShortcut('Ctrl+R')
        reset_view_action.triggered.connect(self.reset_view)
        
        view_menu.addAction(reset_view_action)
        
    def init_ui(self):
        """初始化UI组件"""
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)  # 设置最小尺寸
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        layout.addWidget(self.video_label)
        
        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.sliderPressed.connect(self.slider_pressed)
        self.slider.sliderReleased.connect(self.slider_released)
        layout.addWidget(self.slider)
        
        # 时间显示
        self.time_label = QLabel('00:00 / 00:00')
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)
        
        # 控制按钮
        controls_layout = QHBoxLayout()
        
        self.play_button = QPushButton('Play')
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.play_pause)
        
        self.stop_button = QPushButton('Stop')
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        
        self.prev_button = QPushButton('Previous Frame')
        self.prev_button.setEnabled(False)
        self.prev_button.clicked.connect(self.previous_frame)
        
        self.next_button = QPushButton('Next Frame')
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.next_frame)
        
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.prev_button)
        controls_layout.addWidget(self.next_button)
        
        layout.addLayout(controls_layout)
        
        # 状态标签
        self.status_label = QLabel('No video loaded')
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 设置窗口大小
        self.setMinimumSize(800, 600)
    
    def update_ui_state(self, has_video):
        """更新UI组件状态"""
        self.play_button.setEnabled(has_video)
        self.stop_button.setEnabled(has_video)
        self.prev_button.setEnabled(has_video)
        self.next_button.setEnabled(has_video)
        self.slider.setEnabled(has_video)
    
    def change_resolution(self, width, height):
        """改变视频分辨率"""
        self.video_size = (width, height)
        # 更新视频标签的最小尺寸
        self.video_label.setMinimumSize(width//2, height//2)  # 设置为分辨率的一半，使界面不会太大
        
        # 如果已加载视频，重新加载
        if self.video_path:
            self.load_video(self.video_path)
    
    def detect_video_size(self, file_size):
        """检测视频尺寸"""
        # 可能的分辨率列表
        possible_sizes = [
            (960, 540),    # 540p
            (1920, 1080),  # 1080p
        ]
        
        possible_matches = []
        for width, height in possible_sizes:
            frame_size = width * height * 3  # RGB每像素3字节
            if file_size % frame_size == 0:
                frames = file_size // frame_size
                possible_matches.append((width, height, frames))
        
        if possible_matches:
            # 让用户选择正确的尺寸
            msg = QMessageBox()
            msg.setWindowTitle("Select Video Resolution")
            msg.setText("Please select the correct video resolution:")
            
            size_str = "\n".join([f"{w}x{h} ({f} frames)" for w, h, f in possible_matches])
            msg.setInformativeText(size_str)
            
            buttons = []
            for w, h, _ in possible_matches:
                buttons.append(msg.addButton(f"{w}x{h}", QMessageBox.ActionRole))
            
            msg.exec_()
            
            clicked = msg.clickedButton()
            if clicked in buttons:
                index = buttons.index(clicked)
                return possible_matches[index][0], possible_matches[index][1]
        
        return None, None
    
    def open_video_file(self):
        """打开视频文件对话框"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "RGB Files (*.rgb);;All Files (*.*)"
        )
        
        if file_name:
            try:
                file_size = os.path.getsize(file_name)
                print(f"File size: {file_size} bytes")
                
                # 尝试检测视频尺寸
                width, height = self.detect_video_size(file_size)
                
                if width and height:
                    self.video_size = (width, height)
                    self.load_video(file_name)
                else:
                    QMessageBox.warning(
                        self,
                        "Invalid Video File",
                        "The video file size does not match any supported resolution.\n"
                        "Supported resolutions:\n"
                        "- 540p (960x540)\n"
                        "- 1080p (1920x1080)"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load video file: {str(e)}"
                )
    
    def open_audio_file(self):
        """打开音频文件对话框"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "WAV Files (*.wav);;All Files (*.*)"
        )
        
        if file_name:
            try:
                pygame.mixer.music.load(file_name)
                self.audio_path = file_name
                self.status_label.setText(f'Audio loaded: {os.path.basename(file_name)}')
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load audio file: {str(e)}"
                )
    
    def load_video(self, video_path):
        """加载视频文件"""
        self.video_path = video_path
        
        # 创建RGB视频读取器
        self.rgb_reader = RGBVideoReader(width=self.video_size[0], height=self.video_size[1])
        if self.rgb_reader.open(video_path):
            self.frame_count = self.rgb_reader.frame_count
            
            # 设置滑块范围
            self.slider.setRange(0, self.frame_count - 1)
            
            # 显示第一帧
            self.current_frame = 0
            self.display_frame()
            
            # 更新状态
            self.status_label.setText(f'Video loaded: {os.path.basename(video_path)}')
            self.update_time_label()
            
            # 更新UI状态
            self.update_ui_state(True)
        else:
            self.status_label.setText('Error loading video')
            self.update_ui_state(False)
    
    def update_time_label(self):
        """更新时间显示"""
        current_time = self.current_frame / self.fps
        total_time = self.frame_count / self.fps
        
        current_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
        total_str = f"{int(total_time//60):02d}:{int(total_time%60):02d}"
        
        self.time_label.setText(f"{current_str} / {total_str}")
    
    def reset_view(self):
        """重置视图大小"""
        if hasattr(self, 'rgb_reader'):
            self.video_label.setMinimumSize(self.video_size[0], self.video_size[1])
            self.display_frame()
    def play_pause(self):
        """播放/暂停切换"""
        if not self.rgb_reader:  # 改用rgb_reader替代cap
            return
            
        if self.playing:
            self.timer.stop()
            if self.audio_path:
                pygame.mixer.music.pause()
            self.play_button.setText('Play')
        else:
            self.timer.start(int(1000/self.fps))  # 设置定时器间隔
            if self.audio_path:
                pygame.mixer.music.unpause()
            self.play_button.setText('Pause')
            
        self.playing = not self.playing
    
    def stop(self):
        """停止播放"""
        if not self.rgb_reader:  # 改用rgb_reader替代cap
            return
            
        self.playing = False
        self.timer.stop()
        if self.audio_path:
            pygame.mixer.music.stop()
        self.play_button.setText('Play')
        self.current_frame = 0
        self.slider.setValue(0)
        self.display_frame()
        self.update_time_label()
    
    def update_frame(self):
        """更新视频帧"""
        if self.current_frame >= self.frame_count - 1:
            self.stop()
            return
            
        self.current_frame += 1
        self.slider.setValue(self.current_frame)
        self.display_frame()
        self.update_time_label()
    
    def display_frame(self):
        """显示当前帧"""
        if not self.rgb_reader:
            return
            
        ret, frame = self.rgb_reader.read_frame(self.current_frame)
        if ret:
            h, w, ch = frame.shape
            
            # 调整图像大小以适应窗口
            scale = min(self.video_label.width() / w, self.video_label.height() / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
            # 转换为QImage并显示
            bytes_per_line = ch * new_w
            qt_image = QImage(frame.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def previous_frame(self):
        """显示前一帧"""
        if not self.rgb_reader or self.current_frame <= 0:
            return
            
        self.current_frame -= 1
        self.slider.setValue(self.current_frame)
        self.display_frame()
        self.update_time_label()
    
    def next_frame(self):
        """显示下一帧"""
        if not self.rgb_reader or self.current_frame >= self.frame_count - 1:
            return
            
        self.current_frame += 1
        self.slider.setValue(self.current_frame)
        self.display_frame()
        self.update_time_label()
    
    def slider_pressed(self):
        """滑块被按下时暂停更新"""
        if self.playing:
            self.timer.stop()
    
    def slider_released(self):
        """滑块释放时恢复播放"""
        if self.playing:
            self.timer.start(int(1000/self.fps))
    
    def slider_moved(self, position):
        """响应滑块移动"""
        self.current_frame = position
        self.display_frame()
        self.update_time_label()
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.rgb_reader:
            self.rgb_reader.release()
        pygame.mixer.quit()
        event.accept()

class RGBVideoReader:
    def __init__(self, width, height, frame_rate=30):
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.file = None
        self.frame_size = width * height * 3  # 3 channels (RGB)
        
    def open(self, filename):
        """打开RGB视频文件"""
        try:
            self.file = open(filename, 'rb')
            # 获取文件大小来计算总帧数
            self.file.seek(0, 2)  # 移动到文件末尾
            file_size = self.file.tell()
            self.frame_count = file_size // self.frame_size
            self.file.seek(0)  # 回到文件开始
            return True
        except Exception as e:
            print(f"Error opening file: {e}")
            return False
            
    def read_frame(self, frame_number):
        """读取指定帧"""
        if not self.file or frame_number >= self.frame_count:
            return False, None
            
        # 移动到指定帧的位置
        self.file.seek(frame_number * self.frame_size)
        
        # 读取一帧的数据
        frame_data = self.file.read(self.frame_size)
        if len(frame_data) != self.frame_size:
            return False, None
            
        # 转换为numpy数组
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))
        
        return True, frame
        
    def release(self):
        """关闭文件"""
        if self.file:
            self.file.close()
            self.file = None
    
    def play_pause(self):
        """播放/暂停切换"""
        if not self.cap:
            return
            
        if self.playing:
            self.timer.stop()
            pygame.mixer.music.pause()
            self.play_button.setText('Play')
        else:
            self.timer.start(int(1000/self.fps))  # 设置定时器间隔
            if self.audio_path:
                pygame.mixer.music.unpause()
            self.play_button.setText('Pause')
            
        self.playing = not self.playing
    
    def stop(self):
        """停止播放"""
        if not self.cap:
            return
            
        self.playing = False
        self.timer.stop()
        if self.audio_path:
            pygame.mixer.music.stop()
        self.play_button.setText('Play')
        self.current_frame = 0
        self.slider.setValue(0)
        self.display_frame()
    
    def update_frame(self):
        """更新视频帧"""
        if self.current_frame >= self.frame_count - 1:
            self.stop()
            return
            
        self.current_frame += 1
        self.slider.setValue(self.current_frame)
        self.display_frame()
    
    def display_frame(self):
        """显示当前帧"""
        if not self.cap:
            return
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if ret:
            # 转换颜色空间
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            
            # 调整图像大小以适应窗口
            scale = min(self.video_label.width() / w, self.video_label.height() / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            
            # 转换为QImage并显示
            bytes_per_line = ch * new_w
            qt_image = QImage(frame.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def previous_frame(self):
        """显示前一帧"""
        if not self.cap or self.current_frame <= 0:
            return
            
        self.current_frame -= 1
        self.slider.setValue(self.current_frame)
        self.display_frame()
    
    def next_frame(self):
        """显示下一帧"""
        if not self.cap or self.current_frame >= self.frame_count - 1:
            return
            
        self.current_frame += 1
        self.slider.setValue(self.current_frame)
        self.display_frame()
    
    def slider_moved(self, position):
        """响应滑块移动"""
        self.current_frame = position
        self.display_frame()
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.cap:
            self.cap.release()
        pygame.mixer.quit()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建并显示播放器
    player = VideoPlayer()
    player.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()



