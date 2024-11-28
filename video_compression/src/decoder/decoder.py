import cv2
import numpy as np
import tkinter as tk
import pygame
import threading
import time
from scipy.fftpack import idct
from PIL import Image, ImageTk
import os
import wave


class CMPDecoder:
    def __init__(self, cmp_file, width=960, height=540, n_channels=3):
        self.cmp_file = cmp_file
        self.width = width
        self.height = height
        self.n_channels = n_channels
        self.mb_size = 16
        self.mb_rows = height // self.mb_size
        self.mb_cols = width // self.mb_size
        self.frames = []
        self.n1, self.n2 = None, None
        self.total_frames = 0  # Total number of frames

    def read_header(self, file):
        # Read the quantization parameters
        header = file.readline().decode().strip()
        self.n1, self.n2 = map(int, header.split())
        print(f"[Decoder] Quantization parameters: n1={self.n1}, n2={self.n2}")

    def calculate_total_frames(self):
        # Calculate total frames based on file size
        file_size = os.path.getsize(self.cmp_file)
        header_size = 0
        with open(self.cmp_file, 'rb') as file:
            header = file.readline()
            header_size = len(header)

        # Each macroblock consists of 1 byte (block type) and 1536 bytes (block data)
        macroblock_size = 1 + 1536  # bytes
        total_macroblocks = self.mb_rows * self.mb_cols
        frame_size = macroblock_size * total_macroblocks  # bytes per frame

        # Total frames = (file size - header size) // frame size
        self.total_frames = (file_size - header_size) // frame_size
        print(f"[Decoder] Total number of frames: {self.total_frames}")

    def decode_macroblock(self, coeffs, quantization_step):
        n = quantization_step
        block = np.zeros((self.mb_size, self.mb_size, self.n_channels), dtype=np.uint8)
        index = 0
        qt_base = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        qt = qt_base * (2 ** (n - 4))
        for i in range(0, self.mb_size, 8):
            for j in range(0, self.mb_size, 8):
                for c in range(self.n_channels):
                    quant_block = np.array(coeffs[index:index + 64]).reshape((8, 8))
                    index += 64
                    # Dequantize
                    dct_block = quant_block * qt
                    # Perform inverse DCT
                    block_8x8 = idct(idct(dct_block, axis=1, norm='ortho'), axis=0, norm='ortho')
                    # Clip and convert to uint8
                    block[i:i + 8, j:j + 8, c] = np.clip(block_8x8, 0, 255).astype(np.uint8)
        return block

    def read_frame(self, file):
        try:
            frame = np.zeros((self.height, self.width, self.n_channels), dtype=np.uint8)
            for row in range(self.mb_rows):
                for col in range(self.mb_cols):
                    # Read block type (1 byte)
                    block_type_data = file.read(1)
                    if not block_type_data:
                        # End of file
                        raise EOFError("Unexpected end of file while reading block type")
                    block_type = int.from_bytes(block_type_data, byteorder='big')
                    is_foreground = block_type == 1

                    quantization_step = self.n1 if is_foreground else self.n2

                    # Read macroblock data (1536 bytes)
                    coeffs_data = file.read(1536)
                    if len(coeffs_data) < 1536:
                        # Incomplete macroblock data
                        raise EOFError("Unexpected end of file while reading macroblock data")

                    # Convert bytes to coefficients
                    coeffs = []
                    for i in range(0, 1536, 2):
                        coeff = int.from_bytes(coeffs_data[i:i+2], byteorder='big', signed=True)
                        coeffs.append(coeff)

                    block = self.decode_macroblock(coeffs, quantization_step)

                    y = row * self.mb_size
                    x = col * self.mb_size
                    frame[y:y + self.mb_size, x:x + self.mb_size] = block
            return frame
        except Exception as e:
            print(f"[Decoder] Error while reading frame: {e}")
            raise

    def decode(self):
        with open(self.cmp_file, 'rb') as file:
            self.read_header(file)
            self.calculate_total_frames()  # Calculate total frames at the beginning
            frame_count = 0
            while True:
                try:
                    print(f"[Decoder] Decoding frame {frame_count + 1}/{self.total_frames}")
                    frame = self.read_frame(file)
                    self.frames.append(frame)
                    frame_count += 1
                except EOFError:
                    print(f"[Decoder] Incomplete frame at {frame_count + 1}. Skipping.")
                    break
                except Exception as e:
                    print(f"[Decoder] Error at frame {frame_count + 1}: {e}.")
                    break


class VideoPlayer:
    def __init__(self, frames, audio_path, fps=30):
        self.frames = frames
        self.audio_path = audio_path
        self.fps = fps
        self.playing = False
        self.current_frame = 0
        self.width = 960    # Fixed width
        self.height = 540   # Fixed height
        self.start_time = None

        # Initialize audio using pygame
        pygame.mixer.init(frequency=44100)
        pygame.mixer.music.load(audio_path)

        self.audio_duration = self.get_audio_duration(audio_path)
        self.video_duration = len(self.frames) / self.fps
        print(f"Audio Duration: {self.audio_duration:.2f} seconds")
        print(f"Video Duration: {self.video_duration:.2f} seconds")

        # Initialize the Tkinter GUI window
        self.root = tk.Tk()
        self.root.title("Video Player")
        self.root.geometry(f"{self.width}x{self.height+100}")  # Set fixed window size

        # Video display area
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Display the first frame or a black image to set the initial size
        if self.frames:
            first_frame = self.frames[0]
        else:
            first_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        photo = tk.PhotoImage(data=cv2.imencode(".png", frame_rgb)[1].tobytes())
        self.video_label.config(image=photo)
        self.video_label.image = photo

        # Control panel frame
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(fill=tk.X)

        # Play/pause button
        self.play_button = tk.Button(
            self.control_frame, text="Play", command=self.toggle_playback
        )
        self.play_button.pack(side=tk.LEFT)

        # Stop button
        self.stop_button = tk.Button(
            self.control_frame, text="Stop", command=self.stop_playback
        )
        self.stop_button.pack(side=tk.LEFT)

        # Next frame button
        self.next_button = tk.Button(
            self.control_frame, text="Next", command=self.next_frame
        )
        self.next_button.pack(side=tk.LEFT)

        # Frame number display
        self.frame_label = tk.Label(self.control_frame, text="Frame: 0")
        self.frame_label.pack(side=tk.RIGHT)

        # Start the GUI event loop
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.mainloop()

    def get_audio_duration(self, audio_path):
        with wave.open(audio_path, 'rb') as audio_file:
            frames = audio_file.getnframes()
            rate = audio_file.getframerate()
            duration = frames / float(rate)
            return duration
    def toggle_playback(self):
        if self.playing:

            self.playing = False
            pygame.mixer.music.pause()
            self.play_button.config(text="Play")
            print("[Player] Playback paused.")
        else:

            self.playing = True
            if not pygame.mixer.music.get_busy():

                pygame.mixer.music.play()
            else:

                pygame.mixer.music.unpause()

            threading.Thread(target=self.play_video).start()
            self.play_button.config(text="Pause")
            print("[Player] Playback started.")


    def stop_playback(self):
        self.playing = False
        pygame.mixer.music.stop()  # Stop audio playback
        self.current_frame = 0    # Reset to the first frame
        self.update_frame()
        self.play_button.config(text="Play")
        print("[Player] Playback stopped.")

    def play_video(self):
        frame_duration = 1 / self.fps
        next_frame_time = time.perf_counter()
        while self.playing and self.current_frame < len(self.frames):
            # 显示帧
            self.update_frame()
            print(f"[Player] Playing frame {self.current_frame + 1}")
            self.current_frame += 1

            # 调度下一帧
            next_frame_time += frame_duration
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.perf_counter()

        if self.current_frame >= len(self.frames):
            print("[Player] Video playback finished.")
            self.playing = False
            pygame.mixer.music.stop()

    def next_frame(self):
        if self.current_frame + 1 < len(self.frames):
            self.current_frame += 1
            self.update_frame()
            print(f"[Player] Jumped to frame {self.current_frame + 1}")
        else:
            print("[Player] No more frames to show.")

    def update_frame(self):
        if self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]

            frame_resized = cv2.resize(frame, (self.width, self.height))

            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_resized)
            # img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=photo)
            self.video_label.image = photo
            self.frame_label.config(text=f"Frame: {self.current_frame + 1}")
        else:
            print("[Player] No frame to display.")


    def close(self):
        self.playing = False
        pygame.mixer.quit()
        self.root.destroy()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python decoder.py input_video.cmp input_audio.wav")
        sys.exit(1)

    cmp_file = sys.argv[1]
    audio_file = sys.argv[2]

    decoder = CMPDecoder(cmp_file)
    decoder.decode()

    player = VideoPlayer(decoder.frames, audio_file)
