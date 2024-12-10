import cv2
import numpy as np
import tkinter as tk
import pygame
import threading
import time
import wave
from PIL import Image, ImageTk


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

    def read_header(self, file):
        # Read the quantization parameters
        header = file.readline().decode().strip()
        self.n1, self.n2 = map(int, header.split())
        print(f"[Decoder] Quantization parameters: n1={self.n1}, n2={self.n2}")

    def decode_macroblock(self, coeffs, is_foreground, quantization_step):
        # Perform inverse quantization
        step = 2 ** quantization_step
        dequantized = np.array(coeffs) * step
        # Reshape into 8x8 blocks and perform IDCT
        block = np.zeros((self.mb_size, self.mb_size, self.n_channels), dtype=np.uint8)
        index = 0
        for i in range(0, self.mb_size, 8):
            for j in range(0, self.mb_size, 8):
                for c in range(self.n_channels):
                    dct_block = dequantized[index:index + 64].reshape((8, 8))
                    idct_block = cv2.idct(dct_block).clip(0, 255).astype(np.uint8)
                    block[i:i + 8, j:j + 8, c] = idct_block
                    index += 64
        return block

    def read_frame(self, file):
        try:
            frame = np.zeros((self.height, self.width, self.n_channels), dtype=np.uint8)
            for row in range(self.mb_rows):
                for col in range(self.mb_cols):
                    block_info = file.readline().decode().strip().split()
                    if not block_info:
                        raise EOFError("Unexpected end of file while reading macroblock data")
                    is_foreground = int(block_info[0]) == 1
                    coeffs = list(map(float, block_info[1:]))
                    quantization_step = self.n1 if is_foreground else self.n2
                    block = self.decode_macroblock(coeffs, is_foreground, quantization_step)
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
            frame_count = 0
            while True:
                try:
                    print(f"[Decoder] Decoding frame {frame_count + 1}")
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
        self.width = 960
        self.height = 540
        self.start_time = None
        self.audio_playback_time = 0.0

        pygame.mixer.init(frequency=44100)
        pygame.mixer.music.load(audio_path)

        self.audio_duration = self.get_audio_duration(audio_path)
        self.video_duration = len(self.frames) / self.fps
        print(f"Audio Duration: {self.audio_duration:.2f} seconds")
        print(f"Video Duration: {self.video_duration:.2f} seconds")

        self.root = tk.Tk()
        self.root.title("Video Player")
        self.root.geometry(f"{self.width}x{self.height + 100}")

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        first_frame = self.frames[0] if self.frames else np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(first_frame)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=photo)
        self.video_label.image = photo

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(fill=tk.X)

        self.play_button = tk.Button(
            self.control_frame, text="Play", command=self.toggle_playback
        )
        self.play_button.pack(side=tk.LEFT)

        self.stop_button = tk.Button(
            self.control_frame, text="Stop", command=self.stop_playback
        )
        self.stop_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(
            self.control_frame, text="Next", command=self.next_frame
        )
        self.next_button.pack(side=tk.LEFT)

        self.frame_label = tk.Label(self.control_frame, text="Frame: 0")
        self.frame_label.pack(side=tk.RIGHT)

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
            self.audio_playback_time = pygame.mixer.music.get_pos() / 1000.0
            print(self.audio_playback_time)
            pygame.mixer.music.pause()
            self.play_button.config(text="Play")
            print("[Player] Playback paused.")
        else:
            self.playing = True
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play(start=self.audio_playback_time)
                # pygame.mixer.music.set_pos(self.audio_playback_time * 1000.0)
                self.start_time = time.perf_counter() - self.current_frame / self.fps
                print("not busy")
            else:
                pygame.mixer.music.play(start=self.audio_playback_time)
                # pygame.mixer.music.set_pos(self.audio_playback_time)
                self.start_time = time.perf_counter() - self.current_frame / self.fps
                print(self.audio_playback_time)
                print("busy")
            # threading.Thread(target=self.play_video).start()
            threading.Thread(target=self.play_video).start()
            self.play_button.config(text="Pause")
            print("[Player] Playback started.")

    def stop_playback(self):
        self.playing = False
        pygame.mixer.music.stop()
        self.current_frame = 0
        self.audio_playback_time = 0.0
        self.update_frame()
        self.play_button.config(text="Play")
        print("[Player] Playback stopped.")

    def play_video(self):
        frame_duration = 1 / self.fps
        next_frame_time = time.perf_counter()
        while self.playing and self.current_frame < len(self.frames):
            elapsed_time = time.perf_counter() - self.start_time
            expected_frame = int(elapsed_time * self.fps)

            if self.current_frame <= expected_frame:
                self.update_frame()
                self.current_frame += 1

            # audio_time = pygame.mixer.music.get_pos() / 1000.0
            # video_time = self.current_frame / self.fps
            # if abs(audio_time - video_time) > 0.1:
            #     print(f"[Sync] Adjusting audio to match video: audio={audio_time:.2f}, video={video_time:.2f}")
            #     pygame.mixer.music.set_pos(video_time)

            next_frame_time += frame_duration
            sleep_time = next_frame_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self.current_frame >= len(self.frames):
            print("[Player] Video playback finished.")
            self.playing = False
            pygame.mixer.music.stop()

    def next_frame(self):
        if self.current_frame + 1 < len(self.frames):
            self.current_frame += 1
            self.update_frame()
            self.audio_playback_time = self.current_frame / self.fps
            pygame.mixer.music.set_pos(self.audio_playback_time)
        else:
            print("[Player] No more frames to show.")

    def update_frame(self):
        if self.current_frame < len(self.frames):
            frame = self.frames[self.current_frame]
            frame_resized = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            # frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=photo)
            self.video_label.image = photo
            self.frame_label.config(text=f"Frame: {self.current_frame + 1}")

    def close(self):
        self.playing = False
        pygame.mixer.quit()
        self.root.destroy()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python player.py input_video.cmp input_audio.wav")
        sys.exit(1)

    cmp_file = sys.argv[1]
    audio_file = sys.argv[2]

    decoder = CMPDecoder(cmp_file)
    decoder.decode()

    player = VideoPlayer(decoder.frames, audio_file)
