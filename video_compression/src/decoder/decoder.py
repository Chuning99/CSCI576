import cv2
import numpy as np
import tkinter as tk
import pygame
import threading
import time


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
                        # If macroblock data is missing, assume end of file and stop processing
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
        self.width = 960    # Fixed width
        self.height = 540   # Fixed height
        self.start_time = None

        # Initialize audio using pygame
        pygame.mixer.init(frequency=44100)
        pygame.mixer.music.load(audio_path)

        # Initialize the Tkinter GUI window
        self.root = tk.Tk()
        self.root.title("Video Player")

        # Video display area
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

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

    def toggle_playback(self):
        if self.playing:
            # Pause playback
            self.playing = False
            pygame.mixer.music.pause()  # Pause audio
            self.play_button.config(text="Play")
            print("[Player] Playback paused.")
        else:
            # Resume playback
            self.playing = True
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
                self.start_time = time.time() - self.current_frame / self.fps
            else:
                pygame.mixer.music.unpause()
                self.start_time = time.time() - self.current_frame / self.fps
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
        while self.playing and self.current_frame < len(self.frames):
            # Calculate expected frame display time
            elapsed_time = time.time() - self.start_time
            expected_frame = int(elapsed_time * self.fps)

            # Display the frame if it's time
            if self.current_frame <= expected_frame:
                self.update_frame()
                print(f"[Player] Playing frame {self.current_frame + 1}")
                self.current_frame += 1

            # Sleep for a small duration to avoid busy-waiting
            time.sleep(0.001)

        # Stop playback when video ends
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
            # Resize frame to match the fixed window size
            frame_resized = cv2.resize(frame, (self.width, self.height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Convert the frame to a format Tkinter can use
            photo = tk.PhotoImage(data=cv2.imencode(".png", frame_rgb)[1].tobytes())
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
