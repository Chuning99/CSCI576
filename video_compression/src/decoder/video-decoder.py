import cv2
import numpy as np
from scipy.fftpack import idct
from pathlib import Path
import pygame
import wave
import pyaudio
import argparse
from typing import List, Tuple, Optional
import time
import sys
from tqdm import tqdm

class VideoDecoder:
    def __init__(self, width: int = 960, height: int = 540):
        """Initialize decoder with video dimensions."""
        self.width = width
        self.height = height
        self.mb_rows = self.height // 16
        self.mb_cols = self.width // 16
        self.n1 = None
        self.n2 = None
        print(f"Initialized decoder: {width}x{height}")
        print(f"Macroblocks: {self.mb_cols}x{self.mb_rows}")

    def decode_frame(self, lines: List[str]) -> np.ndarray:
        """Decode a single frame from CMP data."""
        # Initialize frame buffer
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        total_blocks = len(lines)
        expected_blocks = self.mb_rows * self.mb_cols
        
        print(f"Processing frame: {total_blocks} blocks, expected {expected_blocks}")
        
        # Process each macroblock
        for i in range(min(total_blocks, expected_blocks)):
            # Calculate block position
            row = i // self.mb_cols     # Row number
            col = i % self.mb_cols      # Column number
            
            y = row * 16  # Y coordinate in pixels
            x = col * 16  # X coordinate in pixels
            
            # Parse block data
            try:
                line = lines[i].strip()
                parts = line.split()
                is_foreground = int(parts[0]) == 1
                coeffs = list(map(float, parts[1:]))
                
                # Process block
                block = self.process_block(coeffs, is_foreground)
                
                # Place block in frame at correct position
                frame[y:y+16, x:x+16] = block
                
            except Exception as e:
                print(f"Error processing block at ({x},{y}): {str(e)}")
                continue
        
        return frame

    def process_block(self, coeffs: List[float], is_foreground: bool) -> np.ndarray:
        """Process a 16x16 macroblock."""
        block = np.zeros((16, 16, 3), dtype=np.float32)
        n = self.n1 if is_foreground else self.n2
        
        # Process each 8x8 sub-block
        coeff_idx = 0
        for i in range(0, 16, 8):
            for j in range(0, 16, 8):
                for c in range(3):  # RGB channels
                    # Extract coefficients for current 8x8 block
                    if coeff_idx + 64 <= len(coeffs):
                        # Get 8x8 block coefficients
                        block_coeffs = np.array(coeffs[coeff_idx:coeff_idx+64], dtype=np.float32)
                        block_coeffs = block_coeffs.reshape(8, 8)
                        
                        # Dequantize
                        dequant_block = self.dequantize(block_coeffs, n)
                        
                        # Apply IDCT
                        pixel_block = self.apply_idct(dequant_block)
                        
                        # Place in correct position of 16x16 block
                        block[i:i+8, j:j+8, c] = pixel_block
                        
                        coeff_idx += 64
        
        # Clip values to valid range
        return np.clip(block, 0, 255).astype(np.uint8)

    def dequantize(self, block: np.ndarray, n: float) -> np.ndarray:
        """Dequantize block using 2^n as step size."""
        step = 2.0 ** n
        return block * step

    def apply_idct(self, block: np.ndarray) -> np.ndarray:
        """Apply 2D IDCT to 8x8 block."""
        return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

    # def process_block(self, coeffs: List[float], is_foreground: bool) -> np.ndarray:
    #     """Process a 16x16 block."""
    #     block = np.zeros((16, 16, 3))
    #     n = self.n1 if is_foreground else self.n2
        
    #     coeff_idx = 0
    #     for i in range(0, 16, 8):
    #         for j in range(0, 16, 8):
    #             for c in range(3):
    #                 block_coeffs = np.array(coeffs[coeff_idx:coeff_idx+64]).reshape(8, 8)
    #                 dequant_block = self.dequantize(block_coeffs, n)
    #                 pixel_block = self.apply_idct(dequant_block)
    #                 pixel_block = np.clip(pixel_block, 0, 255)
    #                 block[i:i+8, j:j+8, c] = pixel_block
    #                 coeff_idx += 64
        
    #     return block.astype(np.uint8)

    def read_cmp_header(self, file) -> Tuple[float, float]:
        """Read quantization parameters from CMP file header."""
        try:
            # Read first line and ensure it only contains two numbers
            header = file.readline().decode().strip()
            values = header.split()
            
            # Debug print
            print(f"Header line: '{header}'")
            print(f"Split values: {values}")
            
            if len(values) != 2:
                raise ValueError(f"Expected 2 values in header, got {len(values)}: {values}")
                
            self.n1 = float(values[0])
            self.n2 = float(values[1])
            
            print(f"Successfully parsed n1={self.n1}, n2={self.n2}")
            return self.n1, self.n2
            
        except Exception as e:
            print(f"Error parsing header: {str(e)}")
            raise

    # def decode_frame(self, lines: List[str]) -> np.ndarray:
    #     """Decode a single frame from CMP data."""
    #     frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
    #     # Calculate number of blocks in width and height
    #     blocks_width = self.width // 16
    #     blocks_height = self.height // 16
    #     print(f"Frame dimensions in blocks: {blocks_width}x{blocks_height}")
        
    #     for block_idx, line in enumerate(lines):
    #         if block_idx >= blocks_width * blocks_height:
    #             break
                
    #         # Calculate block position
    #         block_y = (block_idx // blocks_width) * 16
    #         block_x = (block_idx % blocks_width) * 16
            
    #         try:
    #             # Parse block data
    #             parts = line.strip().split()
    #             is_foreground = int(parts[0]) == 1
    #             coeffs = list(map(float, parts[1:]))
                
    #             # Process block
    #             block = self.process_block(coeffs, is_foreground)
                
    #             # Place block in frame
    #             frame[block_y:block_y+16, block_x:block_x+16] = block
                
    #         except Exception as e:
    #             print(f"Error processing block at ({block_x}, {block_y}): {e}")
    #             continue
        
    #     return frame

class VideoPlayer:
    def __init__(self, cmp_path: str, wav_path: str):
        """Initialize video player with compressed video and audio files."""
        print("Initializing video player...")
        pygame.init()
        pygame.mixer.init(frequency=44100)
        
        self.decoder = VideoDecoder()
        self.frames = []
        self.current_frame = 0
        self.playing = False
        self.fps = 30
        self.last_update_time = time.time()
        self.frame_delay = 1.0 / self.fps
        
        # Initialize font first
        self.font = pygame.font.Font(None, 24)
        
        # Create loading screen
        self.screen = pygame.display.set_mode((self.decoder.width, self.decoder.height + 50))
        pygame.display.set_caption("Video Player")
        
        # Create buttons
        button_y = self.decoder.height + 10
        self.buttons = {
            'play': pygame.Rect(10, button_y, 80, 30),
            'pause': pygame.Rect(100, button_y, 80, 30),
            'stop': pygame.Rect(190, button_y, 80, 30),
            'next': pygame.Rect(280, button_y, 80, 30)
        }
        
        # Load content
        self.load_content(cmp_path, wav_path)
        
        print("Initialization complete")

    def show_loading_screen(self, progress, total):
        """Display loading progress."""
        self.screen.fill((0, 0, 0))
        
        # Calculate progress percentage
        percentage = (progress / total) * 100 if total > 0 else 0
        
        # Draw progress bar
        bar_width = self.decoder.width - 100
        bar_height = 30
        bar_x = 50
        bar_y = self.decoder.height // 2
        
        # Background bar
        pygame.draw.rect(self.screen, (100, 100, 100), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Progress bar
        progress_width = int((percentage / 100) * bar_width)
        pygame.draw.rect(self.screen, (0, 255, 0), 
                        (bar_x, bar_y, progress_width, bar_height))
        
        # Loading text
        text = self.font.render(f"Loading: {percentage:.1f}%", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.decoder.width//2, bar_y - 30))
        self.screen.blit(text, text_rect)
        
        pygame.display.flip()

    def load_content(self, cmp_path: str, wav_path: str):
        """Load video and audio content with progress display."""
        print("\nLoading video content...")
        try:
            with open(cmp_path, 'rb') as f:
                # Read header
                self.decoder.read_cmp_header(f)
                
                # Calculate blocks per frame
                blocks_per_frame = (self.decoder.width // 16) * (self.decoder.height // 16)
                print(f"Blocks per frame: {blocks_per_frame}")
                
                # Read all lines
                lines = f.readlines()[1:]  # Skip header line
                
                # Process lines into frames
                frame_data = []
                total_blocks = len(lines)
                expected_frames = total_blocks // blocks_per_frame
                print(f"Total blocks: {total_blocks}")
                print(f"Expected frames: {expected_frames}")
                
                # Split into frames
                for i in range(0, total_blocks, blocks_per_frame):
                    frame_lines = [line.decode().strip() for line in lines[i:i + blocks_per_frame]]
                    if len(frame_lines) == blocks_per_frame:  # Only add complete frames
                        frame_data.append(frame_lines)
                
                total_frames = len(frame_data)
                print(f"\nFound {total_frames} complete frames")
                
                # Process frames
                print("\nDecoding frames...")
                for i, frame_lines in enumerate(frame_data):
                    try:
                        # Verify frame data
                        if len(frame_lines) != blocks_per_frame:
                            print(f"Warning: Frame {i+1} has {len(frame_lines)} blocks, expected {blocks_per_frame}")
                            continue
                            
                        # Decode frame
                        frame = self.decoder.decode_frame(frame_lines)
                        
                        # Convert to pygame surface
                        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
                        self.frames.append(surface)
                        
                        # Update progress
                        if i % 10 == 0:
                            print(f"Decoded frame {i+1}/{total_frames}")
                            self.show_loading_screen(i + 1, total_frames)
                        
                    except Exception as e:
                        print(f"Error decoding frame {i+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                print(f"\nSuccessfully decoded {len(self.frames)} frames")
                
                # Load audio
                if Path(wav_path).exists():
                    print("Loading audio...")
                    pygame.mixer.music.load(wav_path)
                    print("Audio loading complete")
                else:
                    print("No audio file found")
                
                # Show first frame
                if self.frames:
                    self.screen.blit(self.frames[0], (0, 0))
                    self.draw_controls()
                    pygame.display.flip()
                
        except Exception as e:
            print(f"Error loading content: {e}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def update_frame(self):
        """Update the current frame based on timing."""
        current_time = time.time()
        if current_time - self.last_update_time >= self.frame_delay:
            self.last_update_time = current_time
            
            if self.current_frame < len(self.frames):
                # Clear the screen first
                self.screen.fill((0, 0, 0))  # Fill with black
                
                # Draw frame
                self.screen.blit(self.frames[self.current_frame], (0, 0))
                
                # Draw controls
                self.draw_controls()
                
                # Update display
                pygame.display.flip()
                
                # Move to next frame
                self.current_frame += 1
                print(f"Playing frame {self.current_frame}/{len(self.frames)}")
                
                # Stop at end
                if self.current_frame >= len(self.frames):
                    self.playing = False
                    pygame.mixer.music.stop()
                    print("Playback complete")
                
                return True
            
            return False

    def handle_click(self, pos):
        """Handle mouse clicks on buttons."""
        for action, rect in self.buttons.items():
            if rect.collidepoint(pos):
                if action == 'play':
                    print(f"Play clicked (current frame: {self.current_frame}/{len(self.frames)})")
                    if not self.playing:
                        self.playing = True
                        if self.current_frame >= len(self.frames):
                            self.current_frame = 0
                        # Clear screen before starting playback
                        self.screen.fill((0, 0, 0))
                        pygame.mixer.music.play()
                        self.last_update_time = time.time()
                
                elif action == 'pause':
                    print(f"Pause clicked (current frame: {self.current_frame}/{len(self.frames)})")
                    self.playing = False
                    pygame.mixer.music.pause()
                
                elif action == 'stop':
                    print(f"Stop clicked (current frame: {self.current_frame}/{len(self.frames)})")
                    self.playing = False
                    self.current_frame = 0
                    pygame.mixer.music.stop()
                    if len(self.frames) > 0:
                        # Clear screen before showing first frame
                        self.screen.fill((0, 0, 0))
                        self.screen.blit(self.frames[0], (0, 0))
                        self.draw_controls()
                        pygame.display.flip()
                
                elif action == 'next':
                    print(f"Next clicked (current frame: {self.current_frame}/{len(self.frames)})")
                    self.playing = False
                    pygame.mixer.music.stop()
                    if self.current_frame < len(self.frames):
                        # Clear screen before showing next frame
                        self.screen.fill((0, 0, 0))
                        self.screen.blit(self.frames[self.current_frame], (0, 0))
                        self.current_frame += 1
                        self.draw_controls()
                        pygame.display.flip()

    def run(self):
        """Main player loop."""
        print("\nStarting player...")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    self.handle_click(pos)
            
            if self.playing:
                self.update_frame()
            
            # Cap the frame rate
            pygame.time.wait(10)  # Small delay to prevent excessive CPU usage
    
    pygame.quit()
    def draw_controls(self):
        """Draw control buttons and frame counter."""
        # Draw button background
        pygame.draw.rect(self.screen, (0, 0, 0), 
                        (0, self.decoder.height, self.decoder.width, 50))
        
        # Draw buttons
        for text, rect in self.buttons.items():
            pygame.draw.rect(self.screen, (200, 200, 200), rect)
            text_surface = self.font.render(text.title(), True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
        
        # Draw frame counter
        counter_text = f"Frame: {self.current_frame + 1}/{len(self.frames)}"
        text_surface = self.font.render(counter_text, True, (255, 255, 255))
        self.screen.blit(text_surface, (370, self.decoder.height + 15))

    # def run(self):
    #     """Main player loop."""
    #     print("\nStarting player...")
    #     running = True
    #     while running:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False
    #             elif event.type == pygame.MOUSEBUTTONDOWN:
    #                 pos = pygame.mouse.get_pos()
    #                 self.handle_click(pos)
            
    #         if self.playing:
    #             self.update_frame()
            
    #         pygame.time.wait(10)  # Small delay to prevent excessive CPU usage
        
    #     pygame.quit()

    

def main():
    parser = argparse.ArgumentParser(description='Video Decoder and Player')
    parser.add_argument('input_cmp', help='Input CMP file')
    parser.add_argument('input_wav', help='Input WAV file')
    
    args = parser.parse_args()
    
    player = VideoPlayer(args.input_cmp, args.input_wav)
    player.run()

if __name__ == "__main__":
    main()