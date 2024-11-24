import cv2
import numpy as np
from scipy.fftpack import dct, idct
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import BinaryIO, Optional, Tuple, List, Dict, Set
import os
from dataclasses import dataclass
import logging
from multiprocessing import Pool, cpu_count
import os
@dataclass
class MotionVector:
    """Represents a motion vector with its position and direction."""
    x: int  # Block x position
    y: int  # Block y position
    dx: int  # Motion in x direction
    dy: int  # Motion in y direction
    mad: float  # Mean absolute difference

class VideoEncoder:
    def __init__(self, input_path: str):
        """Initialize video encoder with automatic resolution detection."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Detect video dimensions
        self._detect_dimensions(input_path)
        
        # Initialize parameters
        self.prev_frame = None
        self.mb_rows = self.height // 16
        self.mb_cols = self.width // 16
        
        # Motion detection parameters
        self.motion_threshold = 20.0
        self.search_range = 16

        self.use_multiprocessing = True
        self.num_processes = os.cpu_count()
        
        self.logger.info(f"Initialized encoder: {self.width}x{self.height}, "
                        f"{self.mb_rows}x{self.mb_cols} macroblocks")

    def _detect_dimensions(self, input_path: str):
        """Detect video dimensions from file size."""
        file_size = os.path.getsize(input_path)
        
        resolutions = {
            (960, 540): 960 * 540 * 3,
            (1920, 1080): 1920 * 1080 * 3
        }
        
        for (w, h), frame_bytes in resolutions.items():
            if file_size % frame_bytes == 0:
                self.width = w
                self.height = h
                self.frame_size = frame_bytes
                return
                
        raise ValueError("Invalid video dimensions. Expected 960x540 or 1920x1080")

    def validate_parameters(self, n1: int, n2: int):
        """Validate quantization parameters."""
        if not isinstance(n1, int) or not isinstance(n2, int):
            raise ValueError("Quantization parameters must be integers")
        if n1 < 0 or n2 < 0:
            raise ValueError("Quantization parameters must be non-negative")
        if n1 > 8 or n2 > 8:
            self.logger.warning("High quantization parameters may result in poor quality")

    def read_frame(self, file: BinaryIO) -> Optional[np.ndarray]:
        """Read a single RGB frame from file."""
        try:
            raw_data = file.read(self.frame_size)
            if len(raw_data) < self.frame_size:
                return None
            
            frame = np.frombuffer(raw_data, dtype=np.uint8)
            return frame.reshape((self.height, self.width, 3))
        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            return None

    def get_block(self, frame: np.ndarray, y: int, x: int, size: int = 16) -> np.ndarray:
        """Extract a block from frame with proper boundary handling."""
        # Create padded block
        block = np.zeros((size, size, frame.shape[2]), dtype=frame.dtype)
        
        # Calculate valid regions
        y_start = max(0, y)
        y_end = min(frame.shape[0], y + size)
        x_start = max(0, x)
        x_end = min(frame.shape[1], x + size)
        
        # Calculate target positions
        target_y = max(0, -y)
        target_x = max(0, -x)
        
        # Copy valid region
        if y_end > y_start and x_end > x_start:
            h = y_end - y_start
            w = x_end - x_start
            block[target_y:target_y+h, target_x:target_x+w] = frame[y_start:y_end, x_start:x_end]
        
        return block

    def calculate_motion_vector(self, curr_block: np.ndarray, prev_frame: np.ndarray, 
                            block_y: int, block_x: int) -> MotionVector:
        """Optimized motion vector calculation with larger step size."""
        if prev_frame is None:
            return MotionVector(block_x, block_y, 0, 0, float('inf'))
        
        # Convert to grayscale for motion estimation
        curr_gray = cv2.cvtColor(curr_block, cv2.COLOR_RGB2GRAY)
        
        min_mad = float('inf')
        best_dx = best_dy = 0
        
        # Reduce search range and increase step size
        self.search_range = 8  # Reduced from 16
        step_size = 4   # Increased from 2
        
        y_min = max(-self.search_range, -block_y)
        y_max = min(self.search_range, self.height - block_y - 16)
        x_min = max(-self.search_range, -block_x)
        x_max = min(self.search_range, self.width - block_x - 16)
        
        for dy in range(y_min, y_max + 1, step_size):
            for dx in range(x_min, x_max + 1, step_size):
                ref_y = block_y + dy
                ref_x = block_x + dx
                
                if 0 <= ref_y < self.height-15 and 0 <= ref_x < self.width-15:
                    ref_block = prev_frame[ref_y:ref_y+16, ref_x:ref_x+16]
                    ref_gray = cv2.cvtColor(ref_block, cv2.COLOR_RGB2GRAY)
                    mad = np.sum(np.abs(curr_gray - ref_gray)) / 256  # 使用sum代替mean加快速度
                    if mad < min_mad:
                        min_mad = mad
                        best_dx, best_dy = dx, dy
        
        return MotionVector(block_x, block_y, best_dx, best_dy, min_mad)

# 在class VideoEncoder中添加以下方法：

    def _estimate_global_motion(self, vectors: List[MotionVector]) -> Tuple[float, float]:
        """估计全局运动（摄像机运动）。"""
        motions = [(mv.dx, mv.dy) for mv in vectors]
        if not motions:
            return (0.0, 0.0)
            
        # 使用中值以减少异常值影响
        median_dx = np.median([dx for dx, _ in motions])
        median_dy = np.median([dy for _, dy in motions])
        return median_dx, median_dy

    def _is_similar_motion(self, mv1: MotionVector, mv2: MotionVector, 
                        angle_threshold: float = 30.0, 
                        magnitude_ratio_threshold: float = 0.7) -> bool:
        """判断两个运动向量是否相似。"""
        # 计算运动大小
        mag1 = np.sqrt(mv1.dx**2 + mv1.dy**2)
        mag2 = np.sqrt(mv2.dx**2 + mv2.dy**2)
        
        # 如果两个都是很小的运动，认为是相似的
        if mag1 < 0.5 and mag2 < 0.5:
            return True
            
        # 如果一个运动很小而另一个不是，认为不相似
        if (mag1 < 0.5) != (mag2 < 0.5):
            return False
        
        # 检查运动大小比例
        if max(mag1, mag2) > 0:
            magnitude_ratio = min(mag1, mag2) / max(mag1, mag2)
            if magnitude_ratio < magnitude_ratio_threshold:
                return False
        
        # 检查运动方向
        if mag1 > 0 and mag2 > 0:
            dot_product = mv1.dx * mv2.dx + mv1.dy * mv2.dy
            cos_angle = dot_product / (mag1 * mag2)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180.0 / np.pi
            return angle <= angle_threshold
        
        return False

    def _is_background(self, mv: MotionVector, global_motion: Tuple[float, float]) -> bool:
        """判断一个块是否属于背景。"""
        motion_magnitude = np.sqrt(mv.dx**2 + mv.dy**2)
        global_mag = np.sqrt(global_motion[0]**2 + global_motion[1]**2)
        
        # 情况1：几乎没有运动
        if motion_magnitude < 0.5 and global_mag < 0.5:
            return True
        
        # 情况2：运动与全局运动一致（摄像机运动）
        if global_mag > 0:
            dot_product = mv.dx * global_motion[0] + mv.dy * global_motion[1]
            cos_angle = dot_product / (motion_magnitude * global_mag)
            magnitude_ratio = min(motion_magnitude, global_mag) / max(motion_magnitude, global_mag)
            return cos_angle > 0.95 and magnitude_ratio > 0.7
        
        return False

    def _find_contiguous_region(self, start_idx: int, vectors: List[MotionVector], 
                            start_row: int, start_col: int) -> Set[int]:
        """Optimized contiguous region search."""
        region = {start_idx}
        queue = [(start_row, start_col)]
        start_mv = vectors[start_idx]
        
        # Only check 4-neighborhood instead of 8
        neighbors = [(-1,0), (0,-1), (0,1), (1,0)]
        
        while queue and len(region) < 20:  # Limit region size for speed
            row, col = queue.pop(0)
            
            for dr, dc in neighbors:
                new_row, new_col = row + dr, col + dc
                
                if not (0 <= new_row < self.mb_rows and 0 <= new_col < self.mb_cols):
                    continue
                
                idx = new_row * self.mb_cols + new_col
                if idx in region or idx >= len(vectors):
                    continue
                
                if self._is_similar_motion(vectors[idx], start_mv):
                    region.add(idx)
                    queue.append((new_row, new_col))
        
        return region

    def group_motion_vectors(self, vectors: List[MotionVector]) -> Dict[int, bool]:
        """改进的运动向量分组方法。"""
        block_is_foreground = {}
        processed = set()
        
        # 估计全局运动
        global_motion = self._estimate_global_motion(vectors)
        print(f"Estimated global motion: dx={global_motion[0]:.2f}, dy={global_motion[1]:.2f}")
        
        # 处理每个块
        for i, mv in enumerate(vectors):
            if i in processed:
                continue
            
            block_row = i // self.mb_cols
            block_col = i % self.mb_cols
            
            # 首先检查是否是背景
            if self._is_background(mv, global_motion):
                block_is_foreground[i] = False
                continue
            
            # 如果不是背景，查找连续的前景区域
            region = self._find_contiguous_region(i, vectors, block_row, block_col)
            
            # 如果区域太小，考虑将其作为背景
            if len(region) < 4:  # 最小区域大小阈值
                for block_idx in region:
                    block_is_foreground[block_idx] = False
            else:
                for block_idx in region:
                    block_is_foreground[block_idx] = True
                    processed.add(block_idx)
        
        return block_is_foreground  

    def apply_dct(self, block: np.ndarray) -> np.ndarray:
        """Apply 2D DCT to 8x8 block."""
        return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

    def quantize(self, dct_block: np.ndarray, n: int) -> np.ndarray:
        """Quantize DCT coefficients using uniform quantization with step size 2^n."""
        step = 2 ** n
        return np.round(dct_block / step)

    def process_macroblock(self, block: np.ndarray, is_foreground: bool, n1: int, n2: int) -> Tuple[int, List[float]]:
        """Process a 16x16 macroblock."""
        n = n1 if is_foreground else n2
        coeffs = []
        
        # Process each 8x8 block
        for i in range(0, 16, 8):
            for j in range(0, 16, 8):
                for c in range(3):  # RGB channels
                    block_8x8 = block[i:i+8, j:j+8, c].astype(float)
                    dct_block = self.apply_dct(block_8x8)
                    quant_block = self.quantize(dct_block, n)
                    coeffs.extend(quant_block.flatten())
        
        return (1 if is_foreground else 0), coeffs

    def _process_frame_chunk(self, args):
        """Process a chunk of frame in parallel."""
        frame_chunk, start_y, n1, n2 = args
        chunk_height = frame_chunk.shape[0]
        encoded_data = []
        
        # Process each block in the chunk
        for y in range(0, chunk_height, 16):
            for x in range(0, self.width, 16):
                abs_y = start_y + y  # Absolute y position in the full frame
                
                # Get current block
                curr_block = frame_chunk[y:y+16, x:x+16]
                if curr_block.shape[0] != 16 or curr_block.shape[1] != 16:
                    continue
                    
                # Calculate motion vector
                mv = self.calculate_motion_vector(curr_block, self.prev_frame, abs_y, x)
                
                # Determine if foreground (simplified for speed)
                motion_magnitude = np.sqrt(mv.dx**2 + mv.dy**2)
                is_foreground = motion_magnitude > self.motion_threshold
                
                # Process block
                block_type, coeffs = self.process_macroblock(curr_block, is_foreground, n1, n2)
                
                # Format data
                block_data = f"{block_type} " + " ".join(map(str, coeffs)) + "\n"
                encoded_data.append(block_data)
        
        return "".join(encoded_data).encode()

    def encode_frame(self, frame: np.ndarray, n1: int, n2: int) -> bytes:
        """Encode a single frame using parallel processing."""
        if not hasattr(self, 'use_multiprocessing'):
            # 如果是第一帧或不想使用并行处理，使用原始方法
            return self._encode_frame_single(frame, n1, n2)
        
        # Split frame into chunks for parallel processing
        chunks = []
        chunk_size = self.height // self.num_processes
        for i in range(self.num_processes):
            start_y = i * chunk_size
            end_y = start_y + chunk_size if i < self.num_processes - 1 else self.height
            chunks.append((frame[start_y:end_y], start_y, n1, n2))
        
        # Process chunks in parallel
        with Pool(self.num_processes) as pool:
            results = pool.map(self._process_frame_chunk, chunks)
        
        # Combine results
        return b''.join(results)

    def _encode_frame_single(self, frame: np.ndarray, n1: int, n2: int) -> bytes:
        """Original single-threaded frame encoding method."""
        # Calculate motion vectors for all blocks
        motion_vectors = []
        for y in range(0, self.height, 16):
            for x in range(0, self.width, 16):
                curr_block = self.get_block(frame, y, x)
                mv = self.calculate_motion_vector(curr_block, self.prev_frame, y, x)
                motion_vectors.append(mv)
        
        # Group blocks based on motion
        block_classifications = self.group_motion_vectors(motion_vectors)
        
        # Encode blocks
        encoded_data = []
        for i, mv in enumerate(motion_vectors):
            # Get block position
            block_y = mv.y
            block_x = mv.x
            
            # Get block
            curr_block = self.get_block(frame, block_y, block_x)
            
            # Determine if foreground
            is_foreground = block_classifications.get(i, False)
            
            # Process block
            block_type, coeffs = self.process_macroblock(curr_block, is_foreground, n1, n2)
            
            # Format data
            block_data = f"{block_type} " + " ".join(map(str, coeffs)) + "\n"
            encoded_data.append(block_data)
        
        return "".join(encoded_data).encode()

    def encode_video(self, input_path: str, n1: int, n2: int):
        """Encode the entire video file."""
        self.validate_parameters(n1, n2)
        output_path = str(Path(input_path).with_suffix('.cmp'))
        
        try:
            with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
                # Write header
                outfile.write(f"{n1} {n2}\n".encode())
                
                # Calculate total frames
                total_frames = os.path.getsize(input_path) // self.frame_size
                self.logger.info(f"Processing {total_frames} frames")
                
                # Process frames
                with tqdm(total=total_frames, desc="Encoding frames") as pbar:
                    while True:
                        frame = self.read_frame(infile)
                        if frame is None:
                            break
                        
                        # Encode frame
                        encoded_frame = self.encode_frame(frame, n1, n2)
                        outfile.write(encoded_frame)
                        
                        # Update previous frame
                        self.prev_frame = frame.copy()
                        pbar.update(1)
                
                self.logger.info(f"Encoding completed: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Video Encoder')
    parser.add_argument('input_video', help='Input RGB video file')
    parser.add_argument('n1', type=int, help='Quantization step for foreground (1-8)')
    parser.add_argument('n2', type=int, help='Quantization step for background (1-8)')
    
    args = parser.parse_args()
    
    if not Path(args.input_video).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_video}")
    
    encoder = VideoEncoder(args.input_video)
    encoder.encode_video(args.input_video, args.n1, args.n2)

if __name__ == "__main__":
    main()