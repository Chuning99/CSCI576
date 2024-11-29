from enum import Enum
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
class BlockType(Enum):
    BACKGROUND = 0
    FOREGROUND = 1

@dataclass
class MotionVector:
    """Represents a motion vector with its position and direction."""
    x: int  # Block x position
    y: int  # Block y position
    dx: int  # Motion in x direction
    dy: int  # Motion in y direction
    mad: float  # Mean absolute difference
class FrameBuffer:
    def __init__(self, buffer_size: int = 2):
        self.buffer_size = buffer_size
        self.frames = []
        self.current_frame_idx = -1
    
    def add_frame(self, frame: np.ndarray):
        if len(self.frames) >= self.buffer_size:
            self.frames.pop(0)
        self.frames.append(frame.copy())
        self.current_frame_idx += 1
    
    def get_previous_frame(self) -> Optional[np.ndarray]:
        if len(self.frames) < 2:
            return None
        return self.frames[-2]
    
    def clear(self):
        self.frames.clear()
        self.current_frame_idx = -1
class VideoEncoder:
    def __init__(self, input_path: str):
        """Initialize video encoder with automatic resolution detection."""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Detect video dimensions
        self._detect_dimensions(input_path)
        
        # Initialize parameters
        self.mb_rows = self.height // 16
        self.mb_cols = self.width // 16
        
        # Motion detection parameters
        self.motion_threshold = 20.0
        self.search_range = 16

        # Parallel processing setup
        self.use_multiprocessing = True
        self.num_processes = os.cpu_count()

        # Initialize frame buffer
        self.frame_buffer = FrameBuffer(buffer_size=2)
        
        self.logger.info(f"Initialized encoder: {self.width}x{self.height}, "
                        f"{self.mb_rows}x{self.mb_cols} macroblocks")
    def _write_header(self, outfile: BinaryIO, n1: int, n2: int):
        """写入压缩文件头部。"""
        header = f"{n1} {n2}\n"
        outfile.write(header.encode())

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
                self.num_frames = file_size // frame_bytes
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
    def _process_frame_chunk(self, args):
        """Process a chunk of frame in parallel."""
        frame_chunk, prev_frame_chunk, start_y, n1, n2 = args
        chunk_height = frame_chunk.shape[0]
        encoded_data = []
        
        # Process each block in the chunk
        for y in range(0, chunk_height - 15, 16):  # 修改循环范围
            for x in range(0, self.width - 15, 16):  # 修改循环范围
                abs_y = start_y + y
                
                # 确保我们有完整的16x16块
                if y + 16 > chunk_height or x + 16 > self.width:
                    continue
                    
                # 获取当前块
                curr_block = frame_chunk[y:y+16, x:x+16]
                
                # 获取对应的前一帧块
                if prev_frame_chunk is not None:
                    if y + 16 > prev_frame_chunk.shape[0] or x + 16 > prev_frame_chunk.shape[1]:
                        continue
                
                # 计算运动向量
                mv = self.calculate_motion_vector(curr_block, prev_frame_chunk, abs_y, x)
                
                # 确定是否为前景
                motion_magnitude = np.sqrt(mv.dx**2 + mv.dy**2)
                is_foreground = motion_magnitude > self.motion_threshold
                
                # 处理块
                block_type, coeffs = self.process_macroblock(curr_block, is_foreground, n1, n2)
                
                # 格式化数据
                block_data = f"{block_type} " + " ".join(map(str, coeffs)) + "\n"
                encoded_data.append(block_data)
        
        return "".join(encoded_data).encode()

    def calculate_motion_vector(self, curr_block: np.ndarray, prev_frame: np.ndarray, 
                            block_y: int, block_x: int) -> MotionVector:
        """改进的层级式运动估计算法。"""
        if prev_frame is None:
            return MotionVector(block_x, block_y, 0, 0, float('inf'))
        
        # 确保当前块是16x16
        if curr_block.shape[0] != 16 or curr_block.shape[1] != 16:
            return MotionVector(block_x, block_y, 0, 0, float('inf'))
            
        # 转换为YUV空间并只使用Y分量
        curr_y = cv2.cvtColor(curr_block, cv2.COLOR_RGB2YUV)[:,:,0]
        
        min_mad = float('inf')
        best_dx = best_dy = 0
        
        # 三层搜索策略
        search_patterns = [
            (16, 8),  # 第一层：大范围粗搜索
            (8, 4),   # 第二层：中等范围搜索
            (4, 2)    # 第三层：精细搜索
        ]
        
        curr_dx = curr_dy = 0
        for search_range, step in search_patterns:
            y_min = max(-search_range + curr_dy, -block_y)
            y_max = min(search_range + curr_dy, self.height - block_y - 16)
            x_min = max(-search_range + curr_dx, -block_x)
            x_max = min(search_range + curr_dx, self.width - block_x - 16)
            
            for dy in range(y_min, y_max + 1, step):
                for dx in range(x_min, x_max + 1, step):
                    ref_y = block_y + dy
                    ref_x = block_x + dx
                    
                    if (0 <= ref_y < self.height-15 and 0 <= ref_x < self.width-15 and
                        ref_y + 16 <= prev_frame.shape[0] and ref_x + 16 <= prev_frame.shape[1]):
                        ref_block = prev_frame[ref_y:ref_y+16, ref_x:ref_x+16]
                        if ref_block.shape[0] != 16 or ref_block.shape[1] != 16:
                            continue
                            
                        ref_y = cv2.cvtColor(ref_block, cv2.COLOR_RGB2YUV)[:,:,0]
                        
                        # 使用SAD代替MAD加快计算
                        sad = np.sum(np.abs(curr_y - ref_y))
                        if sad < min_mad:
                            min_mad = sad
                            best_dx, best_dy = dx, dy
                            curr_dx, curr_dy = dx, dy  # 更新搜索中心
        
        return MotionVector(block_x, block_y, best_dx, best_dy, min_mad / 256)

    def encode_frame(self, frame: np.ndarray, n1: int, n2: int) -> bytes:
        """Encode a single frame using parallel processing."""
        if not hasattr(self, 'use_multiprocessing'):
            return self._encode_frame_single(frame, n1, n2)
        
        chunks = []
        chunk_size = ((self.height + self.num_processes - 1) // self.num_processes) // 16 * 16
        prev_frame = self.frame_buffer.get_previous_frame()
        
        for i in range(0, self.height, chunk_size):
            end_y = min(i + chunk_size, self.height)
            # 确保块高度是16的倍数
            if end_y - i < 16:
                continue
            chunks.append((
                frame[i:end_y],
                prev_frame[i:end_y] if prev_frame is not None else None,
                i, n1, n2
            ))
        
        # Process chunks in parallel
        with Pool(self.num_processes) as pool:
            results = pool.map(self._process_frame_chunk, chunks)
        
        # Combine results
        return b''.join(results)
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
        """改进的背景检测算法。"""
        motion_magnitude = np.sqrt(mv.dx**2 + mv.dy**2)
        global_mag = np.sqrt(global_motion[0]**2 + global_motion[1]**2)
        
        # 静止判断
        if motion_magnitude < self.motion_threshold * 0.5:
            return True
        
        # 摄像机运动判断
        if global_mag > 0:
            motion_direction = np.array([mv.dx, mv.dy]) / motion_magnitude
            global_direction = np.array(global_motion) / global_mag
            
            # 计算方向相似度
            direction_similarity = np.dot(motion_direction, global_direction)
            
            # 计算幅度相似度
            magnitude_ratio = min(motion_magnitude, global_mag) / max(motion_magnitude, global_mag)
            
            return direction_similarity > 0.95 and magnitude_ratio > 0.7
        
        return False
    def _encode_frame_parallel(self, frame: np.ndarray, n1: int, n2: int) -> bytes:
        """改进的并行帧编码。"""
        height_per_process = ((self.height + self.num_processes - 1) 
                            // self.num_processes) // 16 * 16
        
        chunks = []
        for i in range(0, self.height, height_per_process):
            end = min(i + height_per_process, self.height)
            chunks.append((
                frame[i:end],
                self.prev_frame[i:end] if self.prev_frame is not None else None,
                i, n1, n2
            ))
        
        with Pool(self.num_processes) as pool:
            results = pool.map(self._process_frame_chunk, chunks)
        
        return b''.join(results)
    def _write_frame_data(self, outfile: BinaryIO, block_type: int, coeffs: List[float]):
        """规范化的文件写入。"""
        data = [str(block_type)]
        
        # 按RGB通道分组写入系数
        for i in range(0, len(coeffs), 64):
            channel_coeffs = coeffs[i:i+64]
            data.extend(map(str, channel_coeffs))
        
        outfile.write((' '.join(data) + '\n').encode())

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





    def _encode_frame_single(self, frame: np.ndarray, n1: int, n2: int) -> bytes:
        """Original single-threaded frame encoding method."""
        # Calculate motion vectors for all blocks
        motion_vectors = []
        prev_frame = self.frame_buffer.get_previous_frame()
        
        for y in range(0, self.height, 16):
            for x in range(0, self.width, 16):
                curr_block = self.get_block(frame, y, x)
                mv = self.calculate_motion_vector(curr_block, prev_frame, y, x)
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
        self.logger.info(f"Starting encoding with parameters n1={n1}, n2={n2}")
        self.validate_parameters(n1, n2)
        output_path = str(Path(input_path).with_suffix('.cmp'))
        temp_path = output_path + '.tmp'
        
        try:
            with open(input_path, 'rb') as infile, open(temp_path, 'wb') as outfile:
                self._write_header(outfile, n1, n2)
                
                with tqdm(total=self.num_frames) as pbar:
                    frame_count = 0
                    while frame_count < self.num_frames:
                        try:
                            frame = self.read_frame(infile)
                            if frame is None:
                                break
                                
                            # 使用frame_buffer而不是直接更新prev_frame
                            self.frame_buffer.add_frame(frame)
                            
                            encoded_frame = self.encode_frame(frame, n1, n2)
                            outfile.write(encoded_frame)
                            
                            frame_count += 1
                            pbar.update(1)
                            
                        except Exception as e:
                            self.logger.error(f"Frame {frame_count} encoding failed: {e}")
                            if frame_count == 0:
                                raise
                            continue
            
            os.replace(temp_path, output_path)
            self.logger.info(f"Encoding completed: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
        finally:
            self.frame_buffer.clear()
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