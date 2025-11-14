###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import math
import torch
import numpy as np

import subprocess
import os
import time
import uuid
import shutil
import cv2
import glob
import resampy

import queue
from queue import Queue
from threading import Thread, Event, Lock
from collections import deque
from io import BytesIO
import soundfile as sf

import asyncio
from av import AudioFrame, VideoFrame

import av
from fractions import Fraction

from ttsreal import EdgeTTS,SovitsTTS,XTTS,CosyVoiceTTS,FishTTS,TencentTTS,DoubaoTTS,IndexTTS2,AzureTTS
from logger import logger

from tqdm import tqdm
def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def play_audio(quit_event,queue):        
    import pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(
        rate=16000,
        channels=1,
        format=8,
        output=True,
        output_device_index=1,
    )
    stream.start_stream()
    # while queue.qsize() <= 0:
    #     time.sleep(0.1)
    while not quit_event.is_set():
        stream.write(queue.get(block=True))
    stream.close()

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.sessionid = self.opt.sessionid

        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt,self)
        elif opt.tts == "gpt-sovits":
            self.tts = SovitsTTS(opt,self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt,self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt,self)
        elif opt.tts == "fishtts":
            self.tts = FishTTS(opt,self)
        elif opt.tts == "tencent":
            self.tts = TencentTTS(opt,self)
        elif opt.tts == "doubao":
            self.tts = DoubaoTTS(opt,self)
        elif opt.tts == "indextts2":
            self.tts = IndexTTS2(opt,self)
        elif opt.tts == "azuretts":
            self.tts = AzureTTS(opt,self)

        self.speaking = False

        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0
        self.record_output_path = os.path.join("data", "record.mp4")
        self._temp_video_path = None
        self._temp_audio_path = None
        self._temp_dir = os.path.join("data", "tmp")
        self._ffmpeg_binary: str | None = None
        self._record_video_shape: tuple[int, int] | None = None
        self._record_audio_started = False
        self._frame_resize_warned = False

        # default state: 0 (normal). Will be adjusted after loading custom items.
        self.curr_state = 0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self.__loadcustom()

        self.chat_history = deque(maxlen=500)
        self.chat_lock = Lock()
        self.chat_counter = 0
        # if audiotype 2 exists in custom items, set it as the default running state
        try:
            if 2 in self.custom_audio_index:
                self.curr_state = 2
        except Exception:
            pass

    def add_chat_message(self, role:str, text:str, meta:dict|None=None) -> int:
        cleaned = (text or "").strip()
        if not cleaned:
            return self.chat_counter
        entry_meta = meta if isinstance(meta, dict) and meta else None
        timestamp = time.time()
        with self.chat_lock:
            self.chat_counter += 1
            entry = {
                "id": self.chat_counter,
                "role": role or "assistant",
                "text": cleaned,
                "timestamp": timestamp,
            }
            if entry_meta:
                entry["meta"] = entry_meta
            self.chat_history.append(entry)
            return entry["id"]

    def get_chat_messages(self, after_id:int=0) -> list[dict]:
        with self.chat_lock:
            return [
                {key: value for key, value in entry.items()}
                for entry in self.chat_history
                if entry["id"] > after_id
            ]

    def reset_chat_history(self) -> None:
        with self.chat_lock:
            self.chat_history.clear()
            self.chat_counter = 0

    def put_msg_txt(self,msg,datainfo:dict={}):
        self.add_chat_message("assistant", msg, datainfo)
        self.tts.put_msg_txt(msg,datainfo)
    
    def put_audio_frame(self,audio_chunk,datainfo:dict={}): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,datainfo)

    def put_audio_file(self,filebyte,datainfo:dict={}): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:  #and self.state==State.RUNNING
            self.put_audio_frame(stream[idx:idx+self.chunk],datainfo)
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        self.tts.flush_talk()
        self.asr.flush_talk()

    def is_speaking(self)->bool:
        return self.speaking
    
    def __loadcustom(self):
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            # 循环播放audiotype=2的内容
            item.setdefault('loop', True)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        # 默认启动后直接进入audiotype=2状态
        if 2 in self.custom_audio_index:
            self.curr_state = 2
        else:
            self.curr_state = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def notify(self,eventpoint):
        logger.info("notify:%s",eventpoint)

    def start_recording(self, output_path:str|None=None):
        """开始录制视频"""
        if self.recording:
            return

        if output_path:
            self.record_output_path = output_path
        else:
            self.record_output_path = os.path.join("data", "record.mp4")

        output_dir = os.path.dirname(self.record_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        os.makedirs(self._temp_dir, exist_ok=True)

        temp_id = f"{self.sessionid}_{uuid.uuid4().hex}"
        self._temp_video_path = os.path.join(self._temp_dir, f"{temp_id}.mp4")
        self._temp_audio_path = os.path.join(self._temp_dir, f"{temp_id}.aac")

        self._record_video_pipe = None
        self._record_audio_pipe = None
        self._record_video_shape = None
        self._record_audio_started = False
        self._frame_resize_warned = False
        self.recording = True
        # self.recordq_video.queue.clear()
        # self.recordq_audio.queue.clear()
        # self.container = av.open(path, mode="w")
    
        # process_thread = Thread(target=self.record_frame, args=())
        # process_thread.start()
    
    def _ensure_video_recorder(self, width: int, height: int) -> None:
        if self._record_video_pipe is not None:
            return
        ffmpeg_bin = self._get_ffmpeg_binary()
        command = [
            ffmpeg_bin,
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}",
            '-r', '25',
            '-i', '-',
            '-pix_fmt', 'yuv420p',
            '-vcodec', 'libx264',
            '-preset', 'veryfast',
            '-crf', '18',
            self._temp_video_path,
        ]
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
        self._record_video_shape = (width, height)

    def _ensure_audio_recorder(self) -> None:
        if self._record_audio_pipe is not None:
            return
        ffmpeg_bin = self._get_ffmpeg_binary()
        command = [
            ffmpeg_bin,
            '-y', '-vn',
            '-f', 's16le',
            '-ac', '1',
            '-ar', str(self.sample_rate),
            '-i', '-',
            '-acodec', 'aac',
            self._temp_audio_path,
        ]
        self._record_audio_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
        self._record_audio_started = True

    def _prepare_frame_for_recording(
        self,
        image: np.ndarray,
        expected_shape: tuple[int, int],
    ) -> np.ndarray:
        expected_width, expected_height = expected_shape
        if expected_width <= 0 or expected_height <= 0:
            return image

        current_height, current_width = image.shape[:2]
        if current_width <= 0 or current_height <= 0:
            return image

        if (current_width, current_height) == expected_shape:
            return image

        if not self._frame_resize_warned:
            logger.warning(
                "[record] frame size changed during recording: %dx%d -> %dx%d, resizing to keep stream stable",
                current_width,
                current_height,
                expected_width,
                expected_height,
            )
            self._frame_resize_warned = True

        target_ratio = expected_width / expected_height
        current_ratio = current_width / current_height

        if abs(target_ratio - current_ratio) <= 0.05:
            interpolation = cv2.INTER_AREA if current_width > expected_width or current_height > expected_height else cv2.INTER_CUBIC
            return cv2.resize(image, (expected_width, expected_height), interpolation=interpolation)

        scale = min(expected_width / current_width, expected_height / current_height)
        scaled_width = max(1, int(round(current_width * scale)))
        scaled_height = max(1, int(round(current_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)

        canvas = np.zeros((expected_height, expected_width, image.shape[2]), dtype=image.dtype)
        offset_x = (expected_width - scaled_width) // 2
        offset_y = (expected_height - scaled_height) // 2
        canvas[offset_y:offset_y + scaled_height, offset_x:offset_x + scaled_width] = resized
        return canvas

    def record_video_data(self,image):
        if image is None:
            return
        current_height, current_width = image.shape[:2]
        if self.width != current_width or self.height != current_height:
            self.height = current_height
            self.width = current_width
        if not self.recording:
            return
        self._ensure_video_recorder(current_width, current_height)
        expected = self._record_video_shape
        if expected and (current_width, current_height) != expected:
            image = self._prepare_frame_for_recording(image, expected)
            current_height, current_width = image.shape[:2]
            if self.width != current_width or self.height != current_height:
                self.height = current_height
                self.width = current_width
        if self._record_video_pipe and self._record_video_pipe.stdin:
            self._record_video_pipe.stdin.write(image.tobytes())

    def record_audio_data(self,frame):
        if self.recording:
            self._ensure_audio_recorder()
            if self._record_audio_pipe and self._record_audio_pipe.stdin:
                self._record_audio_pipe.stdin.write(frame.tobytes())
    
    # def record_frame(self): 
    #     videostream = self.container.add_stream("libx264", rate=25)
    #     videostream.codec_context.time_base = Fraction(1, 25)
    #     audiostream = self.container.add_stream("aac")
    #     audiostream.codec_context.time_base = Fraction(1, 16000)
    #     init = True
    #     framenum = 0       
    #     while self.recording:
    #         try:
    #             videoframe = self.recordq_video.get(block=True, timeout=1)
    #             videoframe.pts = framenum #int(round(framenum*0.04 / videostream.codec_context.time_base))
    #             videoframe.dts = videoframe.pts
    #             if init:
    #                 videostream.width = videoframe.width
    #                 videostream.height = videoframe.height
    #                 init = False
    #             for packet in videostream.encode(videoframe):
    #                 self.container.mux(packet)
    #             for k in range(2):
    #                 audioframe = self.recordq_audio.get(block=True, timeout=1)
    #                 audioframe.pts = int(round((framenum*2+k)*0.02 / audiostream.codec_context.time_base))
    #                 audioframe.dts = audioframe.pts
    #                 for packet in audiostream.encode(audioframe):
    #                     self.container.mux(packet)
    #             framenum += 1
    #         except queue.Empty:
    #             print('record queue empty,')
    #             continue
    #         except Exception as e:
    #             print(e)
    #             #break
    #     for packet in videostream.encode(None):
    #         self.container.mux(packet)
    #     for packet in audiostream.encode(None):
    #         self.container.mux(packet)
    #     self.container.close()
    #     self.recordq_video.queue.clear()
    #     self.recordq_audio.queue.clear()
    #     print('record thread stop')
		
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
        self.recording = False 
        try:
            if self._record_video_pipe and self._record_video_pipe.stdin:
                self._record_video_pipe.stdin.close()
            if self._record_video_pipe:
                self._record_video_pipe.wait()
            if self._record_audio_pipe and self._record_audio_pipe.stdin:
                self._record_audio_pipe.stdin.close()
            if self._record_audio_pipe:
                self._record_audio_pipe.wait()

            ffmpeg_bin = self._get_ffmpeg_binary()
            combine_cmd = [
                ffmpeg_bin, '-y',
                '-i', self._temp_video_path,
                '-i', self._temp_audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-ar', '48000',
                '-movflags', '+faststart',
                self.record_output_path
            ]
            subprocess.run(combine_cmd, check=True)
        finally:
            if self._record_video_pipe:
                self._record_video_pipe = None
            if self._record_audio_pipe:
                self._record_audio_pipe = None
            if self._temp_audio_path and os.path.exists(self._temp_audio_path):
                try:
                    os.remove(self._temp_audio_path)
                except OSError:
                    pass
            if self._temp_video_path and os.path.exists(self._temp_video_path):
                try:
                    os.remove(self._temp_video_path)
                except OSError:
                    pass

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        """Return a chunk (length self.chunk) for given audiotype.
        Supports optional looping when custom_opt[audiotype]['loop'] is True.
        If not looping and audio ends, pad the returned chunk with zeros and
        set curr_state to 1 (silent).
        """
        arr = self.custom_audio_cycle[audiotype]
        length = arr.shape[0]
        idx = int(self.custom_audio_index[audiotype])
        # guard: empty audio
        if length == 0:
            self.curr_state = 1
            return np.zeros(self.chunk, dtype=np.float32)

        loop = bool(self.custom_opt.get(audiotype, {}).get('loop', False))

        # normal case: enough samples remain
        if idx + self.chunk <= length:
            stream = arr[idx: idx + self.chunk]
            self.custom_audio_index[audiotype] = idx + self.chunk
            # if reached exactly the end
            if self.custom_audio_index[audiotype] >= length and not loop:
                self.curr_state = 1
            elif self.custom_audio_index[audiotype] >= length and loop:
                # wrap around
                self.custom_audio_index[audiotype] = self.custom_audio_index[audiotype] % length
            return stream

        # idx + chunk > length -> need to handle tail
        tail = arr[idx: length]
        need = self.chunk - tail.shape[0]
        if loop:
            # take from beginning to fill
            head_take = need % length if need > length else need
            head = arr[0: head_take]
            stream = np.concatenate((tail, head), axis=0) if head.size else tail
            # set new index
            self.custom_audio_index[audiotype] = head_take
            return stream
        else:
            # not looping: pad with zeros to full chunk and switch to silent state
            pad = np.zeros(need, dtype=np.float32)
            stream = np.concatenate((tail, pad), axis=0)
            self.custom_audio_index[audiotype] = length
            self.curr_state = 1
            return stream
    
    def set_custom_state(self,audiotype, reinit=True):
        print('set_custom_state:',audiotype)
        if self.custom_audio_index.get(audiotype) is None:
            return
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0

    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        if enable_transition:
            _last_speaking = False
            _transition_start = time.time()
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存
        
        if self.opt.transport=='virtualcam':
            import pyvirtualcam
            vircam = None

            audio_tmp = queue.Queue(maxsize=3000)
            audio_thread = Thread(target=play_audio, args=(quit_event,audio_tmp,), daemon=True, name="pyaudio_stream")
            audio_thread.start()
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            # defensive: ensure audio_frames has at least 2 entries to avoid
            # downstream index errors (some inference producers may provide fewer)
            if not isinstance(audio_frames, list):
                try:
                    audio_frames = list(audio_frames)
                except Exception:
                    audio_frames = []
            if len(audio_frames) < 2:
                # determine filler type: prefer existing type, else 1 (silent)
                if len(audio_frames) == 0:
                    filler_type = 1
                else:
                    filler_type = audio_frames[0][1]
                while len(audio_frames) < 2:
                    audio_frames.append((np.zeros(self.chunk, dtype=np.float32), filler_type, None))
            
            if enable_transition:
                # 检测状态变化
                current_speaking = not (audio_frames[0][1]!=0 and audio_frames[1][1]!=0)
                if current_speaking != _last_speaking:
                    logger.info(f"状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                    _transition_start = time.time()
                _last_speaking = current_speaking

            if audio_frames[0][1]!=0 and audio_frames[1][1]!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            cv2.putText(combine_frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            if self.opt.transport=='virtualcam':
                if vircam==None:
                    height, width,_= combine_frame.shape
                    vircam = pyvirtualcam.Camera(width=width, height=height, fps=25, fmt=pyvirtualcam.PixelFormat.BGR,print_fps=True)
                vircam.send(combine_frame)
            elif video_track is not None and loop is not None:
                image = combine_frame
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                frame,type,eventpoint = audio_frame
                frame = (frame * 32767).astype(np.int16)

                if self.opt.transport=='virtualcam':
                    audio_tmp.put(frame.tobytes()) #TODO
                elif audio_track is not None and loop is not None:
                    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                else:
                    if eventpoint:
                        self.notify(eventpoint)
                self.record_audio_data(frame)
            if self.opt.transport=='virtualcam':
                vircam.sleep_until_next_frame()
        if self.opt.transport=='virtualcam':
            audio_thread.join()
            vircam.close()
        logger.info('basereal process_frames thread stop') 

    def _get_ffmpeg_binary(self) -> str:
        if self._ffmpeg_binary and os.path.isfile(self._ffmpeg_binary):
            return self._ffmpeg_binary

        env_path = os.environ.get('FFMPEG_BINARY')
        candidates = []
        if env_path:
            candidates.append(env_path)
        which_path = shutil.which('ffmpeg')
        if which_path:
            candidates.append(which_path)
        local_ffmpeg = os.path.join(os.path.dirname(__file__), 'wav2lip', 'ffmpeg.exe')
        candidates.append(local_ffmpeg)

        for path in candidates:
            if path and os.path.isfile(path):
                self._ffmpeg_binary = path
                return path

        raise FileNotFoundError('未找到 ffmpeg 可执行文件，请确保已安装或在环境变量 FFMPEG_BINARY 中指定路径。')
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1