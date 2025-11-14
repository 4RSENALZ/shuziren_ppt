from __future__ import annotations

import base64
import json
import os
import re
import time
import uuid
import shutil
import tempfile
import subprocess
from fractions import Fraction
from threading import Event, Thread
from typing import Callable, Dict, List, Tuple, Optional, TYPE_CHECKING, Any

from docx import Document
from pptx import Presentation
from pptx.util import Inches

import cv2

from logger import logger

try:
    import comtypes
    import comtypes.client
except ImportError:  # pragma: no cover - optional dependency
    comtypes = None

if TYPE_CHECKING:
    from basereal import BaseReal
else:  # pragma: no cover - typing helper
    BaseReal = object  # type: ignore[assignment]


class DigitalHumanRenderer:
    """Helper to reuse a single digital human instance for multiple scripts."""

    def __init__(
        self,
        session_factory: Callable[[], int],
        builder: Callable[[int], BaseReal],
        warmup_delay: float = 2.0,
        speak_timeout: float = 180.0,
        max_preroll: float = 1.5,
    ) -> None:
        from basereal import BaseReal  # Local import to avoid cycles

        self._session_factory = session_factory
        self._builder = builder
        self._speak_timeout = speak_timeout
        self._max_preroll = max(0.0, float(max_preroll))
        self._start_event = Event()
        self._end_event = Event()
        self._quit_event = Event()
        self._ffmpeg_bin = self._locate_ffmpeg_binary()

        self.session_id = self._session_factory()
        self._nerfreal: BaseReal = self._builder(self.session_id)

        original_notify = getattr(self._nerfreal, "notify", None)

        def patched_notify(eventpoint):
            if callable(original_notify):
                original_notify(eventpoint)
            if not isinstance(eventpoint, dict):
                return
            status = eventpoint.get("status")
            if status == "start":
                self._start_event.set()
            elif status == "end":
                self._end_event.set()

        self._nerfreal.notify = patched_notify  # type: ignore[attr-defined]

        self._render_thread = Thread(
            target=self._nerfreal.render,
            args=(self._quit_event,),
            daemon=True,
        )
        self._render_thread.start()

        warmup_deadline = time.time() + max(0.5, warmup_delay)
        while (self._nerfreal.width == 0 or self._nerfreal.height == 0) and time.time() < warmup_deadline:
            time.sleep(0.1)

    def speak(self, text: str, output_path: str) -> str:
        if not text.strip():
            raise ValueError("朗读内容不能为空。")

        wait_deadline = time.time() + 20.0
        while (self._nerfreal.width == 0 or self._nerfreal.height == 0) and time.time() < wait_deadline:
            time.sleep(0.1)
        if self._nerfreal.width == 0 or self._nerfreal.height == 0:
            raise RuntimeError("数字人视频流未就绪，无法开始录制。")

        self._start_event.clear()
        self._end_event.clear()

        record_start = time.time()
        self._nerfreal.start_recording(output_path)
        self._nerfreal.put_msg_txt(text, {"source": "ppt"})

        if not self._start_event.wait(timeout=self._speak_timeout):
            self._nerfreal.stop_recording()
            raise TimeoutError("数字人朗读未按预期启动，请稍后重试。")

        event_start_time = time.time()

        speak_deadline = time.time() + self._speak_timeout
        poll_interval = 0.05  # shorter poll for quicker start/stop detection
        speech_start_time: Optional[float] = None
        while time.time() < speak_deadline:
            speaking_now = self._nerfreal.is_speaking()
            if speaking_now and speech_start_time is None:
                speech_start_time = time.time()
            if self._end_event.is_set() and not speaking_now:
                break
            time.sleep(poll_interval)
        else:
            self._nerfreal.stop_recording()
            raise TimeoutError("数字人朗读未在限定时间内完成。")

        # 等待音视频管线彻底排空
        drain_deadline = time.time() + 3.0
        while self._nerfreal.is_speaking() and time.time() < drain_deadline:
            time.sleep(poll_interval)
        time.sleep(0.1)

        self._nerfreal.stop_recording()
        self._nerfreal.flush_talk()

        effective_start = speech_start_time or event_start_time
        wait_estimate = max(0.0, effective_start - record_start)
        detected_wait = self._detect_audio_lead(output_path)
        wait_before_speech = detected_wait if detected_wait is not None else wait_estimate
        self._trim_preroll(output_path, wait_before_speech)
        return output_path

    def close(self) -> None:
        self._quit_event.set()
        if self._render_thread.is_alive():
            self._render_thread.join(timeout=5.0)

    def _locate_ffmpeg_binary(self) -> Optional[str]:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(root_dir, "ffmpeg", "ffmpeg.exe"),
            os.path.join(root_dir, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(root_dir, "ffmpeg", "ffmpeg"),
            os.path.join(root_dir, "ffmpeg", "bin", "ffmpeg"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")

    def _detect_audio_lead(self, video_path: str) -> Optional[float]:
        if not self._ffmpeg_bin or not os.path.isfile(video_path):
            return None

        command = [
            self._ffmpeg_bin,
            "-hide_banner",
            "-nostats",
            "-i",
            video_path,
            "-vn",
            "-af",
            "silencedetect=noise=-40dB:d=0.05",
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        except Exception as exc:
            logger.warning("[ppt-augment] detect audio lead failed: %s", exc)
            return None

        pattern = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")
        for line in result.stderr.splitlines():
            match = pattern.search(line)
            if match:
                try:
                    return max(0.0, float(match.group(1)))
                except ValueError:
                    continue
        return None

    def _trim_preroll(self, output_path: str, wait_before_speech: float) -> None:
        if self._max_preroll <= 0.0:
            return
        tolerance = 0.005
        if not self._ffmpeg_bin or not os.path.isfile(output_path):
            logger.warning("[ppt-augment] ffmpeg unavailable, skip preroll trimming")
            return

        current_wait = max(0.0, float(wait_before_speech))
        for attempt in range(3):
            if current_wait <= self._max_preroll + tolerance:
                break
            overshoot = 0.01
            trim_offset = max(0.0, current_wait - self._max_preroll + overshoot)
            if trim_offset <= 0.0:
                break

            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(tmp_fd)
            command = [
                self._ffmpeg_bin,
                "-y",
                "-i",
                output_path,
                "-ss",
                f"{trim_offset:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-movflags",
                "+faststart",
                "-reset_timestamps",
                "1",
                tmp_path,
            ]
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as exc:
                logger.warning("[ppt-augment] trim preroll failed: %s", exc)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                break

            try:
                os.replace(tmp_path, output_path)
            except Exception as exc:
                logger.warning("[ppt-augment] replace trimmed video failed: %s", exc)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                break

            detected_wait = self._detect_audio_lead(output_path)
            if detected_wait is not None:
                current_wait = max(0.0, detected_wait)
            else:
                current_wait = max(0.0, current_wait - trim_offset)



def generate_slide_previews(ppt_bytes: bytes, limit: int = 0) -> Tuple[List[str], Tuple[float, float]]:
    if comtypes is None:
        raise RuntimeError('需要安装 comtypes 并确保系统已安装 Microsoft PowerPoint 才能生成 PPT 预览。')

    workdir = tempfile.mkdtemp(prefix="pptpreview_")
    try:
        ppt_path = os.path.join(workdir, "input.pptx")
        with open(ppt_path, "wb") as f:
            f.write(ppt_bytes)

        ppt_path = os.path.abspath(ppt_path)

        export_dir = os.path.join(workdir, "slides")
        os.makedirs(export_dir, exist_ok=True)

        logger.info("[ppt-preview] start export for %s (limit=%s)", ppt_path, limit)
        comtypes.CoInitialize()
        powerpoint = None
        presentation = None
        try:
            powerpoint = comtypes.client.CreateObject("PowerPoint.Application")
            powerpoint.Visible = 1
            presentation = powerpoint.Presentations.Open(ppt_path, WithWindow=True)
            try:
                slide_width_pts = presentation.PageSetup.SlideWidth
                slide_height_pts = presentation.PageSetup.SlideHeight
                presentation.Export(export_dir, "PNG")
            finally:
                presentation.Close()
        finally:
            if powerpoint is not None:
                try:
                    powerpoint.Quit()
                except Exception:
                    pass
            comtypes.CoUninitialize()

        slide_width_in = float(slide_width_pts) / 72.0
        slide_height_in = float(slide_height_pts) / 72.0

        def _slide_order_key(entry: str) -> tuple[int, str]:
            match = re.search(r"(\d+)(?=\.[^.]+$)", entry)
            if match:
                try:
                    return int(match.group(1)), entry
                except ValueError:
                    pass
            return (10**9, entry)

        images = []
        ordered_files = sorted(os.listdir(export_dir), key=_slide_order_key)
        for idx, filename in enumerate(ordered_files):
            if limit and idx >= limit:
                break
            file_path = os.path.join(export_dir, filename)
            with open(file_path, "rb") as img_file:
                images.append(base64.b64encode(img_file.read()).decode("ascii"))

        if not images:
            raise RuntimeError("未能导出 PPT 预览图，请确认已安装 Microsoft PowerPoint。")

        logger.info(
            "[ppt-preview] done: %d slide images (~%.2f\" x %.2f\")",
            len(images),
            slide_width_in,
            slide_height_in,
        )
        return images, (slide_width_in, slide_height_in)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class PPTDigitalHumanAugmenter:
    def __init__(
        self,
        session_factory: Callable[[], int],
        builder: Callable[[int], BaseReal],
        video_position: Tuple[float, float, float, float] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._builder = builder
        if video_position is None:
            # (left, top, width, height) in inches
            self._video_position = (5.0, 1.0, 4.5, 3.2)
        else:
            self._video_position = video_position
        self._ffmpeg_bin = self._locate_ffmpeg_binary()
        self._course_base_height = 1080
        self._course_static_duration = 3.0
        self._session_store_root = os.path.join(tempfile.gettempdir(), "pptaugment_sessions")
        os.makedirs(self._session_store_root, exist_ok=True)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._max_sessions = 8

    def _parse_scripts(self, docx_path: str) -> Dict[int, str]:
        document = Document(docx_path)
        pattern = re.compile(r"^p(\d+)[\s:：\.-]*", re.IGNORECASE)
        scripts: Dict[int, List[str]] = {}
        current_slide = None
        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            match = pattern.match(text)
            if match:
                slide_index = int(match.group(1))
                content = text[match.end():].strip()
                current_slide = slide_index
                scripts[current_slide] = []
                if content:
                    scripts[current_slide].append(content)
            elif current_slide is not None:
                scripts[current_slide].append(text)
        return {idx: "\n".join(lines).strip() for idx, lines in scripts.items() if lines}

    def _locate_ffmpeg_binary(self) -> Optional[str]:
        root_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(root_dir, "ffmpeg", "ffmpeg.exe"),
            os.path.join(root_dir, "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(root_dir, "ffmpeg", "ffmpeg"),
            os.path.join(root_dir, "ffmpeg", "bin", "ffmpeg"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")

    def _ensure_video_aspect_ratio(self, video_path: str, target_ratio: Optional[float]) -> str:
        if not target_ratio or target_ratio <= 0:
            return video_path

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return video_path
        try:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        finally:
            cap.release()

        if width <= 0 or height <= 0:
            return video_path

        actual_ratio = float(width) / float(height)
        if abs(actual_ratio - target_ratio) <= 0.01:
            return video_path

        if not self._ffmpeg_bin:
            logger.warning("[ppt-augment] ffmpeg not available, skip aspect adjustment")
            return video_path

        fraction = Fraction(target_ratio).limit_denominator(1000)
        aspect_expr = f"{fraction.numerator}:{fraction.denominator}"
        base_dir = os.path.dirname(os.path.abspath(video_path)) or None
        tmp_fd, adjusted_path = tempfile.mkstemp(
            dir=base_dir,
            prefix=f"{os.path.splitext(os.path.basename(video_path))[0]}_dar_",
            suffix=".mp4",
        )
        os.close(tmp_fd)
        command = [
            self._ffmpeg_bin,
            "-y",
            "-i",
            video_path,
            "-vf",
            f"setsar=1,setdar={aspect_expr}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            adjusted_path,
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            logger.warning("[ppt-augment] adjust video aspect failed: %s", exc)
            if os.path.exists(adjusted_path):
                try:
                    os.remove(adjusted_path)
                except OSError:
                    pass
            return video_path

        try:
            shutil.move(adjusted_path, video_path)
        except Exception as exc:
            logger.warning("[ppt-augment] replace adjusted video failed: %s", exc)
            if os.path.exists(adjusted_path):
                try:
                    os.remove(adjusted_path)
                except OSError:
                    pass
            return video_path

        return video_path

    def _cleanup_session(self, session_id: str) -> None:
        record = self._active_sessions.pop(session_id, None)
        session_dir = None
        if record and isinstance(record, dict):
            session_dir = record.get("path")  # type: ignore[assignment]
        if not session_dir:
            session_dir = os.path.join(self._session_store_root, session_id)
        shutil.rmtree(session_dir, ignore_errors=True)

    def _register_session(self, session_id: str, session_dir: str) -> None:
        self._active_sessions[session_id] = {
            "path": session_dir,
            "created": time.time(),
        }
        while len(self._active_sessions) > self._max_sessions:
            oldest_id = min(
                self._active_sessions.items(),
                key=lambda item: item[1].get("created", 0.0),
            )[0]
            if oldest_id == session_id:
                break
            self._cleanup_session(oldest_id)

    def _save_session_meta(self, session_dir: str, meta: Dict[str, Any]) -> None:
        meta_path = os.path.join(session_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file, ensure_ascii=False, indent=2)

    def _persist_session_assets(
        self,
        ppt_bytes: bytes,
        generated_videos: Dict[int, str],
        applied_positions: Dict[int, Dict[str, float]],
        slide_dims_in: Tuple[float, float],
        total_slides: int,
        voice_id: Optional[str] = None,
    ) -> str:
        session_id = uuid.uuid4().hex
        session_dir = os.path.join(self._session_store_root, session_id)
        if os.path.isdir(session_dir):
            shutil.rmtree(session_dir, ignore_errors=True)
        os.makedirs(session_dir, exist_ok=True)

        ppt_filename = "original.pptx"
        with open(os.path.join(session_dir, ppt_filename), "wb") as f:
            f.write(ppt_bytes)

        videos_dir = os.path.join(session_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        video_index: Dict[str, str] = {}
        for slide_idx, src_path in generated_videos.items():
            if not os.path.isfile(src_path):
                continue
            dest_name = f"slide_{slide_idx:03d}.mp4"
            dest_path = os.path.join(videos_dir, dest_name)
            shutil.copy2(src_path, dest_path)
            video_index[str(slide_idx)] = dest_name

        meta: Dict[str, Any] = {
            "total_slides": total_slides,
            "applied_positions": applied_positions,
            "slide_dims": list(slide_dims_in),
            "videos": video_index,
            "ppt_filename": ppt_filename,
            "created_at": time.time(),
            "static_duration": self._course_static_duration,
            "voice": voice_id,
            "course": {
                "ready": False,
                "filename": "course.mp4",
                "duration": None,
                "segments": None,
                "resolution": None,
            },
        }

        self._save_session_meta(session_dir, meta)
        self._register_session(session_id, session_dir)
        return session_id

    def _load_session_assets(
        self,
        session_id: str,
    ) -> Tuple[bytes, Dict[int, str], Dict[int, Dict[str, float]], Tuple[float, float], int, str, Dict[str, Any]]:
        session_dir = os.path.join(self._session_store_root, session_id)
        if not os.path.isdir(session_dir):
            raise KeyError(f"未找到会话 {session_id}")

        meta_path = os.path.join(session_dir, "meta.json")
        if not os.path.isfile(meta_path):
            raise KeyError(f"会话 {session_id} 的元数据缺失")

        with open(meta_path, "r", encoding="utf-8") as meta_file:
            meta = json.load(meta_file)

        ppt_filename = meta.get("ppt_filename", "original.pptx")
        ppt_path = os.path.join(session_dir, ppt_filename)
        if not os.path.isfile(ppt_path):
            raise KeyError(f"会话 {session_id} 的 PPT 数据缺失")

        with open(ppt_path, "rb") as ppt_file:
            ppt_bytes = ppt_file.read()

        videos_dir = os.path.join(session_dir, "videos")
        video_index_raw = meta.get("videos", {}) or {}
        videos: Dict[int, str] = {}
        for key, filename in video_index_raw.items():
            try:
                slide_idx = int(key)
            except (TypeError, ValueError):
                continue
            candidate_path = os.path.join(videos_dir, filename)
            if os.path.isfile(candidate_path):
                videos[slide_idx] = candidate_path

        applied_positions_raw = meta.get("applied_positions", {}) or {}
        applied_positions: Dict[int, Dict[str, float]] = {}
        for key, position in applied_positions_raw.items():
            try:
                slide_idx = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(position, dict):
                applied_positions[slide_idx] = {
                    axis: float(position.get(axis, 0.0))
                    for axis in ("x", "y", "width", "height")
                }

        slide_dims_raw = meta.get("slide_dims", [0.0, 0.0])
        slide_dims_in = (
            float(slide_dims_raw[0]) if len(slide_dims_raw) > 0 else 0.0,
            float(slide_dims_raw[1]) if len(slide_dims_raw) > 1 else 0.0,
        )
        total_slides = int(meta.get("total_slides", max(videos.keys(), default=0)))

        self._register_session(session_id, session_dir)

        return ppt_bytes, videos, applied_positions, slide_dims_in, total_slides, session_dir, meta

    def compose_course_from_session(
        self,
        session_id: str,
        course_filename: str | None = None,
        static_duration: Optional[float] = None,
    ) -> Tuple[bytes, Dict[str, object]]:
        ppt_bytes, videos, applied_positions, slide_dims_in, total_slides, session_dir, meta = self._load_session_assets(session_id)

        if not self._ffmpeg_bin:
            raise RuntimeError("未检测到 ffmpeg，无法生成视频课。")

        course_info = meta.get("course") or {}
        desired_filename = course_filename or course_info.get("filename") or "course.mp4"
        stored_course_filename = os.path.basename(desired_filename)
        course_path = os.path.join(session_dir, stored_course_filename)
        if os.path.isfile(course_path):
            with open(course_path, "rb") as course_file:
                course_bytes = course_file.read()
            stored_resolution = course_info.get("resolution")
            if stored_resolution and isinstance(stored_resolution, list):
                cached_resolution = (stored_resolution[0], stored_resolution[1])
            else:
                cached_resolution = tuple(stored_resolution) if stored_resolution else (0, 0)

            return course_bytes, {
                "filename": stored_course_filename,
                "duration": course_info.get("duration"),
                "segments": course_info.get("segments"),
                "resolution": cached_resolution,
            }

        if static_duration is not None and static_duration > 0:
            original_duration = self._course_static_duration
            self._course_static_duration = static_duration
        else:
            original_duration = None

        workdir = tempfile.mkdtemp(prefix=f"course_{session_id}_")
        try:
            final_filename = course_filename or stored_course_filename
            course_bytes, course_meta = self._compose_course_video(
                ppt_bytes,
                videos,
                applied_positions,
                slide_dims_in,
                total_slides,
                workdir,
                final_filename,
            )

            target_path = os.path.join(session_dir, os.path.basename(final_filename))
            with open(target_path, "wb") as course_file:
                course_file.write(course_bytes)

            stored_course_filename = os.path.basename(final_filename)
            course_path = target_path
            resolution = course_meta.get("resolution")
            if resolution and isinstance(resolution, tuple):
                resolution_value: Any = list(resolution)
            else:
                resolution_value = resolution

            course_info.update(
                {
                    "ready": True,
                    "filename": stored_course_filename,
                    "duration": course_meta.get("duration"),
                    "segments": course_meta.get("segments"),
                    "resolution": resolution_value,
                    "generated_at": time.time(),
                }
            )
            meta["course"] = course_info
            self._save_session_meta(session_dir, meta)

            return course_bytes, course_meta
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            if original_duration is not None:
                self._course_static_duration = original_duration

    def _embed_video(self, slide, video_path: str, position: Optional[Dict[str, float]], slide_dims_in: Tuple[float, float]) -> Dict[str, float]:
        slide_width_in, slide_height_in = slide_dims_in
        target_ratio = None
        if position:
            left_ratio = max(0.0, min(1.0, position.get("x", 0.0)))
            top_ratio = max(0.0, min(1.0, position.get("y", 0.0)))
            width_ratio = position.get("width")
            height_ratio = position.get("height")
            default_width_ratio = self._video_position[2] / slide_width_in if slide_width_in else 0.3
            default_height_ratio = self._video_position[3] / slide_height_in if slide_height_in else 0.3
            if width_ratio is None or width_ratio <= 0:
                width_ratio = default_width_ratio
            if height_ratio is None or height_ratio <= 0:
                height_ratio = default_height_ratio
            width_ratio = max(0.02, min(1.0, width_ratio))
            height_ratio = max(0.02, min(1.0, height_ratio))
            left_in = slide_width_in * min(left_ratio, 1.0 - width_ratio)
            top_in = slide_height_in * min(top_ratio, 1.0 - height_ratio)
            width_in = slide_width_in * width_ratio
            height_in = slide_height_in * height_ratio
        else:
            left_in, top_in, width_in, height_in = self._video_position

        if height_in > 0:
            target_ratio = width_in / height_in

        video_path = self._ensure_video_aspect_ratio(video_path, target_ratio)
        movie = slide.shapes.add_movie(
            video_path,
            Inches(left_in),
            Inches(top_in),
            Inches(width_in),
            Inches(height_in),
            mime_type="video/mp4",
        )
        try:
            movie.lock_aspect_ratio = False
            movie.width = Inches(width_in)
            movie.height = Inches(height_in)
        except Exception:
            pass
        used_position = {
            "x": min(1.0, max(0.0, left_in / slide_width_in)) if slide_width_in else 0.0,
            "y": min(1.0, max(0.0, top_in / slide_height_in)) if slide_height_in else 0.0,
            "width": min(1.0, max(0.0, width_in / slide_width_in)) if slide_width_in else 0.0,
            "height": min(1.0, max(0.0, height_in / slide_height_in)) if slide_height_in else 0.0,
        }
        return used_position

    def _build_course_segment_with_avatar(
        self,
        background_path: str,
        video_path: str,
        position: Dict[str, float],
        target_width: int,
        target_height: int,
        workdir: str,
        slide_idx: int,
    ) -> Tuple[str, float]:
        overlay_x = int(round(position.get("x", 0.0) * target_width))
        overlay_y = int(round(position.get("y", 0.0) * target_height))
        overlay_w = int(round(position.get("width", 0.0) * target_width))
        overlay_h = int(round(position.get("height", 0.0) * target_height))
        overlay_w = max(2, min(target_width, overlay_w))
        overlay_h = max(2, min(target_height, overlay_h))
        if overlay_w % 2 != 0:
            overlay_w += 1
        if overlay_h % 2 != 0:
            overlay_h += 1
        max_x = max(0, target_width - overlay_w)
        max_y = max(0, target_height - overlay_h)
        overlay_x = max(0, min(max_x, overlay_x))
        overlay_y = max(0, min(max_y, overlay_y))

        duration = 0.0
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
                if fps > 0.0 and frame_count > 0.0:
                    duration = float(frame_count / fps)
            finally:
                cap.release()
        else:
            cap.release()

        segment_path = os.path.join(workdir, f"course_seg_{slide_idx:03d}.mp4")
        filter_complex = (
            f"[0:v]scale={target_width}:{target_height}[bg];"
            f"[1:v]scale={overlay_w}:{overlay_h}:force_original_aspect_ratio=increase,"
            f"crop={overlay_w}:{overlay_h},setsar=1[avatar];"
            f"[bg][avatar]overlay={overlay_x}:{overlay_y}:format=yuv420[outv]"
        )

        command = [
            self._ffmpeg_bin,
            "-y",
            "-loop",
            "1",
            "-i",
            background_path,
            "-i",
            video_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-shortest",
            "-r",
            "25",
            segment_path,
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return segment_path, max(duration, 0.0)

    def _build_course_static_segment(
        self,
        background_path: str,
        target_width: int,
        target_height: int,
        workdir: str,
        slide_idx: int,
    ) -> Tuple[str, float]:
        duration = max(0.5, float(self._course_static_duration))
        segment_path = os.path.join(workdir, f"course_seg_{slide_idx:03d}.mp4")
        filter_complex = f"[0:v]scale={target_width}:{target_height},format=yuv420p[outv]"
        command = [
            self._ffmpeg_bin,
            "-y",
            "-loop",
            "1",
            "-i",
            background_path,
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=mono:sample_rate=48000",
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-map",
            "1:a",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-movflags",
            "+faststart",
            "-t",
            f"{duration}",
            "-r",
            "25",
            segment_path,
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return segment_path, duration

    def _compose_course_video(
        self,
        ppt_bytes: bytes,
        generated_videos: Dict[int, str],
        applied_positions: Dict[int, Dict[str, float]],
        slide_dims_in: Tuple[float, float],
        total_slides: int,
        workdir: str,
        course_filename: str,
    ) -> Tuple[bytes, Dict[str, object]]:
        if not self._ffmpeg_bin:
            raise RuntimeError("未检测到 ffmpeg，无法生成视频课。")

        try:
            preview_images, _ = generate_slide_previews(ppt_bytes, total_slides)
        except Exception as exc:
            raise RuntimeError(f"导出幻灯片图像失败: {exc}") from exc

        if not preview_images:
            raise RuntimeError("未能生成幻灯片图像，无法生成视频课。")

        background_paths: Dict[int, str] = {}
        for idx, image_base64 in enumerate(preview_images):
            slide_idx = idx + 1
            bg_path = os.path.join(workdir, f"course_slide_{slide_idx:03d}.png")
            with open(bg_path, "wb") as f:
                f.write(base64.b64decode(image_base64))
            background_paths[slide_idx] = bg_path

        slide_width_in, slide_height_in = slide_dims_in
        if slide_width_in <= 0 or slide_height_in <= 0:
            raise RuntimeError("幻灯片尺寸无效，无法生成视频课。")

        target_height = int(round(self._course_base_height))
        if target_height < 2:
            target_height = 1080
        if target_height % 2 != 0:
            target_height += 1
        target_width = int(round(target_height * (slide_width_in / slide_height_in)))
        if target_width < 16:
            target_width = 1280
        if target_width % 2 != 0:
            target_width += 1

        segments: List[str] = []
        total_duration = 0.0
        for slide_idx in range(1, total_slides + 1):
            bg_path = background_paths.get(slide_idx)
            if not bg_path:
                continue
            video_path = generated_videos.get(slide_idx)
            if video_path and slide_idx in applied_positions:
                segment_path, duration = self._build_course_segment_with_avatar(
                    bg_path,
                    video_path,
                    applied_positions[slide_idx],
                    target_width,
                    target_height,
                    workdir,
                    slide_idx,
                )
            else:
                segment_path, duration = self._build_course_static_segment(
                    bg_path,
                    target_width,
                    target_height,
                    workdir,
                    slide_idx,
                )
            segments.append(segment_path)
            total_duration += max(duration, 0.0)

        if not segments:
            raise RuntimeError("未生成任何可用的视频片段，无法合成视频课。")

        concat_path = os.path.join(workdir, "course_concat.txt")
        with open(concat_path, "w", encoding="utf-8") as f:
            for segment in segments:
                safe_segment = segment.replace("\\", "/")
                f.write(f"file '{safe_segment}'\n")

        final_name = os.path.basename(course_filename)
        final_path = os.path.join(workdir, final_name)
        concat_command = [
            self._ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_path,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            final_path,
        ]
        subprocess.run(concat_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with open(final_path, "rb") as f:
            course_bytes = f.read()

        course_meta: Dict[str, object] = {
            "filename": final_name,
            "duration": total_duration,
            "segments": len(segments),
            "resolution": (target_width, target_height),
        }
        return course_bytes, course_meta

    def generate(
        self,
        ppt_bytes: bytes,
        docx_bytes: bytes,
        base_filename: str | None = None,
        positions: Optional[Dict[int, Dict[str, float]]] = None,
        produce_course: bool = False,
        voice_id: Optional[str] = None,
    ) -> Tuple[bytes, Dict[str, object], Optional[bytes]]:
        workdir = tempfile.mkdtemp(prefix="pptaugment_")
        try:
            ppt_path = os.path.join(workdir, "input.pptx")
            with open(ppt_path, "wb") as f:
                f.write(ppt_bytes)

            doc_path = os.path.join(workdir, "script.docx")
            with open(doc_path, "wb") as f:
                f.write(docx_bytes)

            scripts = self._parse_scripts(doc_path)
            if not scripts:
                raise ValueError("脚本文件中未找到形如 p1/p2 的段落标记。")

            presentation = Presentation(ppt_path)
            total_slides = len(presentation.slides)
            slide_width_in = presentation.slide_width / Inches(1)
            slide_height_in = presentation.slide_height / Inches(1)
            slide_dims_in = (slide_width_in, slide_height_in)

            def build_renderer(session_id: int) -> BaseReal:
                if voice_id:
                    return self._builder(session_id, voice_id)
                return self._builder(session_id)

            renderer = DigitalHumanRenderer(self._session_factory, build_renderer)
            generated_videos: Dict[int, str] = {}
            applied_positions: Dict[int, Dict[str, float]] = {}
            try:
                for slide_idx, text in scripts.items():
                    if slide_idx < 1 or slide_idx > total_slides:
                        continue
                    output_video = os.path.join(workdir, f"slide_{slide_idx:03d}.mp4")
                    renderer.speak(text, output_video)
                    generated_videos[slide_idx] = output_video
            finally:
                renderer.close()

            if not generated_videos:
                raise RuntimeError("未能成功生成任何数字人讲解视频，请检查脚本内容。")

            for slide_idx, video_path in generated_videos.items():
                slide = presentation.slides[slide_idx - 1]
                position = None
                if positions:
                    position = positions.get(slide_idx)
                applied = self._embed_video(slide, video_path, position, slide_dims_in)
                if applied:
                    applied_positions[slide_idx] = applied

            output_path = os.path.join(workdir, "augmented.pptx")
            presentation.save(output_path)

            with open(output_path, "rb") as f:
                result_bytes = f.read()

            safe_name = base_filename or "livetalking-augmented"
            if not safe_name.lower().endswith(".pptx"):
                safe_name += ".pptx"

            meta = {
                "filename": safe_name,
                "total_slides": total_slides,
                "video_slides": sorted(generated_videos.keys()),
            }
            session_id = self._persist_session_assets(
                ppt_bytes,
                generated_videos,
                applied_positions,
                slide_dims_in,
                total_slides,
                voice_id,
            )
            meta["session_id"] = session_id
            course_bytes: Optional[bytes] = None
            if produce_course:
                base_stem = os.path.splitext(os.path.basename(safe_name))[0]
                course_filename = f"{base_stem}-course.mp4"
                course_bytes, course_meta = self.compose_course_from_session(session_id, course_filename)
                meta.update(
                    {
                        "course_filename": course_meta.get("filename", course_filename),
                        "course_duration": course_meta.get("duration"),
                        "course_segments": course_meta.get("segments"),
                        "course_resolution": course_meta.get("resolution"),
                    }
                )

            meta["voice"] = voice_id
            return result_bytes, meta, course_bytes
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
