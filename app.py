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

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import numpy as np
import cv2
from threading import Thread,Event
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
from basereal import BaseReal
from llm import llm_response

import argparse
import copy
import random
import shutil
import asyncio
import torch
import os
import webbrowser
from typing import Dict, Optional, Tuple
from pathlib import Path
import tempfile
import subprocess
import sys
import edge_tts
from logger import logger
import gc
from urllib.parse import quote

from ppt_augmenter import PPTDigitalHumanAugmenter, generate_slide_previews


app = Flask(__name__)
#sockets = Sockets(app)
nerfreals:Dict[int, BaseReal] = {} #sessionid:BaseReal
opt = None
model = None
avatar = None
ppt_generator: PPTDigitalHumanAugmenter | None = None

PROJECT_ROOT = Path(__file__).resolve().parent
AVATAR_RESULTS_ROOT = PROJECT_ROOT / "wav2lip" / "results" / "avatars"
AVATAR_RESULTS_ROOT_FALLBACK = PROJECT_ROOT / "results" / "avatars"
AVATAR_TARGET_ROOT = PROJECT_ROOT / "data" / "avatars"
CUSTOM_AVATAR_BASE_ROOT = PROJECT_ROOT / "data" / "customvideo"
CUSTOM_CONFIG_PATH = PROJECT_ROOT / "data" / "custom_config.json"
DEFAULT_AVATAR_NAME = "shuziren_test2"
_ffmpeg_candidates = []
if os.name == "nt":
    _ffmpeg_candidates.extend([
        PROJECT_ROOT / "ffmpeg" / "ffmpeg.exe",
        PROJECT_ROOT / "ffmpeg" / "bin" / "ffmpeg.exe",
    ])
else:
    _ffmpeg_candidates.extend([
        PROJECT_ROOT / "ffmpeg" / "ffmpeg",
        PROJECT_ROOT / "ffmpeg" / "bin" / "ffmpeg",
    ])

FFMPEG_BIN = None
for _candidate in _ffmpeg_candidates:
    if _candidate.exists():
        FFMPEG_BIN = str(_candidate)
        break

if FFMPEG_BIN is None:
    FFMPEG_BIN = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EDGE_TTS_PREVIEW_TEXT = "你好，我是您的数字人授课助理"
EDGE_TTS_PREVIEW_VOICE_FALLBACK = "zh-CN-YunxiaNeural"
EDGE_TTS_PREVIEW_MAX_VOICE_LENGTH = 128
EDGE_TTS_PREVIEW_MAX_TEXT_LENGTH = 200


def _find_first_thumbnail(avatar_dir: Path) -> Path | None:
    full_dir = avatar_dir / "full_imgs"
    if not full_dir.is_dir():
        return None
    for candidate in sorted(full_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def _probe_avatar_resolution(avatar_name: str) -> tuple[int, int] | None:
    candidate_dirs: list[Path] = [
        CUSTOM_AVATAR_BASE_ROOT / avatar_name / "image",
        AVATAR_TARGET_ROOT / avatar_name / "full_imgs",
        AVATAR_RESULTS_ROOT / avatar_name / "full_imgs",
        AVATAR_RESULTS_ROOT_FALLBACK / avatar_name / "full_imgs",
    ]

    for directory in candidate_dirs:
        if not directory.is_dir():
            continue
        try:
            candidates = sorted(
                (
                    path
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=lambda path: path.name,
            )
        except OSError:
            continue

        for image_path in candidates:
            image = cv2.imread(str(image_path))
            if image is None or image.size == 0:
                continue
            height, width = image.shape[:2]
            if width > 0 and height > 0:
                return width, height

    return None


def _parse_positions_field(raw_value: str | None) -> Dict[int, Dict[str, float]] | None:
    if not raw_value:
        return None
    try:
        raw_positions = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("坐标数据解析失败") from exc

    parsed: Dict[int, Dict[str, float]] = {}
    if isinstance(raw_positions, dict):
        for key, value in raw_positions.items():
            try:
                slide_idx = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            clean_pos: Dict[str, float] = {}
            for axis in ("x", "y", "width", "height"):
                if axis not in value:
                    continue
                try:
                    clean_value = float(value[axis])
                except (TypeError, ValueError):
                    continue
                clean_pos[axis] = min(1.0, max(0.0, clean_value))
            if clean_pos:
                parsed[slide_idx] = clean_pos

    return parsed or None


def _sanitize_voice_id(raw_voice: str | None) -> str:
    if not raw_voice:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_:-]", "", raw_voice.strip())
    if not cleaned:
        return ""
    return cleaned[:EDGE_TTS_PREVIEW_MAX_VOICE_LENGTH]


def _load_custom_config_entry() -> tuple[list, dict, bool, str | None]:
    config_payload: list = []
    config_existed = CUSTOM_CONFIG_PATH.exists()
    original_config_text: str | None = None
    if config_existed:
        try:
            original_config_text = CUSTOM_CONFIG_PATH.read_text(encoding="utf-8")
            config_payload = json.loads(original_config_text)
            if not isinstance(config_payload, list):
                config_payload = []
        except Exception as exc:
            logger.warning("[avatar] load custom config failed: %s", exc)
            config_payload = []
            original_config_text = None

    if config_payload and isinstance(config_payload[0], dict):
        config_entry = config_payload[0]
    else:
        config_entry = {}
        config_payload = [config_entry]

    return config_payload, config_entry, config_existed, original_config_text


def _persist_custom_config(
    config_payload: list,
    config_existed: bool,
    original_config_text: str | None,
) -> tuple[bool, Exception | None]:
    try:
        with open(CUSTOM_CONFIG_PATH, "w", encoding="utf-8") as cfg_file:
            json.dump(config_payload, cfg_file, indent=4, ensure_ascii=False)
        return True, None
    except Exception as exc:
        logger.exception("[avatar] write custom config failed")
        if config_existed and original_config_text is not None:
            try:
                with open(CUSTOM_CONFIG_PATH, "w", encoding="utf-8") as cfg_file:
                    cfg_file.write(original_config_text)
            except Exception as restore_exc:
                logger.warning("[avatar] restore custom config failed: %s", restore_exc)
        elif not config_existed:
            try:
                CUSTOM_CONFIG_PATH.unlink(missing_ok=True)
            except Exception as restore_exc:
                logger.warning("[avatar] remove custom config failed: %s", restore_exc)
        return False, exc


def _tighten_idle_frame_edges(
    image: np.ndarray,
    threshold: int,
    min_nonblack_ratio: float,
) -> Tuple[np.ndarray, bool]:
    trimmed = False
    working = image
    for _ in range(2):
        height, width = working.shape[:2]
        if height < 2 or width < 2:
            break
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        adaptive_threshold = max(threshold, int(gray.mean() * 0.3))
        valid = gray > adaptive_threshold
        row_ratio = valid.sum(axis=1) / float(width)
        col_ratio = valid.sum(axis=0) / float(height)

        top = 0
        while top < height - 1 and row_ratio[top] <= min_nonblack_ratio:
            top += 1
        bottom = height - 1
        while bottom > top and row_ratio[bottom] <= min_nonblack_ratio:
            bottom -= 1

        left = 0
        while left < width - 1 and col_ratio[left] <= min_nonblack_ratio:
            left += 1
        right = width - 1
        while right > left and col_ratio[right] <= min_nonblack_ratio:
            right -= 1

        if top == 0 and left == 0 and bottom == height - 1 and right == width - 1:
            break

        working = working[top:bottom + 1, left:right + 1]
        trimmed = True

    return working, trimmed


def _remove_black_borders_from_image(
    image: np.ndarray,
    threshold: int = 12,
    min_nonblack_ratio: float = 0.02,
    margin: int = 4,
    stretch_to_fill: bool = True,
) -> Tuple[np.ndarray, bool]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image, False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = max(threshold, int(gray.mean() * 0.3))
    valid = gray > adaptive_threshold

    row_ratio = valid.sum(axis=1) / float(width)
    col_ratio = valid.sum(axis=0) / float(height)
    row_mask = row_ratio > min_nonblack_ratio
    col_mask = col_ratio > min_nonblack_ratio
    if not np.any(row_mask) or not np.any(col_mask):
        return image, False

    top = int(np.argmax(row_mask))
    bottom = int(len(row_mask) - np.argmax(row_mask[::-1]) - 1)
    left = int(np.argmax(col_mask))
    right = int(len(col_mask) - np.argmax(col_mask[::-1]) - 1)

    top = max(0, top - margin)
    bottom = min(height - 1, bottom + margin)
    left = max(0, left - margin)
    right = min(width - 1, right + margin)

    if top <= 0 and left <= 0 and bottom >= height - 1 and right >= width - 1:
        return image, False

    new_height = bottom - top + 1
    new_width = right - left + 1
    if new_height < int(height * 0.5) or new_width < int(width * 0.5):
        return image, False

    cropped = image[top:bottom + 1, left:right + 1]
    tightened_image, tightened = _tighten_idle_frame_edges(cropped, threshold, min_nonblack_ratio)
    if tightened and (
        tightened_image.shape[0] < int(height * 0.45)
        or tightened_image.shape[1] < int(width * 0.45)
    ):
        tightened_image = cropped

    if stretch_to_fill and (
        tightened_image.shape[0] != height or tightened_image.shape[1] != width
    ):
        tightened_image = cv2.resize(tightened_image, (width, height), interpolation=cv2.INTER_CUBIC)

    return tightened_image, True


def _cleanup_idle_frames(images_dir: Path) -> Tuple[int, int]:
    processed = 0
    total = 0
    if not images_dir.is_dir():
        return processed, total

    candidates = sorted(
        (
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name,
    )
    for image_path in candidates:
        total += 1
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        cleaned_image, changed = _remove_black_borders_from_image(image)
        if not changed:
            continue
        if cv2.imwrite(str(image_path), cleaned_image):
            processed += 1

    return processed, total

_avatar_generation_lock: asyncio.Lock | None = None
        

#####webrtc###############################
pcs = set()


def _get_avatar_generation_lock() -> asyncio.Lock:
    global _avatar_generation_lock
    if _avatar_generation_lock is None:
        _avatar_generation_lock = asyncio.Lock()
    return _avatar_generation_lock


def _normalize_avatar_name(raw: str) -> str:
    name = raw.strip()
    if not name:
        return ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name):
        return ""
    return name

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def build_nerfreal(sessionid:int, voice_override: Optional[str] = None)->BaseReal:
    if voice_override and voice_override != getattr(opt, "REF_FILE", None):
        working_opt = copy.deepcopy(opt)
        working_opt.REF_FILE = voice_override
    else:
        working_opt = opt

    working_opt.sessionid = sessionid

    if working_opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(working_opt, model, avatar)
    elif working_opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(working_opt, model, avatar)
    # elif working_opt.model == 'ernerf':
    #     from nerfreal import NeRFReal
    #     nerfreal = NeRFReal(working_opt,model,avatar)
    elif working_opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(working_opt, model, avatar)
    else:
        raise ValueError(f"不支持的模型类型: {working_opt.model}")
    return nerfreal

#@app.route('/offer', methods=['POST'])
async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # if len(nerfreals) >= opt.max_session:
    #     logger.info('reach max session')
    #     return web.Response(
    #         content_type="application/json",
    #         text=json.dumps(
    #             {"code": -1, "msg": "reach max session"}
    #         ),
    #     )
    sessionid = randN(6) #len(nerfreals)
    nerfreals[sessionid] = None
    logger.info('sessionid=%d, session num=%d',sessionid,len(nerfreals))
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal
    
    #ice_server = RTCIceServer(urls='stun:stun.l.google.com:19302')
    ice_server = RTCIceServer(urls='stun:stun.miwifi.com:3478')
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[ice_server]))
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
        if pc.connectionState == "closed":
            pcs.discard(pc)
            del nerfreals[sessionid]
            gc.collect()

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    #return jsonify({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

async def human(request):
    try:
        params = await request.json()

        try:
            sessionid = int(params.get('sessionid', 0))
        except (TypeError, ValueError):
            raise ValueError("非法的会话编号")

        nerfreal = nerfreals.get(sessionid)
        if nerfreal is None:
            raise ValueError("会话不存在，请重新建立连接")

        text_raw = params.get('text', '')
        text = text_raw.strip()

        if params.get('interrupt'):
            nerfreal.flush_talk()

        msg_type = params.get('type')
        if msg_type == 'echo':
            if not text:
                raise ValueError("播报内容不能为空")
            nerfreal.put_msg_txt(text_raw)
        elif msg_type == 'chat':
            if not text:
                raise ValueError("消息不能为空")
            nerfreal.add_chat_message('user', text, {"source": "chat"})
            asyncio.get_event_loop().run_in_executor(None, llm_response, text_raw, nerfreal)
        else:
            raise ValueError("未知的请求类型")

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )


async def chat_history(request):
    try:
        sessionid_raw = request.query.get('sessionid', '0')
        after_raw = request.query.get('after', '0')
        try:
            sessionid = int(sessionid_raw)
        except (TypeError, ValueError):
            raise ValueError("非法的会话编号")
        try:
            after_id = int(after_raw)
        except (TypeError, ValueError):
            after_id = 0

        nerfreal = nerfreals.get(sessionid)
        if nerfreal is None:
            raise ValueError("会话不存在")

        messages = nerfreal.get_chat_messages(after_id)
        return web.Response(
            content_type="application/json",
            text=json.dumps({"code": 0, "messages": messages}, ensure_ascii=False),
        )
    except Exception as exc:
        logger.warning("chat history failed: %s", exc)
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(exc)}, ensure_ascii=False),
        )


async def interrupt_talk(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        nerfreals[sessionid].flush_talk()
        
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def humanaudio(request):
    try:
        form= await request.post()
        sessionid = int(form.get('sessionid',0))
        fileobj = form["file"]
        filename=fileobj.filename
        filebytes=fileobj.file.read()
        nerfreals[sessionid].put_audio_file(filebytes)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def set_audiotype(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)    
        nerfreals[sessionid].set_custom_state(params['audiotype'],params['reinit'])

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def record(request):
    try:
        params = await request.json()

        sessionid = params.get('sessionid',0)
        if params['type']=='start_record':
            # nerfreals[sessionid].put_msg_txt(params['text'])
            nerfreals[sessionid].start_recording()
        elif params['type']=='end_record':
            nerfreals[sessionid].stop_recording()
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg":"ok"}
            ),
        )
    except Exception as e:
        logger.exception('exception:')
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": str(e)}
            ),
        )

async def is_speaking(request):
    params = await request.json()

    sessionid = params.get('sessionid',0)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": nerfreals[sessionid].is_speaking()}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def schedule_restart_with_avatar(avatar_id: str) -> None:
    await asyncio.sleep(2.0)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "app.py"),
        "--transport",
        "webrtc",
        "--model",
        "wav2lip",
        "--avatar_id",
        avatar_id,
        "--customvideo_config",
        "data/custom_config.json",
    ]
    logger.info("[avatar] restarting application with avatar_id=%s", avatar_id)
    os.execv(cmd[0], cmd)

async def post(url,data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url,data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        logger.info(f'Error: {e}')


async def avatar_generate(request):
    lock = _get_avatar_generation_lock()
    async with lock:
        try:
            form = await request.post()
        except Exception as exc:
            logger.exception("avatar generate form error")
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"请求解析失败: {exc}"}, ensure_ascii=False),
            )

        avatar_name = _normalize_avatar_name(form.get("avatarName", ""))
        if not avatar_name:
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "头像标识仅支持字母、数字、下划线和短横线，长度 1-32。"}, ensure_ascii=False),
            )

        video_field = form.get("video")
        static_video_field = form.get("video_static")
        if video_field is None:
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "请上传用于生成数字人的视频文件"}, ensure_ascii=False),
            )
        if static_video_field is None:
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "请上传用于提取画面与音频的静态视频"}, ensure_ascii=False),
            )
        if FFMPEG_BIN is None:
            logger.error("[avatar] ffmpeg executable not found in PATH")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "未检测到 ffmpeg，请先安装并添加到系统 PATH。"}, ensure_ascii=False),
            )

        tmp_dir = Path(tempfile.mkdtemp(prefix="avatar_upload_"))
        video_path = tmp_dir / (video_field.filename or "avatar.mp4")
        static_video_path = tmp_dir / (static_video_field.filename or "avatar_static.mp4")
        try:
            with open(video_path, "wb") as f:
                f.write(video_field.file.read())
            with open(static_video_path, "wb") as f:
                f.write(static_video_field.file.read())

            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "wav2lip" / "genavatar.py"),
                "--video_path",
                str(video_path),
                "--img_size",
                "256",
                "--avatar_id",
                avatar_name,
                "--face_det_batch_size",
                "6",
                "--auto_crop_borders",
                "--stretch_to_fill",
            ]
            logger.info("[avatar] running %s", " ".join(cmd))

            loop = asyncio.get_running_loop()

            def _run_genavatar():
                process = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                collected: list[str] = []
                assert process.stdout is not None
                try:
                    for line in process.stdout:
                        if not line:
                            continue
                        stripped = line.rstrip()
                        collected.append(stripped)
                        logger.info("[avatar][genavatar] %s", stripped)
                finally:
                    process.stdout.close()
                returncode = process.wait()
                return returncode, "\n".join(collected)

            def _run_cmd(cmd_list):
                return subprocess.run(cmd_list, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

            returncode, stdout_log = await loop.run_in_executor(None, _run_genavatar)

            if returncode != 0:
                failure_line = stdout_log.splitlines()[-1] if stdout_log else "未知错误"
                logger.error("[avatar] genavatar failed, return code %s: %s", returncode, failure_line)
                return web.Response(
                    status=500,
                    content_type="application/json",
                    text=json.dumps({"code": -1, "msg": f"数字人生成失败: {failure_line}"}, ensure_ascii=False),
                )
            if stdout_log:
                logger.info("[avatar] genavatar output summary: %s", stdout_log.splitlines()[-1])

            results_dir = AVATAR_RESULTS_ROOT / avatar_name
            if not results_dir.exists():
                fallback_dir = AVATAR_RESULTS_ROOT_FALLBACK / avatar_name
                if fallback_dir.exists():
                    results_dir = fallback_dir

            if not results_dir.exists():
                logger.error("[avatar] expected results %s not found", results_dir)
                return web.Response(
                    status=500,
                    content_type="application/json",
                    text=json.dumps({"code": -1, "msg": "未找到生成结果，请检查输入视频是否有效。"}, ensure_ascii=False),
                )

            avatar_dir = CUSTOM_AVATAR_BASE_ROOT / avatar_name
            images_dir = avatar_dir / "image"
            audio_path = avatar_dir / "audio.wav"
            if avatar_dir.exists():
                shutil.rmtree(avatar_dir)
            images_dir.mkdir(parents=True, exist_ok=True)

            def _cleanup_assets():
                shutil.rmtree(avatar_dir, ignore_errors=True)

            frame_pattern = images_dir / "%08d.png"
            frame_output = f"./{frame_pattern.relative_to(PROJECT_ROOT).as_posix()}"
            frame_cmd = [
                FFMPEG_BIN,
                "-i",
                str(static_video_path),
                "-vf",
                "fps=25",
                "-qmin",
                "1",
                "-q:v",
                "1",
                "-start_number",
                "0",
                frame_output,
            ]
            logger.info("[avatar] extracting frames: %s", " ".join(frame_cmd))

            frame_proc = await loop.run_in_executor(None, lambda: _run_cmd(frame_cmd))
            if frame_proc.returncode != 0:
                logger.error("[avatar] extract frames failed: %s", frame_proc.stderr.strip())
                _cleanup_assets()
                return web.Response(
                    status=500,
                    content_type="application/json",
                    text=json.dumps({"code": -1, "msg": f"静态视频抽帧失败: {frame_proc.stderr.strip() or '未知错误'}"}, ensure_ascii=False),
                )

            cleaned_frames, total_frames = _cleanup_idle_frames(images_dir)
            if cleaned_frames:
                logger.info(
                    "[avatar] cleaned idle frames borders for %s (%d/%d frames)",
                    avatar_name,
                    cleaned_frames,
                    total_frames,
                )
            elif total_frames:
                logger.info(
                    "[avatar] idle frames already clean for %s (%d frames)",
                    avatar_name,
                    total_frames,
                )

            audio_output = f"./{audio_path.relative_to(PROJECT_ROOT).as_posix()}"
            audio_cmd = [
                FFMPEG_BIN,
                "-i",
                str(static_video_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ac",
                "1",
                "-ar",
                "16000",
                audio_output,
            ]
            logger.info("[avatar] extracting audio: %s", " ".join(audio_cmd))

            audio_proc = await loop.run_in_executor(None, lambda: _run_cmd(audio_cmd))
            if audio_proc.returncode != 0:
                logger.error("[avatar] extract audio failed: %s", audio_proc.stderr.strip())
                _cleanup_assets()
                return web.Response(
                    status=500,
                    content_type="application/json",
                    text=json.dumps({"code": -1, "msg": f"静态视频提取音频失败: {audio_proc.stderr.strip() or '未知错误'}"}, ensure_ascii=False),
                )

            relative_img_path = f"data/customvideo/{avatar_name}/image"
            relative_audio_path = f"data/customvideo/{avatar_name}/audio.wav"

            config_payload, config_entry, config_existed, original_config_text = _load_custom_config_entry()
            config_entry.update(
                {
                    "audiotype": config_entry.get("audiotype", 2),
                    "imgpath": relative_img_path,
                    "audiopath": relative_audio_path,
                }
            )

            success, error = _persist_custom_config(
                config_payload,
                config_existed,
                original_config_text,
            )
            if not success:
                _cleanup_assets()
                return web.Response(
                    status=500,
                    content_type="application/json",
                    text=json.dumps({"code": -1, "msg": f"更新 custom_config 失败: {error}"}, ensure_ascii=False),
                )

            target_dir = AVATAR_TARGET_ROOT / avatar_name
            try:
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(results_dir), str(target_dir))
            except Exception as exc:
                logger.exception("[avatar] move files failed")
                _cleanup_assets()
                if config_existed and original_config_text is not None:
                    try:
                        with open(CUSTOM_CONFIG_PATH, "w", encoding="utf-8") as cfg_file:
                            cfg_file.write(original_config_text)
                    except Exception as restore_exc:
                        logger.warning("[avatar] restore custom config failed: %s", restore_exc)
                elif not config_existed:
                    try:
                        CUSTOM_CONFIG_PATH.unlink(missing_ok=True)
                    except Exception as restore_exc:
                        logger.warning("[avatar] remove custom config failed: %s", restore_exc)
                return web.Response(
                    status=500,
                    content_type="application/json",
                    text=json.dumps({"code": -1, "msg": f"拷贝生成资源失败: {exc}"}, ensure_ascii=False),
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    asyncio.create_task(schedule_restart_with_avatar(avatar_name))

    payload = {
        "code": 0,
        "avatarId": avatar_name,
        "msg": "数字人生成成功，服务即将重启，请稍候页面自动刷新。",
    }
    return web.Response(
        content_type="application/json",
        text=json.dumps(payload, ensure_ascii=False),
    )


async def avatar_list(request):
    avatars: list[dict[str, str]] = []
    if AVATAR_TARGET_ROOT.is_dir():
        for avatar_dir in sorted(AVATAR_TARGET_ROOT.iterdir()):
            if not avatar_dir.is_dir():
                continue
            thumbnail = _find_first_thumbnail(avatar_dir)
            if thumbnail is None:
                continue
            rel_path = thumbnail.relative_to(PROJECT_ROOT).as_posix()
            resolution = _probe_avatar_resolution(avatar_dir.name)
            avatar_entry: dict[str, object] = {
                "name": avatar_dir.name,
                "thumbnail": f"/{rel_path}",
            }
            if resolution:
                avatar_entry["videoWidth"], avatar_entry["videoHeight"] = resolution
            avatars.append(avatar_entry)

    payload = {"code": 0, "avatars": avatars}
    return web.Response(
        content_type="application/json",
        text=json.dumps(payload, ensure_ascii=False),
    )


async def avatar_current(request):
    candidate_names: list[str] = []

    try:
        _payload, config_entry, _, _ = _load_custom_config_entry()
    except Exception as exc:  # pragma: no cover - defensive, should not trigger in normal flow
        logger.warning("[avatar] load custom config for current avatar failed: %s", exc)
        config_entry = {}

    img_path = None
    if isinstance(config_entry, dict):
        img_path = config_entry.get("imgpath") or config_entry.get("imgPath")

    if isinstance(img_path, str) and img_path:
        normalized = img_path.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if len(parts) >= 2:
            candidate_names.append(parts[-2])

    avatar_id = getattr(opt, "avatar_id", None)
    if avatar_id:
        candidate_names.append(avatar_id)

    candidate_names.append(DEFAULT_AVATAR_NAME)

    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for name in candidate_names:
        if name and name not in seen:
            seen.add(name)
            ordered_candidates.append(name)

    avatar_name: str | None = None
    thumbnail_url: str | None = None
    for name in ordered_candidates:
        avatar_dir = AVATAR_TARGET_ROOT / name
        if not avatar_dir.is_dir():
            continue
        avatar_name = name
        thumbnail_path = _find_first_thumbnail(avatar_dir)
        if thumbnail_path:
            thumbnail_url = f"/{thumbnail_path.relative_to(PROJECT_ROOT).as_posix()}"
        break

    video_width: int | None = None
    video_height: int | None = None
    if avatar_name:
        resolution = _probe_avatar_resolution(avatar_name)
        if resolution:
            video_width, video_height = resolution

    payload = {
        "code": 0,
        "avatarName": avatar_name,
        "thumbnail": thumbnail_url,
        "videoWidth": video_width,
        "videoHeight": video_height,
    }
    return web.Response(
        content_type="application/json",
        text=json.dumps(payload, ensure_ascii=False),
    )


async def avatar_select(request):
    lock = _get_avatar_generation_lock()
    async with lock:
        try:
            data = await request.json()
        except Exception as exc:
            logger.exception("avatar select json error")
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"请求解析失败: {exc}"}, ensure_ascii=False),
            )

        avatar_name = _normalize_avatar_name(data.get("avatarName", ""))
        if not avatar_name:
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "请选择有效的数字人"}, ensure_ascii=False),
            )

        source_dir = AVATAR_TARGET_ROOT / avatar_name
        if not source_dir.is_dir():
            return web.Response(
                status=404,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "未找到对应的数字人资源"}, ensure_ascii=False),
            )

        avatar_dir = CUSTOM_AVATAR_BASE_ROOT / avatar_name
        images_dir = avatar_dir / "image"
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.exception("[avatar] ensure customvideo directory failed")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"创建自定义资源目录失败: {exc}"}, ensure_ascii=False),
            )

        cleaned_frames, total_frames = _cleanup_idle_frames(images_dir)
        if cleaned_frames:
            logger.info(
                "[avatar] refreshed idle frames for %s (%d/%d frames cleaned)",
                avatar_name,
                cleaned_frames,
                total_frames,
            )

        relative_img_path = f"data/customvideo/{avatar_name}/image"
        relative_audio_path = f"data/customvideo/{avatar_name}/audio.wav"

        config_payload, config_entry, config_existed, original_config_text = _load_custom_config_entry()
        config_entry.update(
            {
                "audiotype": config_entry.get("audiotype", 2),
                "imgpath": relative_img_path,
                "audiopath": relative_audio_path,
            }
        )

        success, error = _persist_custom_config(
            config_payload,
            config_existed,
            original_config_text,
        )
        if not success:
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": f"更新 custom_config 失败: {error}"}, ensure_ascii=False),
            )

    asyncio.create_task(schedule_restart_with_avatar(avatar_name))

    payload = {
        "code": 0,
        "avatarId": avatar_name,
        "msg": f"已切换至数字人 {avatar_name}，服务即将重启，请稍候页面自动刷新。",
    }
    return web.Response(
        content_type="application/json",
        text=json.dumps(payload, ensure_ascii=False),
    )


async def ppt_preview(request):
    logger.info("[ppt-preview] incoming request")
    try:
        form = await request.post()
    except Exception as exc:
        logger.exception("ppt preview form error")
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": f"请求解析失败: {exc}"}, ensure_ascii=False),
        )

    ppt_field = form.get("ppt")
    if ppt_field is None:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "请上传 PPT 文件"}, ensure_ascii=False),
        )

    limit = 0
    limit_raw = form.get("limit")
    if limit_raw:
        try:
            if isinstance(limit_raw, str) and limit_raw.strip().lower() == "all":
                limit = 0
            else:
                parsed_limit = int(limit_raw)
                limit = 0 if parsed_limit <= 0 else min(parsed_limit, 200)
        except ValueError:
            pass

    logger.info(
        "[ppt-preview] processing file=%s limit=%d",
        getattr(ppt_field, "filename", "<unknown>"),
        limit,
    )

    try:
        ppt_bytes = ppt_field.file.read()
    except Exception as exc:
        logger.exception("ppt preview read file error")
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": f"读取文件失败: {exc}"}, ensure_ascii=False),
        )

    loop = asyncio.get_event_loop()
    try:
        images, dims_in = await loop.run_in_executor(
            None,
            lambda: generate_slide_previews(ppt_bytes, limit),
        )
    except Exception as exc:
        logger.exception("ppt preview fail")
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(exc)}, ensure_ascii=False),
        )

    logger.info(
        "[ppt-preview] success: %d images, width=%.2f\" height=%.2f\"",
        len(images),
        dims_in[0],
        dims_in[1],
    )

    response_payload = {
        "code": 0,
        "images": [f"data:image/png;base64,{img}" for img in images],
        "slideWidthIn": dims_in[0],
        "slideHeightIn": dims_in[1],
    }
    return web.Response(
        content_type="application/json",
        text=json.dumps(response_payload, ensure_ascii=False),
    )


async def ppt_augment(request):
    global ppt_generator
    if ppt_generator is None:
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "PPT 处理模块尚未初始化"}, ensure_ascii=False),
        )

    try:
        form = await request.post()
    except Exception as exc:
        logger.exception("ppt augment form error")
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": f"请求解析失败: {exc}"}, ensure_ascii=False),
        )

    ppt_field = form.get('ppt')
    script_field = form.get('script')
    if ppt_field is None or script_field is None:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "请同时上传 PPT 和脚本文档"}, ensure_ascii=False),
        )

    positions_field = form.get("positions")
    try:
        positions = _parse_positions_field(positions_field)
    except ValueError as exc:
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(exc)}, ensure_ascii=False),
        )

    try:
        ppt_bytes = ppt_field.file.read()
        script_bytes = script_field.file.read()
    except Exception as exc:
        logger.exception("ppt augment read file error")
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": f"读取上传文件失败: {exc}"}, ensure_ascii=False),
        )

    ppt_filename = ppt_field.filename or "presentation.pptx"
    base_name = os.path.splitext(os.path.basename(ppt_filename))[0]
    output_name = f"{base_name}-livetalking-augmented.pptx"

    voice_field = form.get("voice")
    voice_choice: Optional[str] = None
    if isinstance(voice_field, str):
        trimmed = voice_field.strip()
        if trimmed:
            voice_choice = trimmed[:128]

    loop = asyncio.get_event_loop()
    try:
        result_bytes, meta, _ = await loop.run_in_executor(
            None,
            lambda: ppt_generator.generate(
                ppt_bytes,
                script_bytes,
                output_name,
                positions,
                voice_id=voice_choice,
            ),
        )
    except Exception as exc:
        logger.exception('ppt augment fail')
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(exc)}, ensure_ascii=False),
        )
    filename = meta.get('filename', output_name)
    disposition = f"attachment; filename*=UTF-8''{quote(filename)}"
    session_id = meta.get('session_id') if isinstance(meta, dict) else None

    return web.Response(
        body=result_bytes,
        headers={
            'Content-Type': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'Content-Disposition': disposition,
            **({'X-LiveTalking-Session': session_id} if session_id else {}),
        },
    )


async def ppt_course(request):
    global ppt_generator
    if ppt_generator is None:
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "PPT 处理模块尚未初始化"}, ensure_ascii=False),
        )

    try:
        payload = await request.json()
    except Exception as exc:
        logger.exception("ppt course payload error")
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": f"请求解析失败: {exc}"}, ensure_ascii=False),
        )

    session_id = payload.get("sessionId") if isinstance(payload, dict) else None
    if not session_id or not isinstance(session_id, str):
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "缺少有效的 sessionId"}, ensure_ascii=False),
        )

    course_filename = payload.get("filename") if isinstance(payload, dict) else None
    if isinstance(course_filename, str):
        course_filename = course_filename.strip() or None
    else:
        course_filename = None

    static_duration_raw = payload.get("staticDuration") if isinstance(payload, dict) else None
    static_duration: float | None = None
    if static_duration_raw is not None:
        try:
            static_duration = float(static_duration_raw)
        except (TypeError, ValueError):
            return web.Response(
                status=400,
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "静态停留时长必须是数字"}, ensure_ascii=False),
            )

    loop = asyncio.get_event_loop()
    try:
        course_bytes, course_meta = await loop.run_in_executor(
            None,
            lambda: ppt_generator.compose_course_from_session(session_id, course_filename, static_duration),
        )
    except KeyError as exc:
        logger.warning('ppt course session missing: %s', exc)
        return web.Response(
            status=404,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "未找到对应的生成任务，请重新生成 PPT"}, ensure_ascii=False),
        )
    except Exception as exc:
        logger.exception('ppt course fail')
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": str(exc)}, ensure_ascii=False),
        )
    final_filename = course_meta.get("filename") if isinstance(course_meta, dict) else None
    if not final_filename:
        final_filename = f"{session_id}-livetalking-course.mp4"

    disposition = f"attachment; filename*=UTF-8''{quote(final_filename)}"

    course_meta_header = None
    if isinstance(course_meta, dict):
        meta_payload = course_meta.copy()
        resolution = meta_payload.get("resolution")
        if isinstance(resolution, tuple):
            meta_payload["resolution"] = list(resolution)
        course_meta_header = json.dumps(meta_payload, ensure_ascii=True)

    return web.Response(
        body=course_bytes,
        headers={
            'Content-Type': 'video/mp4',
            'Content-Disposition': disposition,
            **({'X-LiveTalking-CourseMeta': course_meta_header} if course_meta_header else {}),
        },
    )

async def _generate_edge_preview_audio(voice_id: str, text: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice_id)
    audio_payload = bytearray()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            data = chunk.get("data")
            if data:
                audio_payload.extend(data)
    return bytes(audio_payload)


async def tts_preview(request: web.Request) -> web.Response:
    if opt is None:
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "系统初始化尚未完成"}, ensure_ascii=False),
        )

    if getattr(opt, "tts", "") != "edgetts":
        return web.Response(
            status=400,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "当前 TTS 配置不支持语音试听"}, ensure_ascii=False),
        )

    voice_id = _sanitize_voice_id(request.query.get("voice"))
    if not voice_id:
        voice_id = _sanitize_voice_id(getattr(opt, "REF_FILE", "")) or EDGE_TTS_PREVIEW_VOICE_FALLBACK

    preview_text = request.query.get("text")
    if isinstance(preview_text, str):
        preview_text = preview_text.strip().replace("\r", " ").replace("\n", " ")
    else:
        preview_text = EDGE_TTS_PREVIEW_TEXT

    if not preview_text:
        preview_text = EDGE_TTS_PREVIEW_TEXT

    if len(preview_text) > EDGE_TTS_PREVIEW_MAX_TEXT_LENGTH:
        preview_text = preview_text[:EDGE_TTS_PREVIEW_MAX_TEXT_LENGTH]

    try:
        audio_bytes = await _generate_edge_preview_audio(voice_id, preview_text)
    except Exception:
        logger.exception("edge tts preview failed: voice=%s", voice_id)
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "语音预览生成失败，请稍后重试。"}, ensure_ascii=False),
        )

    if not audio_bytes:
        return web.Response(
            status=500,
            content_type="application/json",
            text=json.dumps({"code": -1, "msg": "未获取到语音音频数据"}, ensure_ascii=False),
        )

    return web.Response(
        body=audio_bytes,
        headers={
            "Content-Type": "audio/mpeg",
            "Cache-Control": "no-store",
        },
    )


async def run(push_url,sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url,pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer,type='answer'))
##########################################
# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    
    # audio FPS
    parser.add_argument('--fps', type=int, default=50, help="audio fps,must be 50")
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")

    #musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1', help="define which avatar in data/avatars")
    #parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")

    parser.add_argument('--customvideo_config', type=str, default='', help="custom action json")

    parser.add_argument('--tts', type=str, default='edgetts', help="tts service type") #xtts gpt-sovits cosyvoice fishtts tencent doubao indextts2 azuretts
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",help="参考文件名或语音模型ID，默认值为 edgetts的语音模型ID zh-CN-YunxiaNeural, 若--tts指定为azuretts, 可以使用Azure语音模型ID, 如zh-CN-XiaoxiaoMultilingualNeural")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880') # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='musetalk') #musetalk wav2lip ultralight

    parser.add_argument('--transport', type=str, default='rtcpush') #webrtc rtcpush virtualcam
    parser.add_argument('--push_url', type=str, default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream') #rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  #multi session count
    parser.add_argument('--listenport', type=int, default=8010, help="web listen port")

    opt = parser.parse_args()
    #app.config.from_object(opt)
    #print(app.config)
    opt.customopt = []
    if opt.customvideo_config!='':
        with open(opt.customvideo_config,'r') as file:
            opt.customopt = json.load(file)

    # if opt.model == 'ernerf':       
    #     from nerfreal import NeRFReal,load_model,load_avatar
    #     model = load_model(opt)
    #     avatar = load_avatar(opt) 
    if opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model()
        avatar = load_avatar(opt.avatar_id) 
        warm_up(opt.batch_size,model)      
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,256)
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        logger.info(opt)
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)

    ppt_generator = PPTDigitalHumanAugmenter(lambda: randN(6), build_nerfreal)

    # if opt.transport=='rtmp':
    #     thread_quit = Event()
    #     nerfreals[0] = build_nerfreal(0)
    #     rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
    #     rendthrd.start()
    if opt.transport=='virtualcam':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/avatar/generate", avatar_generate)
    appasync.router.add_get("/avatar/list", avatar_list)
    appasync.router.add_get("/avatar/current", avatar_current)
    appasync.router.add_post("/avatar/select", avatar_select)
    appasync.router.add_post("/ppt/preview", ppt_preview)
    appasync.router.add_post("/ppt/augment", ppt_augment)
    appasync.router.add_post("/ppt/course", ppt_course)
    appasync.router.add_get("/tts/preview", tts_preview)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/interrupt_talk", interrupt_talk)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_get("/chat/history", chat_history)
    appasync.router.add_static('/data', path=str(PROJECT_ROOT / "data"))
    appasync.router.add_static('/',path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='webrtcapi.html'
    if opt.transport=='rtmp':
        pagename='echoapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        # 自动打开ppt_augment网页
        try:
            webbrowser.open_new_tab(f"http://127.0.0.1:{opt.listenport}/pptaugment.html")
        except Exception as exc:
            logger.warning("auto-open browser failed: %s", exc)
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(run(push_url,k))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()
    
    
