from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch
import pickle
from typing import Tuple
import face_detection


parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
parser.add_argument('--img_size', default=96, type=int)
parser.add_argument('--avatar_id', default='wav2lip_avatar1', type=str)
parser.add_argument('--video_path', default='', type=str)
parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')
parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=16)
parser.add_argument('--auto_crop_borders', action='store_true',
					help='Automatically detect and remove dark borders before training')
parser.add_argument('--stretch_to_fill', action='store_true',
					help='After cropping borders, resize back to the original frame size to fill the view')
parser.add_argument('--crop_threshold', type=int, default=12,
					help='Intensity threshold (0-255) to detect non-black pixels when auto-cropping borders')
parser.add_argument('--crop_min_ratio', type=float, default=0.02,
					help='Minimum proportion of non-black pixels required per row/column to retain it when cropping')
parser.add_argument('--crop_margin', type=int, default=4,
					help='Extra pixel margin to keep around detected content when cropping borders')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _tighten_crop_edges(
	frame: np.ndarray,
	threshold: int,
	min_nonblack_ratio: float,
) -> Tuple[np.ndarray, bool]:
	"""Iteratively trim faint borders that slip through the initial crop."""
	trimmed = False
	working = frame
	for _ in range(2):
		h, w = working.shape[:2]
		if h < 2 or w < 2:
			break
		gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
		adaptive_thr = max(threshold, int(gray.mean() * 0.3))
		valid = gray > adaptive_thr
		row_ratio = valid.sum(axis=1) / float(w)
		col_ratio = valid.sum(axis=0) / float(h)
		top = 0
		while top < h - 1 and row_ratio[top] <= min_nonblack_ratio:
			top += 1
		bottom = h - 1
		while bottom > top and row_ratio[bottom] <= min_nonblack_ratio:
			bottom -= 1
		left = 0
		while left < w - 1 and col_ratio[left] <= min_nonblack_ratio:
			left += 1
		right = w - 1
		while right > left and col_ratio[right] <= min_nonblack_ratio:
			right -= 1
		if top == 0 and left == 0 and bottom == h - 1 and right == w - 1:
			break
		working = working[top:bottom + 1, left:right + 1]
		trimmed = True
	return working, trimmed


def _remove_black_borders(
	frame: np.ndarray,
	threshold: int = 12,
	min_nonblack_ratio: float = 0.02,
	margin: int = 4,
	stretch_to_fill: bool = False,
) -> Tuple[np.ndarray, bool]:
	orig_h, orig_w = frame.shape[:2]
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# safeguard against extremely dark scenes by adapting threshold to overall brightness
	adaptive_thr = max(threshold, int(gray.mean() * 0.3))
	valid = gray > adaptive_thr
	row_ratio = valid.sum(axis=1) / float(orig_w)
	col_ratio = valid.sum(axis=0) / float(orig_h)
	row_mask = row_ratio > min_nonblack_ratio
	col_mask = col_ratio > min_nonblack_ratio
	if not np.any(row_mask) or not np.any(col_mask):
		return frame, False
	top = int(np.argmax(row_mask))
	bottom = int(len(row_mask) - np.argmax(row_mask[::-1]) - 1)
	left = int(np.argmax(col_mask))
	right = int(len(col_mask) - np.argmax(col_mask[::-1]) - 1)
	top = max(0, top - margin)
	bottom = min(orig_h - 1, bottom + margin)
	left = max(0, left - margin)
	right = min(orig_w - 1, right + margin)
	if top <= 0 and left <= 0 and bottom >= orig_h - 1 and right >= orig_w - 1:
		return frame, False
	new_h = bottom - top + 1
	new_w = right - left + 1
	# Avoid over-cropping when the detected area is too small
	if new_h < int(orig_h * 0.5) or new_w < int(orig_w * 0.5):
		return frame, False
	cropped = frame[top:bottom + 1, left:right + 1]
	# tighten residual thin borders introduced by the margin
	cropped, tightened = _tighten_crop_edges(cropped, threshold, min_nonblack_ratio)
	if tightened and (cropped.shape[0] < int(orig_h * 0.45) or cropped.shape[1] < int(orig_w * 0.45)):
		# if trimming became too aggressive, revert to the previous crop
		cropped = frame[top:bottom + 1, left:right + 1]
	if stretch_to_fill and (cropped.shape[0] != orig_h or cropped.shape[1] != orig_w):
		cropped = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
	return cropped, True


def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    cropped_once = False
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            if args.auto_crop_borders:
                frame, cropped = _remove_black_borders(
                    frame,
                    threshold=args.crop_threshold,
                    min_nonblack_ratio=max(0.0, min(0.5, args.crop_min_ratio)),
                    margin=max(0, args.crop_margin),
                    stretch_to_fill=args.stretch_to_fill,
                )
                if cropped and not cropped_once:
                    print(f"Auto-cropped black borders: frame resized to {frame.shape[1]}x{frame.shape[0]}")
                    cropped_once = True
            cv2.putText(frame, "LiveTalking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
											flip_input=False, device=device)

	batch_size = args.face_det_batch_size
	
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

if __name__ == "__main__":
    avatar_path = f"./results/avatars/{args.avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    osmakedirs([avatar_path,full_imgs_path,face_imgs_path])
    print(args)

    #if os.path.isfile(args.video_path):
    video2imgs(args.video_path, full_imgs_path, ext = 'png')
    input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))

    frames = read_imgs(input_img_list)
    face_det_results = face_detect(frames) 
    coord_list = []
    idx = 0
    for frame,coords in face_det_results:        
        #x1, y1, x2, y2 = bbox
        resized_crop_frame = cv2.resize(frame,(args.img_size, args.img_size)) #,interpolation = cv2.INTER_LANCZOS4)
        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
        coord_list.append(coords)
        idx = idx + 1
	
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)
