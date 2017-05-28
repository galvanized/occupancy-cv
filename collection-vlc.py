import pyxhook
import time
import sys
import subprocess
import os

"""
collection-vlc.py

Gathers webcam images from /dev/video1 and sorts them into "here" and "away" folders.
Written for a computer vision exercise.

Requires Python 3.2+.

protip: use with devilspie to automatically minimize

TODO:
- mouse filtering (require mouse to move for a certain duration before triggering update)

"""

save_path = 'data/'
command = "exec cvlc -I dummy v4l2:///dev/video1 --video-filter scene --no-audio --scene-path " + save_path + "{} --scene-prefix {}- --scene-format png --run-time=1"

# vlc is started every n seconds
here_photo_delay = 60*5
away_photo_delay = 60*5

run_time = 2 # how long vlc runs (2s is long enough for 1 photo)

# state activation times
here_delay = 2 # 'here' for these many seconds after an activity
away_delay = 60*15 # 'away' after these many seconds after an activity





photo_delay = away_photo_delay
last_state = "ambiguous"
last_press_time = 0
last_photo_time = 0
process = None

def update_press_time(a):
	global last_press_time
	last_press_time = time.time()


def consider_shot():
	global last_state, last_photo_time, photo_delay
	global process
	if last_press_time == 0:
		# we can't safely assume anything about state
		return

	press_delta = time.time() - last_press_time

	if press_delta > away_delay:
		state = "away"
		photo_delay = away_photo_delay
	elif press_delta > here_delay:
		state = "ambiguous"
	else:
		state = "here"
		photo_delay = here_photo_delay

	if process and last_state != state:
		process.kill()
		time.sleep(1)
		process = None

	last_state = state

	if state != "ambiguous":

		photo_delta = time.time() - last_photo_time

		if photo_delta > photo_delay:
			process = subprocess.Popen([command.format(state, round(time.time()))], shell=True)
			last_photo_time = time.time()
		elif process and photo_delta > run_time:
			process.kill()
			time.sleep(1)
			process = None


hook = pyxhook.HookManager()
hook.KeyDown = update_press_time
hook.MouseMovement = update_press_time
hook.HookKeyboard()
hook.HookMouse()
hook.start()

os.makedirs(save_path+'here', exist_ok=True)
os.makedirs(save_path+'away', exist_ok=True)

try:
	while 1:
		time.sleep(0.1)
		consider_shot()
except:
	hook.cancel()
