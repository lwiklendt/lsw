import numpy as np
import subprocess as sp
import re
import os


def get_shape(filename):
    """
    :param filename:
    :return: w, h, num_frames, fps
    """

    # get info
    proc = sp.Popen(['ffmpeg', '-i', filename, '-'], bufsize=-1, stdout=sp.PIPE, stderr=sp.PIPE)
    proc.stdout.readline()
    proc.terminate()
    lines = proc.stderr.read().decode('utf8').splitlines()

    # frame shape and rate (rate is used to calculate number of frames)
    line = [l for l in lines if ' Video: ' in l][0]
    shape = map(int, re.search(r' (\d+)x(\d+)[ ,]', line).groups())
    fps = float(re.search(r'(\d+) tbr', line).groups()[0])

    # number of frames
    line = [l for l in lines if ' Duration: ' in l][0]
    hms = np.array(map(float, re.search(r'(\d{2}):(\d{2}):(\d{2}(?:.\d+))', line).groups()))
    num_frames = int(hms.dot([3600, 60, 1]) * fps)

    return shape[0], shape[1], num_frames, fps


def read(filename):

    w, h, num_frames, fps = get_shape(filename)

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-c:v', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=-1)
    s = pipe.stdout.read(w * h * 3 * num_frames)
    frames = np.fromstring(s, dtype='uint8')

    frames = frames.reshape((num_frames, h, w, 3))

    return frames, fps


def write(filename, frames, fps, codec='msmpeg4v2', br='4000k'):

    num_frames, h, w, num_chan = frames.shape

    print(frames.shape)

    command = ['ffmpeg',
               '-y',  # overwrite existing file
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', '{}x{}'.format(w, h),
               '-pix_fmt', 'rgb24',
               '-r', '{}'.format(fps),
               '-i', '-',
               '-an',  # no audio
               '-vcodec', codec,
               '-pix_fmt', 'yuv420p',
               '-b:v', br,
               filename]

    devnull = open(os.devnull, 'wb')
    proc = sp.Popen(command, stdin=sp.PIPE, stdout=devnull, stderr=devnull)

    proc.stdin.write(frames.tostring())


class VideoWriter(object):
    def __init__(self, filename, w, h, fps, codec='msmpeg4v2', br='4M'):
        self.filename = filename
        self.command = ['ffmpeg',
                        '-y',  # overwrite existing file
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-s', '{}x{}'.format(w, h),
                        '-pix_fmt', 'rgb24',
                        '-r', '{}'.format(fps),
                        '-i', '-',
                        '-an',  # no audio
                        '-vcodec', codec,
                        '-pix_fmt', 'yuv420p',
                        '-b:v', br,
                        filename]
        devnull = open(os.devnull, 'wb')
        self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stdout=devnull, stderr=devnull)

    def write_frames(self, frame):
        self.pipe.stdin.write(frame.tostring())


class VideoReader(object):
    def __init__(self, filename):
        self.filename = filename
        w, h, num_frames, fps = get_shape(filename)
        self.w = w
        self.h = h
        self.num_frames = num_frames
        self.fps = fps
        self.command = None
        self.pipe = None

        command = ['ffmpeg',
                   '-i', self.filename,
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-c:v', 'rawvideo', '-']
        self.command = command
        self.pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=-1)

    def read_frames(self, num_frames=1):
        """
        This doesn't work well for large videos, use cv2 instead.
        :param num_frames:
        :return:
        """
        s = self.pipe.stdout.read(num_frames * self.h * self.w * 3)
        frames = np.fromstring(s, dtype='uint8')
        return frames.reshape((num_frames, self.h, self.w, 3)).squeeze()
