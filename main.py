#!/usr/bin/env python


if __name__ == "__main__":
    from install_update import *
else:
    import os
if os.name == "nt":
    os.system("color")
import numpy, time, psutil, sys, collections, random, contextlib, re, concurrent.futures
suppress = contextlib.suppress
from math import *
if __name__ == "__main__":
    # Requires a thread pool to manage concurrent pipes
    exc = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    from PIL import Image, ImageChops

np = numpy
deque = collections.deque

# Simple function to detect URLs
url_match = re.compile("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+$")
is_url = lambda url: url_match.search(url)

# Simple function to evaluate json inputs
eval_json = lambda s: eval(s, dict(true=True, false=False, null=None, none=None), {})


# Simple dictionary-like hash table that additionally allows class-like indexing
class cdict(dict):

    __slots__ = ()

    __init__ = lambda self, *args, **kwargs: super().__init__(*args, **kwargs)
    __repr__ = lambda self: f"{self.__class__.__name__}({super().__repr__() if super().__len__() else ''})"
    __str__ = lambda self: super().__repr__()
    __iter__ = lambda self: iter(tuple(super().__iter__()))
    __call__ = lambda self, k: self.__getitem__(k)

    def __getattr__(self, k):
        with suppress(AttributeError):
            return self.__getattribute__(k)
        if not k.startswith("__") or not k.endswith("__"):
            try:
                return self.__getitem__(k)
            except KeyError as ex:
                raise AttributeError(*ex.args)
        raise AttributeError(k)

    def __setattr__(self, k, v):
        if k.startswith("__") and k.endswith("__"):
            return object.__setattr__(self, k, v)
        return self.__setitem__(k, v)

    def __dir__(self):
        data = set(object.__dir__(self))
        data.update(self)
        return data

    @property
    def __dict__(self):
        return self

    ___repr__ = lambda self: super().__repr__()
    to_dict = lambda self: dict(**self)
    to_list = lambda self: list(super().values())


# Converts an amount of seconds into a time display {days:hours:minutes:seconds}
def time_disp(s):
    if not isfinite(s):
        return str(s)
    s = round(s)
    output = str(s % 60)
    if len(output) < 2:
        output = "0" + output
    if s >= 60:
        temp = str((s // 60) % 60)
        if len(temp) < 2 and s >= 3600:
            temp = "0" + temp
        output = temp + ":" + output
        if s >= 3600:
            temp = str((s // 3600) % 24)
            if len(temp) < 2 and s >= 86400:
                temp = "0" + temp
            output = temp + ":" + output
            if s >= 86400:
                output = str(s // 86400) + ":" + output
    else:
        output = "0:" + output
    return output


# Maps colours to their respective terminal escape codes
C = COLOURS = cdict(
    black="\u001b[30m",
    red="\u001b[31m",
    green="\u001b[32m",
    yellow="\u001b[33m",
    blue="\u001b[34m",
    magenta="\u001b[35m",
    cyan="\u001b[36m",
    white="\u001b[37m",
    reset="\u001b[0m",
)
# Removes terminal colour codes from text
def nocol(s):
    for i in C.values():
        s = s.replace(i, "")
    return s

# Generates a progress bar of terminal escape codes and various block characters
bar = "∙░▒▓█"
col = [C.red, C.yellow, C.green, C.cyan, C.blue, C.magenta]
def create_progress_bar(ratio, length=32, offset=None):
    # there are 4 possible characters for every position, meaning that the value of a single bar is 4
    high = length * 4
    position = min(high, round(ratio * high))
    items = deque()
    if offset is not None:
        offset = round(offset * len(col))
    for i in range(length):
        new = min(4, position)
        if offset is not None:
            items.append(col[offset % len(col)])
            offset += 1
        items.append(bar[new])
        position -= new
    return "".join(items)


# Default settings for the program
sample_rate = 48000
fps = 60
amplitude = 1 / 8
render = display = particles = play = 0
speed = resolution = 1
screensize = size = (960, 540)


# Main class to manage the program's functionality
class Render:

    def __init__(self, f_in):
        # Cutoff between particle and spectrum display is 1/4 of the screen
        self.cutoff = screensize[0] >> 2
        # Start ffmpeg process to calculate single precision float samples from the audio if required
        opt = "-n"
        if os.path.exists(f2):
            if is_url(source) or os.path.getmtime(f2) < max(os.path.getctime(source), os.path.getmtime(source)):
                opt = "-y"
        args = ["ffmpeg", opt, "-hide_banner", "-loglevel", "error", "-i", f_in, "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", f2]
        print(" ".join(args))
        fut1 = exc.submit(psutil.Popen, args, stderr=subprocess.PIPE)
        # Start ffmpeg process to convert audio to wav if required
        opt = "-n"
        if os.path.exists(f3):
            if is_url(source) or os.path.getmtime(f2) < max(os.path.getctime(source), os.path.getmtime(source)):
                opt = "-y"
        args = ["ffmpeg", opt, "-hide_banner", "-loglevel", "error", "-i", f_in, f3]
        print(" ".join(args))
        proc = psutil.Popen(args, stderr=subprocess.PIPE)
        # Wait for wav file to appear before continuing
        try:
            fl = os.path.getsize(f3)
        except FileNotFoundError:
            fl = 0
        while fl < res_scale << 6:
            if not proc.is_running():
                err = proc.stderr.read().decode("utf-8", "replace")
                if err:
                    ex = RuntimeError(err)
                else:
                    ex = RuntimeError("FFmpeg did not start correctly, or file was too small.")
                raise ex
            time.sleep(0.1)
            try:
                fl = os.path.getsize(f3)
            except FileNotFoundError:
                fl = 0
        if render:
            # Start ffmpeg process to convert output bitmap images and wav audio into a mp4 video
            args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-r", str(fps), "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", "x".join(str(i) for i in screensize), "-i", "-"]
            if play:
                args.extend(("-vn", "-i", f3))
            args.extend(("-c:v", "h264", "-b:v", "4M"))
            if play:
                d = round((screensize[0] - self.cutoff) / speed / fps * 1000)
                args.extend(("-af", f"adelay=delays={d}:all=1"))
            args.append(f4)
            print(" ".join(args))
            fut2 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if display:
            # Start python process running display.py to display the preview
            args = [("python3", "python")[os.name == "nt"], "display.py", *[str(x) for x in screensize]]
            print(" ".join(args))
            fut3 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if particles:
            # Start python process running particles.py to render the particles using amplitude sample data
            args = [("python3", "python")[os.name == "nt"], "particles.py", str(particles), str(self.cutoff), str(screensize[1])]
            print(" ".join(args))
            fut4 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if play:
            # Start ffmpeg and ffplay piping 16 bit int samples to simulate audio being played
            args = ["ffplay", "-loglevel", "error", "-hide_banner", "-nodisp", "-autoexit", "-f", "s16le", "-ar", str(sample_rate), "-ac", "2", "-i", "-"]
            print(" ".join(args))
            fut5 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE)
            args = ["ffmpeg", "-loglevel", "error", "-i", f3, "-f", "s16le", "-ar", str(sample_rate), "-ac", "2", "-"]
            print(" ".join(args))
            fut6 = exc.submit(psutil.Popen, args, stdout=subprocess.PIPE)
        # Buffer to store events waiting on audio playback
        self.player_buffer = None
        self.effects = deque()
        self.glow_buffer = deque()
        # Buffer to store empty input data (in single precision floating point)
        self.emptybuff = np.zeros(res_scale, dtype=np.float32)
        # Initialize main input data to be empty
        self.buffer = self.emptybuff
        # Size of the discrete fourier transform of the input frames
        dfts = (res_scale >> 1) + 1
        # Frequency list of the fast fourier transform algorithm output
        self.fff = np.fft.fftfreq(res_scale, 1 / sample_rate)[:dfts]
        maxfreq = np.max(self.fff)
        # FFT returns the values along a linear scale, we want the display the data as a logarithmic scale (because that's how pitch in music works)
        self.fftrans = np.zeros(dfts, dtype=int)
        for i, x in enumerate(self.fff):
            if x <= 0:
                x = screensize[1] - 1
            else:
                x = round((1 - log(x, maxfreq)) * pi / 2 * (screensize[1] - 1))
            if x > screensize[1] - 1:
                x = screensize[1] - 1
            self.fftrans[i] = x
        # Linearly scale amplitude data (unused)
        self.linear_scale = np.arange(screensize[1], dtype=np.float64) / screensize[1]
        # Initialize hue of output image to a vertical rainbow
        self.hue = Image.fromarray(np.expand_dims((self.linear_scale * 256).astype(np.uint8), 0), mode="L")
        # Initialize saturation of output image to be maximum
        self.sat = Image.new("L", (screensize[1], 1), 255)
        # Initialize value of output image to be maximum
        self.val = self.sat
        # Amplitude scale from config, divided by DFT size
        self.scale = (ascale / dfts)
        self.fut = None
        self.playing = True
        proc = fut1.result()
        # Wait for float sample file to appear before continuing
        try:
            fl = os.path.getsize(f2)
        except FileNotFoundError:
            fl = 0
        while fl < res_scale << 6:
            if not proc.is_running():
                err = proc.stderr.read().decode("utf-8", "replace")
                if err:
                    ex = RuntimeError(err)
                else:
                    ex = RuntimeError("FFmpeg did not start correctly, or file was too small.")
                raise ex
            time.sleep(0.1)
            try:
                fl = os.path.getsize(f2)
            except FileNotFoundError:
                fl = 0
        # Open float sample file to read as input
        self.file = open(f2, "rb")
        # Generate blank (fully black) image to begin
        self.image = np.zeros((screensize[1], screensize[0], 3), dtype=np.uint8)
        # Use matrix transposition to paste horizontal images vertically
        self.trans = np.swapaxes(self.image, 0, 1)
        # Wait for remaining processes to start before continuing
        if render:
            self.rend = fut2.result()
        if display:
            self.disp = fut3.result()
        if particles:
            self.part = fut4.result()
            exc.submit(self.animate)
        if play:
            self.player = (fut5.result(), fut6.result())

    def animate(self):
        # Read one frame of animation from the particle renderer process, store in the buffer for retrieval and rendering when the spectrum lines on the display hit the particle display area
        shape = (self.cutoff, screensize[1], 3)
        size = np.prod(shape)
        while True:
            img = bytes()
            self.playing = False
            while len(img) < size:
                temp = self.part.stdout.read(size - len(img))
                if not temp:
                    break
                img += temp
            self.playing = True
            img = np.frombuffer(img, dtype=np.uint8)
            # Must transpose image data as X and Y coordinates are swapped
            ordered = np.reshape(img, (shape[1], shape[0], shape[2]))
            if ordered.shape[0] != shape[0]:
                if ordered.shape[1] == shape[0]:
                    ordered = np.swapaxes(ordered, 0, 1)
                else:
                    ordered = np.swapaxes(ordered, 0, 2)
            if ordered.shape[1] != shape[1]:
                ordered = np.swapaxes(ordered, 1, 2)
            self.effects.append(ordered)

    def read(self):
        # Calculate required amount of input data to read, converting to 32 bit floats
        req = (res_scale) - len(self.buffer)
        if req > 0:
            data = self.file.read(req << 2)
            self.buffer = np.concatenate((self.buffer, np.frombuffer(data, dtype=np.float32)))
        else:
            data = True
        # Calculate required amount of input data again, in case the file was exhausted, appending empty data if required
        req = (res_scale) - len(self.buffer)
        if req > 0:
            self.buffer = np.concatenate((self.buffer, self.emptybuff[:req]))
        if not particles and not data:
            # If no data was found at all, stop the program as we have reached the end of audio input
            if not (np.min(self.buffer) or np.max(self.buffer)):
                raise StopIteration
        # Calculate real fast fourier transform of input samples
        dft = np.fft.rfft(self.buffer[:res_scale])
        # Advance sample buffer by sample rate divided by output fps
        self.buffer = self.buffer[sample_rate // fps:]
        # Real fft algorithm returns complex numbers as polar coordinate pairs, initialize empty array to store their sums across a log scale
        arr = np.zeros(screensize[1], dtype=np.complex128)
        # This function took me way too long to find lmao, extremely useful here as there may be more than one of the same output index per input position due to the log scale
        np.add.at(arr, self.fftrans, dft)
        # After the addition, we no longer require the phase of the complex numbers as their waves (and thus interference) have been summed, take absolute value of data array
        amp = np.abs(arr, dtype=np.float32)
        # Multiply array by amplitude scale
        amp = np.multiply(np.multiply(amp, self.scale, out=amp), 256, out=amp)
        # Saturation decreases when input is above 255, becoming fully desaturated and maxing out at 511
        sat = np.clip(511 - amp, 0, 255).astype(np.uint8)
        # Value increases as input is above 0, becoming full brightness and maxing out at 255
        val = np.clip(amp, 0, 255).astype(np.uint8)
        # Glow buffer is the brightness of the line separating particles and spectrum lines, it changes as data passes it
        if len(self.glow_buffer) >= (screensize[0] - self.cutoff) / speed:
            self.trans[self.cutoff] = self.glow_buffer.popleft()
        self.glow_buffer.append(min(255, int(sum(amp) / self.scale / 524288) + 127))
        # Write a copy of the resulting spectrum data to particles subprocess if applicable
        if particles and data:
            np.multiply(amp, 1 / 64, out=amp)
            np.clip(amp, 0, 64, out=amp)
            if getattr(self.part, "fut", None):
                try:
                    self.part.fut.result()
                except:
                    print(self.part.stderr.read().decode("utf-8", "replace"))
                    raise
            compat = None
            c = 4
            for i in range(c):
                temp = amp[i::c]
                if compat is None:
                    compat = temp
                else:
                    compat[:len(temp)] += temp
            self.part.fut = exc.submit(self.part.stdin.write, compat.tobytes())
        # Convert saturation and brightness arrays into 2D arrays of length 1, to prepare them for image conversion
        imgsat = Image.fromarray(np.expand_dims(sat, 0), mode="L")
        imgval = Image.fromarray(np.expand_dims(val, 0), mode="L")
        self.sat = imgsat
        self.val = imgval
        # Merge arrays into a single HSV image, converting to RGB and extracting as a 1D array
        return np.uint8(Image.merge("HSV", (self.hue, self.sat, self.val)).convert("RGB"))[0]

    def start(self):
        with suppress(StopIteration):
            # Default bar colour to (127, 127, 127) grey for no data
            self.trans[self.cutoff] = 127
            # List of futures to wait for at each frame
            futs = None
            # Current time in nanoseconds, timestamps to wait for every frame as well as estimate remaining render time
            ts = time.time_ns()
            timestamps = deque()
            for i in range(2147483648):
                # Force the first frame to calculate immediately if not yet set
                if self.fut is None:
                    self.fut = exc.submit(self.read)
                # A single line of RGB pixel values from the read() method
                line = self.fut.result()
                # Signal to concurrently begin the next frame's render
                self.fut = exc.submit(self.read)
                # Shift entire image `speed` pixels to the left
                self.trans[self.cutoff + 1:-speed] = self.trans[self.cutoff + speed + 1:]
                # If the current iteration of the loop would indicate that the spectrum lines have passed the bar, begin rendering buffered particle data
                if i >= (screensize[0] - self.cutoff) / speed:
                    if particles:
                        # Wait for particle render if unavailable
                        while not self.effects:
                            time.sleep(0.01)
                            if not self.playing:
                                raise StopIteration
                    if getattr(self, "player", None):
                        # Wait for audio playback if audio is lagging for whatever reason
                        if self.player_buffer:
                            self.player_buffer.result()
                        self.player_buffer = exc.submit(self.play_audio)
                    if particles:
                        img = self.effects.popleft()
                        self.trans[:self.cutoff] = img
                # Fill right side of the image that's now been shifted across with the RGB values calculated earlier
                for x in range(speed):
                    self.trans[-x - 1] = line
                # Convert entire frame's image to byte object, preparing to send to render and display subprocesses
                b = self.image.tobytes()
                # Ensure that all subprocesses are functioning correctly
                for p in ("render", "display", "particles"):
                    if globals().get(p):
                        proc = getattr(self, p[:4], None)
                        if not (proc and proc.is_running() and not proc.stdin.closed):
                            print(proc.stderr.read().decode("utf-8", "replace"))
                            raise StopIteration
                # Wait for current frame to complete for all subprocesses
                if futs:
                    for fut in futs:
                        fut.result()
                # Initialize work buffer for next frame
                futs = deque()
                # Enqueue rendering the display and video to the worker processes
                for p in ("display", "render"):
                    if globals().get(p):
                        proc = getattr(self, p[:4], None)
                        if proc:
                            futs.append(exc.submit(proc.stdin.write, b))
                # Calculate current audio position, total audio duration, estimated remaining render time
                billion = 1000000000
                t = time.time_ns()
                while timestamps and timestamps[0] < t - 60 * billion:
                    timestamps.popleft()
                timestamps.append(t)
                fs = os.path.getsize(f2)
                x = fs / (sample_rate // fps << 2)
                t = max(0, i - (screensize[0] - self.cutoff) / speed)
                ratio = min(1, t / x)
                rem = inf
                with suppress(OverflowError, ZeroDivisionError):
                    rem = (fs / sample_rate / 4 - t / fps) / (len(timestamps) / fps / 60)
                # Display output as a progress bar on the console
                out = f"\r{C.white}|{create_progress_bar(ratio, 64, ((-t * 16 / fps) % 6 / 6))}{C.white}| ({C.green}{time_disp(t / fps)}{C.white}/{C.red}{time_disp(fs / sample_rate / 4)}{C.white}) | Estimated time remaining: {C.magenta}[{time_disp(rem)}]"
                out += " " * (120 - len(nocol(out))) + C.white
                sys.stdout.write(out)
                sys.stdout.flush()
                # Wait until the time for the next frame
                while time.time_ns() < ts + billion / fps:
                    time.sleep(0.001)
                ts = max(ts + billion / fps, time.time_ns() - billion / fps)
        # Close everything and exit
        self.file.close()
        if render:
            self.rend.stdin.close()
        if display:
            self.disp.stdin.close()
        if render:
            self.rend.wait()
            if self.rend.returncode:
                raise RuntimeError(self.rend.returncode)
        if display:
            self.disp.wait()
            if self.disp.returncode:
                raise RuntimeError(self.disp.returncode)
        proc = psutil.Process()
        for child in proc.children(True):
            with suppress(psutil.NoSuchProcess):
                child.kill()
        proc.kill()

    def play_audio(self):
        # Plays one frame of audio calculated by fps and sample rate
        req = sample_rate // fps * 4
        self.player[0].stdin.write(self.player[1].stdout.read(req))


if __name__ == "__main__":
    ytdl = None
    # Get config file data, create a new one if unavailable
    if not os.path.exists("config.json"):
        data = "{" + "\n\t".join((
                '"source": "", # This field may be omitted to be prompted for an input at runtime; may be a file path or URL.',
                '"size": [960, 540], # Both dimensions should be divisible by 4 for best results.',
                '"fps": 60, # Framerate of the output video.',
                '"sample_rate": 48000, # Sample rate to evaluate fourier transforms at, should be a multiple of fps.',
                '"amplitude": 0.0625, # Amplitude to scale audio volume, adjust as necessary.',
                '"speed": 2, # Speed of screen movemement in pixels per frame, does not change audio playback speed.',
                '"resolution": 192, # Resolution of DFT in bars per pixel, this should be a relatively high number due to the logarithmic scale.',
                '"particles": "bubble", # May be one of None, "bar", "bubble", "hexagon", or a file path/URL in quotes to indicate image to use for particles.',
                '"display": true, # Whether to preview the rendered video in a separate window.',
                '"render": true, # Whether to output the result to a video file.',
                '"play": true, # Whether to play the actual audio being rendered.',
            )) + "\n}"
        with open("config.json", "w") as f:
            f.write(data)
    else:
        with open("config.json", "rb") as f:
            data = f.read()
    # Send settings to global variables (because I am lazy lol)
    globals().update(eval_json(data))
    # Take command-line flags such as "-size" to interpret as setting overrides
    argv = sys.argv[1:]
    while len(argv) >= 2:
        if argv[0].startswith("-"):
            arg = argv[1]
            with suppress(SyntaxError, NameError):
                arg = eval(arg)
            globals()[argv[0][1:]] = arg
            argv = argv[2:]
        else:
            break
    if len(argv):
        inputs = argv
    else:
        if not source:
            source = input("No audio input found in config.json; please input audio file by path or URL: ")
        inputs = [source]
    # Calculate remaining required variables from the input settings
    screensize = size
    res_scale = resolution * screensize[1]
    ascale = amplitude * screensize[1] / 2
    sources = deque()
    futs = deque()
    # Render all input sources, initializing audio downloader if any of them are download URLs
    for path in inputs:
        if is_url(path):
            if ytdl is None:
                from audio_downloader import AudioDownloader
                ytdl = AudioDownloader()
            futs.append(exc.submit(ytdl.extract, path))
        else:
            futs.append(path)
    for fut in futs:
        if type(fut) is not str:
            sources.extend(fut.result())
        else:
            sources.append(fut)
    for entry in sources:
        if type(entry) is not str:
            ytdl.extract_single(entry)
            source = entry.stream
            fn = entry.name
        else:
            source = entry
            fn = source.rsplit(".", 1)[0]
        f2 = fn + ".pcm"
        f3 = fn + ".wav"
        f4 = fn + ".mp4"
        print("Loading", source)
        r = Render(source)
        if render:
            print("Rendering", f4)
        r.start()