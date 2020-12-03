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
    exc = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    from PIL import Image, ImageChops

np = numpy
deque = collections.deque

url_match = re.compile("^(?:http|hxxp|ftp|fxp)s?:\\/\\/[^\\s<>`|\"']+$")
is_url = lambda url: url_match.search(url)

eval_json = lambda s: eval(s, dict(true=True, false=False, null=None, none=None), {})


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

bar = "∙░▒▓█"
col = [C.red, C.yellow, C.green, C.cyan, C.blue, C.magenta]
def create_progress_bar(ratio, length=32, offset=None):
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


sample_rate = 48000
fps = 60
amplitude = 1 / 8
render = display = particles = play = 0
speed = resolution = 1
screensize = size = (960, 540)


class Render:

    def __init__(self, f_in):
        self.cutoff = screensize[0] >> 2
        opt = "-n"
        if os.path.exists(f2):
            if is_url(source) or os.path.getmtime(f2) < max(os.path.getctime(source), os.path.getmtime(source)):
                opt = "-y"
        args = ["ffmpeg", opt, "-hide_banner", "-loglevel", "error", "-i", f_in, "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", f2]
        print(" ".join(args))
        fut1 = exc.submit(psutil.Popen, args, stderr=subprocess.PIPE)
        if render:
            args = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-r", str(fps), "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", "x".join(str(i) for i in screensize), "-i", "-"]
            if play:
                args.extend(("-vn", "-i", f_in))
            args.extend(("-b:v", "4M"))
            if play:
                d = round((screensize[0] - self.cutoff) / speed / fps * 1000)
                args.extend(("-af", f"adelay=delays={d}:all=1"))
            args.append(f3)
            print(" ".join(args))
            fut2 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if display:
            args = [("python3", "python")[os.name == "nt"], "display.py", *[str(x) for x in screensize]]
            print(" ".join(args))
            fut3 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if particles:
            args = [("python3", "python")[os.name == "nt"], "particles.py", str(particles), str(self.cutoff), str(screensize[1])]
            print(" ".join(args))
            fut4 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if play:
            args = ["ffplay", "-loglevel", "error", "-hide_banner", "-nodisp", "-autoexit", "-f", "s16le", "-ar", str(sample_rate), "-ac", "2", "-i", "-"]
            print(" ".join(args))
            fut5 = exc.submit(psutil.Popen, args, stdin=subprocess.PIPE)
            args = ["ffmpeg", "-loglevel", "error", "-i", f_in, "-f", "s16le", "-ar", str(sample_rate), "-ac", "2", "-"]
            print(" ".join(args))
            fut6 = exc.submit(psutil.Popen, args, stdout=subprocess.PIPE)
        self.player_buffer = None
        self.emptybuff = np.zeros(res_scale, dtype=np.float32)
        self.buffer = self.emptybuff
        dfts = (res_scale >> 1) + 1
        self.fff = np.fft.fftfreq(res_scale, 1 / sample_rate)[:dfts]
        maxfreq = np.max(self.fff)
        self.fftrans = np.zeros(dfts, dtype=int)
        for i, x in enumerate(self.fff):
            if x <= 0:
                x = screensize[1] - 1
            else:
                x = round((1 - log(x, maxfreq)) * pi / 2 * (screensize[1] - 1))
            if x > screensize[1] - 1:
                x = screensize[1] - 1
            self.fftrans[i] = x
        self.linear_scale = np.arange(screensize[1], dtype=np.float64) / screensize[1]
        self.hue = Image.fromarray(np.expand_dims((self.linear_scale * 256).astype(np.uint8), 0), mode="L")
        self.sat = Image.new("L", (screensize[1], 1), 255)
        self.val = self.sat
        self.scale = (ascale / dfts) #*np.sqrt(self.linear_scale[::-1])
        self.fut = None
        self.playing = True
        proc = fut1.result()
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
        self.file = open(f2, "rb")
        self.image = np.zeros((screensize[1], screensize[0], 3), dtype=np.uint8)
        self.trans = np.swapaxes(self.image, 0, 1)
        self.effects = deque()
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
        req = (res_scale) - len(self.buffer)
        if req > 0:
            data = self.file.read(req << 2)
            self.buffer = np.concatenate((self.buffer, np.frombuffer(data, dtype=np.float32)))
        else:
            data = True
        req = (res_scale) - len(self.buffer)
        if req > 0:
            self.buffer = np.concatenate((self.buffer, self.emptybuff[:req]))
        if not particles and not data:
            if not (np.min(self.buffer) or np.max(self.buffer)):
                raise StopIteration
        dft = np.fft.rfft(self.buffer[:res_scale])
        self.buffer = self.buffer[sample_rate // fps:]
        arr = np.zeros(screensize[1], dtype=np.complex128)
        np.add.at(arr, self.fftrans, dft)
        amp = np.abs(arr, dtype=np.float32)
        amp = np.multiply(np.multiply(amp, self.scale, out=amp), 256, out=amp)
        sat = np.clip(511 - amp, 0, 255).astype(np.uint8)
        val = np.clip(amp, 0, 255).astype(np.uint8)
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
        imgsat = Image.fromarray(np.expand_dims(sat, 0), mode="L")
        imgval = Image.fromarray(np.expand_dims(val, 0), mode="L")
        self.sat = imgsat
        self.val = imgval
        return np.uint8(Image.merge("HSV", (self.hue, self.sat, self.val)).convert("RGB"))[0]

    def start(self):
        with suppress(StopIteration):
            self.trans[self.cutoff] = 255
            futs = None
            ts = time.time_ns()
            timestamps = deque()
            for i in range(2147483648):
                if self.fut is None:
                    self.fut = exc.submit(self.read)
                line = self.fut.result()
                self.fut = exc.submit(self.read)
                self.trans[self.cutoff + 1:-speed] = self.trans[self.cutoff + speed + 1:]
                if i >= (screensize[0] - self.cutoff) / speed:
                    if particles:
                        while not self.effects:
                            time.sleep(0.01)
                            if not self.playing:
                                raise StopIteration
                    if getattr(self, "player", None):
                        if self.player_buffer:
                            self.player_buffer.result()
                        self.player_buffer = exc.submit(self.play_audio)
                    if particles:
                        img = self.effects.popleft()
                        self.trans[:self.cutoff] = img
                for x in range(speed):
                    self.trans[-x - 1] = line
                b = self.image.tobytes()
                for p in ("render", "display", "particles"):
                    if globals().get(p):
                        proc = getattr(self, p[:4], None)
                        if not (proc and proc.is_running() and not proc.stdin.closed):
                            print(proc.stderr.read().decode("utf-8", "replace"))
                            raise StopIteration
                if futs:
                    for fut in futs:
                        fut.result()
                futs = deque()
                for p in ("display", "render"):
                    if globals().get(p):
                        proc = getattr(self, p[:4], None)
                        if proc:
                            futs.append(exc.submit(proc.stdin.write, b))
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
                sys.stdout.write(f"\r{C.white}|{create_progress_bar(ratio, 64, ((-t * 16 / fps) % 6 / 6))}{C.white}| ({C.green}{time_disp(t / fps)}{C.white}/{C.red}{time_disp(fs / sample_rate / 4)}{C.white}) | Estimated time remaining: {C.magenta}[{time_disp(rem)}]{' ' * 6}{C.white}")
                # sys.stdout.write("\033[2K\033[1G" + str(ratio) + "%")
                sys.stdout.flush()
                while time.time_ns() < ts + billion / fps:
                    time.sleep(0.001)
                ts = max(ts + billion / fps, time.time_ns() - billion / fps)
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
        req = sample_rate // fps * 4
        self.player[0].stdin.write(self.player[1].stdout.read(req))


if __name__ == "__main__":
    ytdl = None

    if not os.path.exists("config.json"):
        data = "{" + "\n\t".join((
                '"source": "Paladin.mp3",',
                '"size": [960, 540],',
                '"fps": 60,',
                '"sample_rate": 48000,',
                '"amplitude": 0.0625,',
                '"speed": 2,',
                '"resolution": 192,',
                '"particles": "bubble",',
                '"display": true,',
                '"render": true,',
                '"play": true,',
            )) + "\n}"
        with open("config.json", "w") as f:
            f.write(data)
    else:
        with open("config.json", "rb") as f:
            data = f.read() 
    globals().update(eval_json(data))
    argv = sys.argv[1:]
    while len(argv) >= 2:
        if argv[0].startswith("-"):
            globals()[argv[0][1:]] = argv[1]
            argv = argv[2:]
        else:
            break
    if len(argv):
        inputs = argv
    else:
        inputs = [source]
    screensize = size
    res_scale = resolution * screensize[1]
    ascale = amplitude * screensize[1] / 2
    sources = deque()
    futs = deque()
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
        f3 = fn + ".mp4"
        print("Loading", source)
        r = Render(source)
        if render:
            print("Rendering", f3)
        r.start()