from main import *
import PIL, colorsys, requests, io
from PIL import Image, ImageDraw, ImageMath, ImageOps, ImageChops


# Read input from the main process, set screen size and decide what to use as the particle
screensize = [int(x) for x in sys.argv[2:4]]
particles = sys.argv[1].casefold()
IMAGE = None
try:
    particles = eval(particles)
except (SyntaxError, NameError):
    if particles == "bar":
        pid = 1
    elif particles in ("bubble", "hexagon"):
        pid = 2
    elif particles == "trail":
        pid = 3
    elif is_url(particles):
        IMAGE = Image.open(io.BytesIO(requests.get(particles).content))
        pid = -2
    else:
        try:
            IMAGE = Image.open(particles)
            pid = -1
        except (OSError, FileNotFoundError):
            raise TypeError(f"Unable to find particle image file: <{particles}>")
# If the particle image was specified, convert to a greyscale image, taking alpha values into account with a black background (since we are using an additive-like blending operation)
if IMAGE:
    if "A" in str(IMAGE.mode):
        A = IMAGE.getchannel("A")
    else:
        A = None
    if str(IMAGE.mode) != "L":
        IMAGE = ImageOps.grayscale(IMAGE)
    if A:
        IMAGE = ImageChops.multiply(IMAGE, A)


# This was originally only for bubble particles but I was too lazy to change the variable name
CIRCLES = {}

class Circles:

    @classmethod
    def sprite(cls, radius, colour):
        tup = (radius, colour)
        try:
            # Use cached sprite if possible
            return CIRCLES[tup]
        except KeyError:
            # Otherwise attempt to create a new sprite and add to the cache, deleting the oldest one if the system has run out of memory
            while True:
                try:
                    try:
                        # Attempt to pull out cached greyscale image of a certain size, generating a new one if required
                        surf = CIRCLES[radius]
                    except KeyError:
                        if IMAGE:
                            # If the particle type is a source image, resize it to the required size
                            img = IMAGE.resize((radius << 1,) * 2, resample=Image.LANCZOS)
                            surf = CIRCLES[radius] = img
                        else:
                            # Otherwise draw a series of concentric hexagons or circles to produce particle shape
                            r = radius + 2
                            surf = CIRCLES[radius] = Image.new("L", (r * 2,) * 2, 0)
                            draw = ImageDraw.Draw(surf)
                            for c in range(2, radius):
                                # Radii of shapes are a bit finnicky to manage in order to ensure that they fit together with no gaps
                                if particles == "hexagon":
                                    draw.regular_polygon((radius, radius, r - c), 6, 0, (c * 192) // (radius - 1) + 63 if c > 5 else c * 51, None)
                                else:
                                    draw.ellipse((r - c, r - c, r + c, r + c), None, (c * 255) // (radius - 1), 2)
                            if particles == "bubble":
                                draw.ellipse((3, 3, r * 2 - 3, r * 2 - 3), None, 192, 1)
                    # Convert greyscale to colour image by scaling RGB channels as required
                    curr = dict(size=surf.size)
                    for c, v in zip("RGB", colour):
                        if v == 255:
                            CIRCLES[tup][c] = surf
                        elif v:
                            CIRCLES[tup][c] = surf.point((np.arange(256) * v / 255).astype(np.uint8))
                    CIRCLES[tup] = curr
                    return curr
                except MemoryError:
                    # Upon running out of memory, remove the oldest cached image
                    CIRCLES.pop(next(iter(CIRCLES))).close()


# Main particle renderer
PARTICLES = {}

class Particles:

    def __init__(self):
        # Determine particle type to use
        self.Particle = Particle = (None, Bar, Bubble, Trail)[pid]
        # Use an index count equal to half the screen height (one position every two pixels)
        s2 = screensize[1] >> 1
        # Calculate array of hues to render particles as
        self.colours = [tuple(round(x * 255) for x in colorsys.hsv_to_rgb(i / s2, 1, 1)) for i in range(s2 + 2)]
        # Use an appropriate buffer for exponentially decreasing values, similar to how typical audio volume bars are designed
        if Particle == Bar:
            self.bars = [Bar(i << 3, self.colours[i << 1]) for i in range(s2 + 1 >> 1)]
        elif Particle in (Bubble, Trail):
            self.hits = np.zeros(s2 + 3 >> 1, dtype=float)

    def animate(self, spawn):
        Particle = self.Particle
        if Particle:
            # Initialize next frame to be an empty black image
            sfx = Image.new("RGB", screensize, (0, 0, 0))
            if Particle == Bar:
                # Raise bars to their current values as required
                for i, pwr in enumerate(spawn):
                    self.bars[i >> 1].ensure(sqrt(pwr) * 64)
                # Display and update bars
                for bar in self.bars:
                    bar.render(sfx=sfx)
                    bar.update()
            elif Particle in (Bubble, Trail):
                # Calculate appropriate particle size and colour, create particle at corresponding pixel positions
                mins = screensize[0] / 1048576
                minp = screensize[0] / 16384
                np.multiply(self.hits, 63 / 64, out=self.hits)
                for x in sorted(range(len(spawn)), key=lambda z: -spawn[z])[:64]:
                    pwr = spawn[x]
                    if pwr >= mins and pwr >= self.hits[x >> 1] * 1.5:
                        self.hits[x >> 1] = pwr
                        pwr /= 1.5
                        pwr += 1 / 64
                        PARTICLES[Particle((screensize[0], x * 4 + 2), colour=self.colours[x << 1], intensity=pwr)] = None
                # Display and update particles
                for particle in PARTICLES:
                    particle.render(sfx=sfx)
                    particle.update()
        return sfx.tobytes()

    def start(self):
        # Byte count, we are reading one 4 byte float per pixel of screen height
        count = screensize[1] // 4 << 2
        while True:
            # Read input amplitude array for current frame, render and animate it, then write output data back to main process
            arr = np.frombuffer(sys.stdin.buffer.read(count), dtype=np.float32)
            if not len(arr):
                break
            temp = self.animate(arr)
            sys.stdout.buffer.write(temp)


# Default (abstract) particle class, does nothing
class Particle(collections.abc.Hashable):

    __slots__ = ("hash",)

    __init__ = lambda self: setattr(self, "hash", random.randint(-2147483648, 2147483647))
    __hash__ = lambda self: self.hash
    update = lambda self: None
    render = lambda self, surf: None


# Bar particle class, simulates an exponentially decreasing bar with a gradient
class Bar(Particle):

    __slots__ = ("y", "colour", "surf", "surf2", "size")
    width = 8
    line = Image.new("RGB", (1, width), 16777215)

    def __init__(self, y, colour):
        self.y = y
        self.colour = colour
        # Generate gradient of 2 pixels
        surf = Image.new("RGB", (2, 1), self.colour)
        surf.putpixel((0, 0), 0)
        self.surf = surf.resize((2, self.width), resample=Image.NEAREST)
        self.size = 0
        super().__init__()

    def update(self):
        if self.size:
            self.size = self.size * 0.97 - 1
            if self.size < 0:
                self.size = 0

    def ensure(self, value):
        if self.size < value:
            self.size = value

    def render(self, sfx, **void):
        size = round(self.size)
        if size:
            # Resize gradient to form a bar, pasting onto the current frame
            surf = self.surf.resize((size, self.width), resample=Image.BILINEAR)
            sfx.paste(surf, (screensize[0] - size, self.y))
            pos = max(0, screensize[0] - size)
            sfx.paste(self.line, (pos, self.y))


# Bubble particle class, simulates a growing and fading particle that floats in a random direction upon spawning
class Bubble(Particle):

    __slots__ = ("pos", "vel", "colour", "intensity", "tick", "rotation", "rotvel")
    halve = (np.arange(1, 257) >> 1).astype(np.uint8)
    darken = np.concatenate((np.zeros(128, dtype=np.uint8), np.arange(128, dtype=np.uint8)))

    def __init__(self, pos, colour=(255, 255, 255), intensity=1, speed=-3, spread=1.6):
        self.pos = np.array(pos, dtype=float)
        angle = (random.random() - 0.5) * spread
        if speed:
            self.vel = np.array([speed * cos(angle), speed * sin(angle)])
            self.rotvel = (random.random() - 0.5) * 20
        else:
            self.vel = 0
            self.rotvel = 0
        self.rotation = random.random() * 360
        self.colour = colour
        self.intensity = intensity
        self.tick = 0.
        super().__init__()

    def update(self):
        if issubclass(type(self.vel), collections.abc.Sized) and len(self.vel):
            self.pos += self.vel
        maxsize = min(256, sqrt(self.intensity) * 16)
        if self.tick >= maxsize:
            self.intensity *= 0.93
            if self.intensity <= 1 / 64:
                self.intensity = 0
                PARTICLES.pop(self)
        else:
            self.tick += sqrt(maxsize) / 4
        if particles == "hexagon":
            self.rotation += self.rotvel
            self.rotvel *= 0.95

    def render(self, sfx, **void):
        intensity = min(self.intensity / 16, 3)
        colour = [x * intensity / screensize[0] * 256 for x in self.colour]
        for i, x in enumerate(colour):
            if x > 255:
                temp = (x - 256) / 4
                colour[i] = 255
                for j in range(len(colour)):
                    if i != j:
                        colour[j] = min(255, temp + colour[j])
        sprite = Circles.sprite(max(4, int(self.tick)), tuple(min(255, round(x / 8) << 3) for x in colour))
        if len(sprite) > 1:
            size = sprite["size"]
            offs = [x >> 1 for x in size]
            pos = tuple(int(x) for x in self.pos - offs)
            crop = sfx.crop(pos + tuple(pos[i] + size[i] for i in range(2)))
            if crop.size[0] and crop.size[1]:
                # Hexagon particles support random rotation while being displayed
                if particles == "hexagon":
                    sprite = {k: v.rotate(self.rotation) for k, v in sprite.items() if type(v) is Image.Image}
                if crop.size != size:
                    sprite = {k: v.crop((0,) * 2 + crop.size) for k, v in sprite.items() if type(v) is Image.Image}
                # Custom "overflow" blend operation, similar to additive but taking spectral sensitivity into account and scaling up other channels instead of simply capping at 255
                # This results in colours like two pure reds adding to a pink, instead of the same red
                spl = crop.split()
                out = list(spl)
                for i, c, I in zip(range(3), "RGB", spl):
                    try:
                        s = sprite[c]
                    except KeyError:
                        continue
                    if out[i] == I:
                        out[i] = ImageChops.add(I, s)
                    else:
                        out[i] = ImageChops.add(out[i], s)
                    # As the triple add is an expensive operation, only calculate it if it would be required
                    extrema = out[i].getextrema()
                    if extrema[-1] == 255:
                        # The value to add to other channels is currently (x / 2 + y / 2) - 128
                        im1 = I.point(self.halve)
                        im2 = s.point(self.halve)
                        overflow = ImageChops.add(im1, im2)
                        overflow = overflow.point(self.darken)
                        extrema = overflow.getextrema()
                        # If the output scaled channels have values in them, place the resulting scaled images back into output
                        if extrema[-1]:
                            for j in (x for x in range(3) if x != i):
                                out[j] = ImageChops.add(out[j], overflow)
                result = Image.merge("RGB", out)
                # Paste clipped output area onto current frame as a sprite
                sfx.paste(result, pos)


class Trail(Particle):
    pass


if __name__ == "__main__":
    engine = Particles()
    engine.start()