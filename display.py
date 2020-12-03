from main import *
import pygame
from pygame.locals import *
exc = concurrent.futures.ThreadPoolExecutor(max_workers=3)


screensize = [int(x) for x in sys.argv[1:3]]


class Display:

    def __init__(self):
        self.length = screensize[0] * screensize[1] * 3
        self.disp = pygame.display.set_mode(screensize)
        pygame.display.set_caption("SpectralPulse ~ Render Preview")
        pygame.display.set_icon(pygame.image.load("icon.png"))

    def start(self):
        fut = None
        while True:
            if fut is None:
                fut = exc.submit(sys.stdin.buffer.read, self.length)
            image = fut.result()
            if not image:
                break
            fut = exc.submit(sys.stdin.buffer.read, self.length)
            surf = pygame.image.fromstring(image, screensize, "RGB")
            self.disp.blit(surf, (0, 0))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == QUIT:
                    break
            pygame.event.clear()


if __name__ == "__main__":
    r = Display()
    r.start()