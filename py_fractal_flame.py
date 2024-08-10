"""A fractal flame generator, written in Python"""

"""work in progress"""

from PIL import Image
from threading import Thread, Lock
import queue
import random
import math


class Coefficient:

    def __init__(self):
        self.ac = random.random() * 1.5
        self.bc = random.random() * 1.5
        self.cc = random.random() * 2.0
        self.dc = random.random() * 1.5
        self.ec = random.random() * 1.5
        self.fc = random.random() * 2.0
        self.pa1 = random.random() * 2.0
        self.pa2 = random.random() * 2.0
        self.pa3 = random.random() * 2.0
        self.pa4 = random.random() * 2.0
        self.r = random.randint(64, 256)
        self.g = random.randint(64, 256)
        self.b = random.randint(64, 256)

    def get_rgb(self):
        return self.r, self.g, self.b

    def set_rgb(self, r, g, b):
        self.r, self.g, self.b = r, g, b

    def set_coeffs(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


class Pixel:

    def __init__(self, xy, im):
        self.lock = Lock()
        self.count = 0
        self.normal = 0.0
        self.xy = xy
        self.im = im

    def increment(self, pt_rgb):
        with self.lock:
            if self.count == 0:
                self.im.putpixel(self.xy, pt_rgb)
            else:
                r, g, b = pt_rgb
                red, green, blue = self.im.getpixel(self.xy)
                red = (red + r) // 2
                green = (green + g) // 2
                blue = (blue + b) // 2
                self.im.putpixel(self.xy, (red, green, blue))
            self.count += 1

    def log_adjust(self, gamma=2.2):
        correction = pow(self.normal, 1.0 / gamma)
        red, green, blue = self.im.getpixel(self.xy)
        red = min(255, int(red * correction))
        blue = min(255, int(blue * correction))
        green = min(255, int(green * correction))
        self.im.putpixel(self.xy, (red, green, blue))


def affine(x, y, c):
    return c.ac * x + c.bc * y + c.cc, c.dc * x + c.ec * y + c.fc


def linear(x, y, c):
    x, y = affine(x, y, c)
    return x, y


def sinusoidal(x, y, c):
    x, y = affine(x, y, c)
    return math.sin(x), math.sin(y)


def spherical(x, y, c):
    x, y = affine(x, y, c)
    r = 1.0 / (x * x + y * y)
    return r * x, r * y


def diamond(x, y, c):
    x, y = affine(x, y, c)
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)
    return math.sin(theta) * math.cos(r), math.cos(theta) * math.sin(r)


class TransformGroup:
    defined = [linear, spherical, diamond]

    def __init__(self):
        self.transforms = []
        self.weights = []
        while not self.transforms:
            for entry in TransformGroup.defined:
                if random.choice((True, False)):
                    self.transforms.append(entry)
                    self.weights.append(random.uniform(0.0, 1.0))

    def get_transforms(self, k):
        yield from random.choices(self.transforms, weights=self.weights, k=k)


class FractalFlame:

    def __init__(self, width, height, n, super_sample=3):
        self.super_sample = super_sample
        self.width = width * super_sample
        self.height = height * super_sample
        self.im = Image.new(mode="RGB", size=(self.width, self.height), color="black")
        self.xmin = -10.0
        self.xmax = 10.0
        self.ymin = -10.0
        self.ymax = 10.0
        self.ranx = self.xmax - self.xmin
        self.rany = self.ymax - self.ymin
        self.max_hits = 0
        self.post_process_queue = queue.Queue()
        self.normalize_queue = queue.Queue()
        self.reduce_queue = queue.Queue() if super_sample > 1 else None
        self.pixels = dict()
        for y in range(self.width):
            for x in range(self.height):
                self.pixels[(x, y)] = Pixel((x, y), self.im)
        self.coefficients = [Coefficient() for _ in range(n)]
        self.xforms = TransformGroup()

    def generate(self, starts=100, steps=2000):
        for _ in range(starts):
            x = self.xmin + self.ranx * random.random()
            y = self.ymin + self.rany * random.random()
            xform = self.xforms.get_transforms(20 + steps + 1)
            for step in range(-20, steps):
                c = random.choice(self.coefficients)
                x, y = next(xform)(x, y, c)
                if step > 0:
                    if self.xmin <= x <= self.xmax \
                            and self.ymin <= y <= self.ymax:
                        i = self.width - int(((self.xmax - x) / self.ranx) * self.width)
                        j = self.height - int(((self.ymax - y) / self.rany) * self.height)
                        if (i, j) in self.pixels:
                            self.pixels[(i, j)].increment(c.get_rgb())


    def enqueue_pixels_post(self):
        for pixel in self.pixels.values():
            self.max_hits = max(self.max_hits, pixel.count)
            self.post_process_queue.put(pixel)

    def enqueue_pixels_norm(self):
        for pixel in self.pixels.values():
            self.normalize_queue.put(pixel)

    def post_process_worker(self):
        while True:
            pixel = self.post_process_queue.get()
            pixel.normal = math.log(pixel.count) if pixel.count else 0.0
            self.post_process_queue.task_done()

    def normalize_worker(self):
        while True:
            pixel = self.normalize_queue.get()
            pixel.normal /= self.max_hits
            pixel.log_adjust()
            self.normalize_queue.task_done()

    def render(self, num_threads=8):
        threads = set()
        for _ in range(max(num_threads, 2)):
            thread = Thread(target=flame.generate)
            thread.start()
            threads.add(thread)

        for thread in threads:
            thread.join()

        print("done with render")
        flame.enqueue_pixels_post()
        print("done with enqueue")
        for _ in range(6):
            thread = Thread(target=flame.post_process_worker, daemon=True)
            thread.start()
        flame.post_process_queue.join()
        while not flame.post_process_queue.empty():
            pass
        print("done with post")
        flame.enqueue_pixels_norm()
        print("done with enqueue 2")
        self.max_hits = 1.0 if self.max_hits == 0.0 else math.log(self.max_hits)
        for _ in range(4):
            thread = Thread(target=flame.normalize_worker, daemon=True)
            thread.start()
        flame.normalize_queue.join()
        while not flame.normalize_queue.empty():
            pass
        print("done with normalize")
        if self.super_sample > 1:
            self.width //= self.super_sample
            self.height //= self.super_sample
            self.im = self.im.resize((self.width, self.height), resample=5)

    def write_to_file(self, file_name):
        self.im.save(file_name)


def apply_coeff_file(flame, config_file):
    flame.coefficients = []
    coeffs = dict()
    with open(config_file, 'r') as config_file:
        for entry in config_file:
            coeff = Coefficient()
            kwarg = dict()
            for k, v in zip(["ac", "bc", "cc", "dc", "ec", "fc"], entry.strip().split()):
                kwarg[k] = float(v)
            coeff.set_coeffs(**kwarg)
            flame.coefficients.append(coeff)


if __name__ == "__main__":
    flame = FractalFlame(500, 500, 10)
    apply_coeff_file(flame, "./test.coeff")
    flame.render()
    flame.write_to_file("/tmp/fractal.png")
