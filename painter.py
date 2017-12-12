from PIL import Image
import numpy as np
from scipy.misc import imsave
import random
from enum import Enum
from collections import deque
from time import time

# class Visited(Enum):
#     Yes = False
#     No = True

Visited = False
NoVisited = True

class simqueue(object):
    def __init__(self):
        self.de = deque()

    def push(self, temp):
        self.de.append(temp)

    def pop(self):
        return self.de.popleft()

    def extend(self, l):
        self.de.extend(l)

    def empty(self):
        return len(self.de) == 0

    def size(self):
        return len(self.de)


class Painter(object):
    def __init__(self, path):
        self.img = Image.open(path)
        self.img = self.img.convert('RGB')
        self.out = list(self.img.getdata())
        self.out = np.reshape(self.out, (self.img.size[1], self.img.size[0], 3))

        self.img = self.img.convert('1')
        self.mat = list(self.img.getdata())
        self.mat = np.reshape(self.mat, (self.img.size[1], self.img.size[0]))
        self.visited = self.mat.astype(bool)

        self.rgb_range = [0, 255, 0, 255, 0, 255]
        self.pos = (0, 0)
        self.height, self.width = self.img.size[1], self.img.size[0]
        self.left_pixels = self.img.size[0]*self.img.size[1]-np.sum(self.visited)

    def set_rgb_range(self, range_):
        self.rgb_range = range_

    def pick_a_color(self):
        rgb = []
        for i in range(0, 5, 2):
            rgb.append(random.randint(self.rgb_range[i], self.rgb_range[i+1]))
        return tuple(rgb)

    def select_a_no_visited_pixel(self):
        for w in range(self.pos[1], self.width):
            if self.visited[(self.pos[0], w)] == NoVisited:
                return (self.pos[0], w)
        for h in range(self.pos[0]+1, self.height):
            for w in range(0, self.width):
                if self.visited[(h, w)] == NoVisited:
                    return (h, w)
        return None

    def is_pos_illgal(self, pos):
        return  0 <= pos[0] < self.height and 0 <= pos[1] < self.width

    def no_visited_surrounding(self, pos):
        out_pos = []
        sur = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        for tu in sur:
            ps = (pos[0]+tu[0], pos[1]+tu[1])
            if self.is_pos_illgal(ps) and self.visited[ps] == NoVisited:
                out_pos.append(ps)
        return out_pos

    def visit_pixels(self, pixels):
        for pixel in pixels:
            self.visited[pixel] = Visited

    def visit_pixel(self, pixel):
        self.visited[pixel] = Visited


    def pick_one_block(self, pos):
        self.pos = pos
        block = []
        q = simqueue()
        q.push(pos)
        while not q.empty():
            temp = q.pop()
            self.visit_pixel(temp)
            block.append(temp)
            poss = self.no_visited_surrounding(temp)
            self.visit_pixels(poss)
            q.extend(poss)
        return block

    def paint_a_block(self, block, color):
        self.left_pixels -= len(block)
        for pixel in block:
            self.out[pixel] = color

    def run(self):
        while self.left_pixels != 0:
            pos = self.select_a_no_visited_pixel()
            if pos is None:
                return
            block = self.pick_one_block(pos)
            color = self.pick_a_color()
            self.paint_a_block(block, color)

    def save(self, filename):
        imsave(filename, self.out)

if __name__ == '__main__':
    start = time()
    painter = Painter('pics/test3.jpeg')
    # print painter.pick_a_color()
    painter.run()
    painter.save('pics/out3.jpg')
    end = time()
    print 'It takes ', end-start, 's.'







