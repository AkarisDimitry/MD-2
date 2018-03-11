import pygame, time, thread
import numpy as np
from display.Display import *

pygame.init()
display = Display()
cursor = Cursor()

def main(atomic_data, graph_data):
    timer = pygame.time.Clock()
    pygame.display.set_caption('MD - molecular Dinamics')
    display.atomic_data  = atomic_data
    display.graph_data = [graph_data]
    thread.start_new_thread( display.update, () )

    while True:
         display.keys_press(cursor)
         display.change_zoom( cursor, 1.1)
         timer.tick(30)



