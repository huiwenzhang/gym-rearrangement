import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLOR = [
    (0, 0, 210),
    (0, 210, 0),
    (210, 0, 0),
    (150, 150, 0),
    (150, 0, 150),
    (0, 150, 150),
    # add more colors here if needed
]

N_GRID = 3
NUM_COLOR = len(COLOR)
# the number of objects presented in each image
# NUM_SHAPE = self.n_objects
# avoid a color shared by more than one objects
# NUM_SHAPE = min(NUM_SHAPE, NUM_COLOR)
NUM_Q = 5


def color2str(code):
    return {
        0: 'blue',
        1: 'green',
        2: 'red',
        3: 'yellow',
        4: 'magenta',
        5: 'cyan',
    }[code]


def question2str(qv):
    def q_type(q):
        return {
            0: 'is it a circle or a rectangle?',
            1: 'is it closer to the bottom of the image?',
            2: 'is it on the left of the image?',
            3: 'the color of the nearest object?',
            4: 'the color of the farthest object?',
        }[q]

    color = np.argmax(qv[:NUM_COLOR])
    q_num = np.argmax(qv[NUM_COLOR:])  # which questions should we ask?
    return '[Query object color: {}] [Query: {}]'.format(color2str(color),
                                                         q_type(q_num))


def answer2str(av, prefix=None):
    def a_type(a):
        return {
            0: 'blue',
            1: 'green',
            2: 'red',
            3: 'yellow',
            4: 'magenta',
            5: 'cyan',
            6: 'circle',
            7: 'rectangle',
            8: 'yes',
            9: 'no',
        }[np.argmax(a)]

    if not prefix:
        return '[Answer: {}]'.format(a_type(av))
    else:
        return '[{} Answer: {}]'.format(prefix, a_type(av))


def visualize_iqa(img, q, a):
    fig = plt.figure()
    plt.imshow(img)
    plt.title(question2str(q))
    plt.xlabel(answer2str(a))
    return fig


class Representation:

    def __init__(self, x, y, color, shape):
        self.x = x
        self.y = y
        self.color = color
        self.shape = shape

    def print_graph(self):
        for i in range(len(self.x)):
            s = 'circle' if self.shape[i] else 'rectangle'
            print('{} {} at ({}, {})'.format(color2str(self.color[i]),
                                             s, self.x[i], self.y[i]))
