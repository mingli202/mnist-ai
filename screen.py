import numpy as np
import pygame

from model import Model


def main(m: Model, X=None):
    pygame.init()
    screen = pygame.display.set_mode((1600, 840))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 25)
    run = True

    board = Board(X)
    x = -1
    y = -1
    count = 0

    while run:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False

            if e.type == pygame.MOUSEMOTION or e.type == pygame.MOUSEBUTTONDOWN:
                _x, _y = pygame.mouse.get_pos()

                if int(_x / board.rect_width) != x or int(_y / board.rect_width) != y:
                    x = int(_x / board.rect_width)
                    y = int(_y / board.rect_width)
                    count = 0

                else:
                    count += 1

            if e.type == pygame.KEYDOWN:
                if (
                    pygame.key.get_pressed()[pygame.K_BACKSPACE]
                    or pygame.key.get_pressed()[pygame.K_ESCAPE]
                    or pygame.key.get_pressed()[pygame.K_TAB]
                ):
                    board.x = np.zeros((28, 28))

        screen.fill("black")

        if pygame.mouse.get_pressed()[0]:
            if count < 1 and 0 <= x <= 27 and 0 <= y <= 27:
                board.updateGrid(x, y)

        board.grid(screen)
        board.predict(m, font, screen)

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


class Board:
    def __init__(self, x):
        self.x = np.zeros((28, 28)) if x is None else x.T
        self.rect_width = 30

    def grid(self, screen: pygame.Surface):
        spaces = np.arange(0, 29 * self.rect_width, self.rect_width)

        vertical = [((i, 0), (i, 840)) for i in spaces]
        horizontal = [((0, i), (840, i)) for i in spaces]
        pts = vertical + horizontal

        for i in range(28):
            for k in range(28):
                color = self.x[i][k]
                pygame.draw.rect(
                    screen,
                    (color, color, color),
                    pygame.Rect(
                        self.rect_width * i,
                        self.rect_width * k,
                        self.rect_width,
                        self.rect_width,
                    ),
                )

        for pi, pf in pts:
            pygame.draw.line(screen, (100, 100, 100), pi, pf)

    def updateGrid(self, i_x: int, i_y: int):
        center = 100
        side = 50
        diag = 25

        self.x[i_x][i_y] = min(self.x[i_x][i_y] + center, 255)

        if i_x - 1 >= 0:
            self.x[i_x - 1][i_y] = min(self.x[i_x - 1][i_y] + side, 255)

            if i_y - 1 >= 0:
                self.x[i_x - 1][i_y - 1] = min(self.x[i_x - 1][i_y - 1] + diag, 255)

            if i_y + 1 <= 27:
                self.x[i_x - 1][i_y + 1] = min(self.x[i_x - 1][i_y + 1] + diag, 255)

        if i_x + 1 <= 27:
            self.x[i_x + 1][i_y] = min(self.x[i_x + 1][i_y] + side, 255)

            if i_y - 1 >= 0:
                self.x[i_x + 1][i_y - 1] = min(self.x[i_x + 1][i_y - 1] + diag, 255)

            if i_y + 1 <= 27:
                self.x[i_x + 1][i_y + 1] = min(self.x[i_x + 1][i_y + 1] + diag, 255)

        if i_y - 1 >= 0:
            self.x[i_x][i_y - 1] = min(self.x[i_x][i_y - 1] + side, 255)

        if i_y + 1 <= 27:
            self.x[i_x][i_y + 1] = min(self.x[i_x][i_y + 1] + side, 255)

    def predict(self, m: Model, font: pygame.font.Font, screen: pygame.Surface):
        z0 = self.x.T.reshape(784, 1) / 255

        _, _, z = m.forward_propagation(z0)

        prediction = np.argmax(z)

        msg = font.render(f"{prediction}", True, (255, 255, 255))
        screen.blit(
            msg,
            (760 / 2 - msg.get_width() / 2 + screen.get_height(), 30),
        )

        spacing = np.linspace(0, 700 - 60, 10) + 870

        for i, s in enumerate(spacing):
            msgn = font.render(f"{i}", True, (255, 255, 255))
            screen.blit(msgn, (s + 30 - msgn.get_width() / 2, 800))

            pygame.draw.rect(
                screen,
                (255, 255, 255),
                pygame.Rect(s, 790 - 600 * z[i][0], 60, 600 * z[i][0]),
            )
