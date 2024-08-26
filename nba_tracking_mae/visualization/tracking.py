import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from PIL import Image

import constants
from data.transforms import normalize_coordinates


SCALING_CONSTANT = 100


class TrackingVisualization:
    def __init__(self, tracking_data):
        self.tracking_data = tracking_data

    @classmethod
    def from_file_path(cls, path: str):
        tracking_data = torch.load(path)
        tracking_data = normalize_coordinates(x=tracking_data)
        return cls(tracking_data=tracking_data)

    @classmethod
    def from_tensor(cls, tracking_tensor: torch.tensor):
        return cls(tracking_data=tracking_tensor)

    def update_radius(self, i, player_circles, ball_circle, annotations):
        for j, circle in enumerate(player_circles):
            circle.center = (
                self.tracking_data[j + 1][i][0] * SCALING_CONSTANT,
                self.tracking_data[j + 1][i][1] * SCALING_CONSTANT,
            )
            annotations[j].set_position(circle.center)
        ball_circle.center = (
            self.tracking_data[0][i][0] * SCALING_CONSTANT,
            self.tracking_data[0][i][1] * SCALING_CONSTANT,
        )
        return player_circles, ball_circle

    def execute(self):
        ax = plt.axes(
            xlim=(-1 * SCALING_CONSTANT, 1 * SCALING_CONSTANT), ylim=(-1 * SCALING_CONSTANT, 1 * SCALING_CONSTANT)
        )
        ax.axis("off")
        fig = plt.gcf()
        ax.grid(False)

        annotations = [
            ax.annotate(
                i, xy=[0, 0], color="w", horizontalalignment="center", verticalalignment="center", fontweight="bold"
            )
            for i in range(10)
        ]

        player_circles = [plt.Circle((0, 0), 5, color="blue" if i < 5 else "red") for i in range(10)]
        ball_circle = plt.Circle((0, 0), 3, color="black")
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
            fig,
            self.update_radius,
            fargs=(player_circles, ball_circle, annotations),
            frames=self.tracking_data.size(dim=1),
            interval=150,
        )
        plt.show()


class DualTrackingVisualization:
    def __init__(self, tracking_data_0: torch.tensor, tracking_data_1: torch.tensor, masked_indexes: list[int]):
        self.tracking_data_0 = tracking_data_0
        self.tracking_data_1 = tracking_data_1
        self.masked_indexes = masked_indexes

    @classmethod
    def from_tensor(cls, tracking_tensor_0: torch.tensor, tracking_tensor_1: torch.tensor, masked_indexes: list[int]):
        return cls(tracking_data_0=tracking_tensor_0, tracking_data_1=tracking_tensor_1, masked_indexes=masked_indexes)

    @staticmethod
    def update_radius(tracking_data, i, player_circles, ball_circle, annotations):
        for j, circle in enumerate(player_circles):
            circle.center = (
                tracking_data[j + 1][i][0] * SCALING_CONSTANT,
                tracking_data[j + 1][i][1] * SCALING_CONSTANT,
            )
            annotations[j].set_position(circle.center)
        ball_circle.center = (
            tracking_data[0][i][0] * SCALING_CONSTANT,
            tracking_data[0][i][1] * SCALING_CONSTANT,
        )
        return player_circles, ball_circle

    def run_animation(self, i, player_circles, ball_circles, annotations):
        data = [self.tracking_data_0, self.tracking_data_1]
        for a, _ in enumerate(player_circles):
            player_circles[a], ball_circles[a] = self.update_radius(
                tracking_data=data[a],
                i=i,
                player_circles=player_circles[a],
                ball_circle=ball_circles[a],
                annotations=annotations[a],
            )

        return player_circles, ball_circles

    def execute(self):
        fig, (ax_0, ax_1) = plt.subplots(1, 2)
        fig.set_figheight(5)
        fig.set_figwidth(10)

        ax_0.set_title("Masked Input")
        ax_1.set_title("Model Output")

        for ax in [ax_0, ax_1]:
            ax.set_xlim(-1 * SCALING_CONSTANT, 1 * SCALING_CONSTANT)
            ax.set_ylim(-1 * SCALING_CONSTANT, 1 * SCALING_CONSTANT)
            ax.axis("off")
            ax.grid(False)

        annotations_0 = [
            ax_0.annotate(
                i, xy=[0, 0], color="w", horizontalalignment="center", verticalalignment="center", fontweight="bold"
            )
            for i in range(10)
        ]

        player_circles_0 = [
            plt.Circle(
                (0, 0),
                5,
                color=("#327ba8" if i + 1 in self.masked_indexes else "blue")
                if i < 5
                else ("#d15c5c" if i + 1 in self.masked_indexes else "red"),
            )
            for i in range(10)
        ]
        ball_circle_0 = plt.Circle((0, 0), 3, color="grey" if 0 in self.masked_indexes else "black")
        for circle in player_circles_0:
            ax_0.add_patch(circle)
        ax_0.add_patch(ball_circle_0)

        annotations_1 = [
            ax_1.annotate(
                i, xy=[0, 0], color="w", horizontalalignment="center", verticalalignment="center", fontweight="bold"
            )
            for i in range(10)
        ]

        player_circles_1 = [
            plt.Circle(
                (0, 0),
                5,
                color=("#327ba8" if i + 1 in self.masked_indexes else "blue")
                if i < 5
                else ("#d15c5c" if i + 1 in self.masked_indexes else "red"),
            )
            for i in range(10)
        ]
        ball_circle_1 = plt.Circle((0, 0), 3, color="grey" if 0 in self.masked_indexes else "black")
        for circle in player_circles_1:
            ax_1.add_patch(circle)
        ax_1.add_patch(ball_circle_1)

        annotations = [annotations_0, annotations_1]
        player_circles = [player_circles_0, player_circles_1]
        ball_circles = [ball_circle_0, ball_circle_1]

        anim = animation.FuncAnimation(
            fig,
            self.run_animation,
            fargs=(player_circles, ball_circles, annotations),
            frames=self.tracking_data_0.size(dim=1),
            interval=150,
        )

        plt.show()
        # anim.save(f"{constants.ROOT_DIR}/data/visualizations/output.gif", writer="pillow", fps=5)


def do_work(path: str):
    tv = TrackingVisualization.from_file_path(path=path)
    tv.execute()
