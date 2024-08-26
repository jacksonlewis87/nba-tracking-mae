from dataclasses import dataclass


@dataclass
class Coordinate:
    x: float
    y: float
    z: float

    def to_row(self):
        return [self.x, self.y, self.z]


@dataclass
class Frame:
    ball: Coordinate
    players: dict[int:Coordinate]


@dataclass
class Event:
    game_id: str
    event_id: int
    game_clock_start: float
    game_clock_end: float
    wall_clock_start: float
    wall_clock_end: float
    team_0_id: int
    team_1_id: int
    team_0_players: list[int]
    team_1_players: list[int]
    frames: list[Frame]
