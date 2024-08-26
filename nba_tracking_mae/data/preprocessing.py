import copy
import os
import torch
from data.tracking_classes import Coordinate, Frame, Event
from data.preprocessing_config import PreprocessingConfig
from math import floor
from utils import load_json, list_files_in_directory


def create_coordinate_from_tracking_row(row: list):
    return Coordinate(x=row[2], y=row[3], z=row[4])


def process_tracking_event(tracking_event: dict, game_id: str):
    try:
        event_id = int(tracking_event["eventId"])
        quarter = None
        game_clock_start = None
        game_clock_end = None
        wall_clock_start = None
        wall_clock_end = None
        teams = {}
        frames = []
        for frame in tracking_event["moments"]:

            # quarter
            if quarter is None:
                quarter = frame[0]

            # clocks
            if game_clock_start is None:
                game_clock_start = frame[2]
            if wall_clock_start is None:
                wall_clock_start = frame[1]
            game_clock_end = frame[2]
            wall_clock_end = frame[1]

            # players
            ball = None
            players = {}
            for player in frame[5]:
                # ball
                if player[0] == -1 or player[1] == -1:
                    if ball is not None:
                        print("Error: duplicate ball")
                        raise Exception
                    ball = create_coordinate_from_tracking_row(row=player)
                else:
                    team_id = player[0]
                    player_id = player[1]
                    if team_id not in teams:
                        if len(teams.keys()) >= 2:
                            print("Error: more than two teams event")
                            raise Exception
                        teams[team_id] = []

                    if player_id not in teams[team_id]:
                        if len(teams[team_id]) >= 5:
                            print("Error: more than 5 players on a team within event")
                            raise Exception
                        teams[team_id] += [player_id]

                    if player_id in players:
                        print("Error: duplicate player in frame")
                        raise Exception

                    players[player_id] = create_coordinate_from_tracking_row(row=player)

            if ball is None:
                print("Error: missing ball")
                raise Exception

            frames += [
                Frame(
                    ball=ball,
                    players=players,
                )
            ]

        team_ids = list(teams.keys())
        if len(team_ids) != 2:
            print("Error: wrong number of teams")
            raise Exception

        if len(teams[team_ids[0]]) != 5 or len(teams[team_ids[1]]) != 5:
            print("Error: wrong number of players per team")
            raise Exception

        return Event(
            game_id=game_id,
            event_id=event_id,
            game_clock_start=game_clock_start,
            game_clock_end=game_clock_end,
            wall_clock_start=wall_clock_start,
            wall_clock_end=wall_clock_end,
            team_0_id=team_ids[0],
            team_1_id=team_ids[1],
            team_0_players=teams[team_ids[0]],
            team_1_players=teams[team_ids[1]],
            frames=frames,
        )

    except Exception as e:
        print("bad event")
        return None


def load_game(path: str, game_id: str):
    tracking_data = load_json(f"{path}/{game_id}.json")

    events = []
    for tracking_event in tracking_data["events"]:
        event = process_tracking_event(tracking_event=tracking_event, game_id=game_id)
        if event is not None:
            events += [event]

    return events


def filter_events(events: list[Event], config: PreprocessingConfig):
    filtered_events = []
    for event in events:
        if (
            len(event.frames) >= config.event_duration * config.target_frame_rate
            and abs(
                (1000.0 / config.target_frame_rate)
                - ((event.wall_clock_end - event.wall_clock_start) / len(event.frames))
            )
            < 0.8
        ):
            filtered_events += [event]

    return filtered_events


def downsample_events(events: list[Event], target_frame_rate: int, training_frame_rate: int):
    factor = int(target_frame_rate / training_frame_rate)
    for event in events:
        event.frames = event.frames[::factor]

    return events


def split_list_into_parts(input_list: list, n: int):
    part_size = len(input_list) // n
    remainder = len(input_list) % n
    return [
        input_list[i * part_size + min(i, remainder) : (i + 1) * part_size + min(i + 1, remainder)] for i in range(n)
    ]


def split_long_events(events: list[Event], duration: int):
    split_events = []
    for event in events:
        split_frames = split_list_into_parts(input_list=event.frames, n=floor(len(event.frames) / duration))
        if len(split_frames) > 1:
            for i, frames in enumerate(split_frames):
                new_event = copy.deepcopy(event)
                new_event.frames = frames
                new_event.event_id = (new_event.event_id * 1000) + i
                split_events += [new_event]
        else:
            split_events += [event]

    return split_events


def get_player_coordinates_tensor_from_frames(player_id: int, frames: list[Frame]):
    if player_id == -1:
        # ball
        return torch.tensor([frame.ball.to_row() for frame in frames])
    else:
        # player
        return torch.tensor([frame.players[player_id].to_row() for frame in frames])


def convert_event_to_tensors(events: list[Event]):
    tensors = {}
    for event in events:
        try:
            file_name = f"{event.game_id}_{event.event_id}"

            list_of_tensors = [get_player_coordinates_tensor_from_frames(player_id=-1, frames=event.frames)]

            for player_id in event.team_0_players:
                list_of_tensors += [get_player_coordinates_tensor_from_frames(player_id=player_id, frames=event.frames)]

            for player_id in event.team_1_players:
                list_of_tensors += [get_player_coordinates_tensor_from_frames(player_id=player_id, frames=event.frames)]

            tensors[file_name] = torch.stack(list_of_tensors, dim=0)
        except Exception as e:
            print(f"Error: {file_name}")
            print(e)

    return tensors


def preprocess_tracking_data(config: PreprocessingConfig):
    game_ids = list_files_in_directory(path=config.raw_tracking_path, suffix=".json")

    for game_id in game_ids:
        events = load_game(path=config.raw_tracking_path, game_id=game_id)

        events = filter_events(events=events, config=config)
        events = downsample_events(
            events=events,
            target_frame_rate=config.target_frame_rate,
            training_frame_rate=config.training_frame_rate,
        )
        if config.split_long_events:
            events = split_long_events(events=events, duration=config.event_duration * config.training_frame_rate)

        tensors = convert_event_to_tensors(events=events)

        os.makedirs(config.tensor_path, exist_ok=True)
        for filename, tensor in tensors.items():
            torch.save(tensor, f"{config.tensor_path}/{filename}.pt")
