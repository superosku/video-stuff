import dataclasses
import random


@dataclasses.dataclass
class PourInstruction:
    from_: int
    to: int
    pour_amount: int
    pour_color: int
    destination_empty: int


HashablePuzzleState = tuple[tuple[int, int, int, int]]


class WaterPuzzleState:
    pipes: list[list[int]]
    distance: int

    def __init__(self, pipes: list[list[int]], distance:int = 0):
        self.pipes = pipes
        self.distance = distance

    @classmethod
    def from_hashable_state(cls, hashable_state: HashablePuzzleState) -> "WaterPuzzleState":
        pipes = [
            list(p)
            for p in hashable_state
        ]
        return cls(pipes)

    @classmethod
    def new_random(cls, num_colors=4, random_seed: int | None = None) -> "WaterPuzzleState":
        all_colors = sum([
            [i + 1 for i in range(num_colors)]
            for _ in range(4)
        ], [])
        if random_seed is not None:
            random.seed(random_seed)
        random.shuffle(all_colors)
        pipes = [
            [all_colors[i], all_colors[i+1], all_colors[i+2], all_colors[i+3]]
            for i in range(0, len(all_colors), 4)
        ] + [[0, 0, 0, 0]] + [[0, 0, 0, 0]]

        return cls(pipes)

    def hashable(self) -> HashablePuzzleState:
        tuples = [tuple(p) for p in self.pipes]
        return tuple(reversed(sorted(tuples)))  # Need to sort since the ordering does not really matter

    def possible_options(self) -> list[tuple[PourInstruction, "Self"]]:
        possible_options: list[tuple[PourInstruction, WaterPuzzleState]] = []
        for i in range(len(self.pipes)):
            for j in range(len(self.pipes)):
                if i == j:
                    continue
                pour_from = self.pipes[i]
                pour_to = self.pipes[j]

                # Can not pour from empty
                if all(c == 0 for c in pour_from):
                    continue
                # Can not pour to full
                if pour_to[3] != 0:
                    continue
                empty_spots_destination = len([None for c in pour_to if c == 0])
                empty_spots_source = len([None for c in pour_from if c == 0])
                pour_color = pour_from[-1 - empty_spots_source]

                destination_color = None
                if empty_spots_destination != 4:
                    destination_color = pour_to[-1 - empty_spots_destination]

                # Can not pour to something other than own color
                if destination_color is not None and destination_color != pour_color:
                    continue

                same_color_stacked_in_source = 0
                for c in reversed(pour_from):
                    if c == 0:
                        continue
                    if c == pour_color:
                        same_color_stacked_in_source += 1
                    else:
                        break

                how_many_will_be_poured = min(same_color_stacked_in_source, empty_spots_destination)
                new_pipes = [list(p) for p in self.pipes]
                for pour_n in range(how_many_will_be_poured):
                    new_pipes[i][-1 - pour_n - empty_spots_source] = 0
                    new_pipes[j][4 - empty_spots_destination + pour_n] = pour_color

                possible_options.append((
                    PourInstruction(
                        from_=i,
                        to=j,
                        pour_amount=how_many_will_be_poured,
                        pour_color=pour_color,
                        destination_empty=empty_spots_destination,
                    ),
                    WaterPuzzleState(new_pipes, distance=self.distance + 1)
                ))
        return possible_options

    def is_solved(self):
        all_rows_have_same_color = all([len(set(pipe)) <= 1 for pipe in self.pipes])
        all_rows_have_len_4_or_0 = all([len([p for p in pipe if p != 0]) in [0, 4] for pipe in self.pipes])
        return all_rows_have_same_color and all_rows_have_len_4_or_0

    def print(self, indent=0):
        print(f"{' ' * indent}PIPES:")
        for row in self.pipes:
            print(f"{' ' * indent}{row}")


class WaterPuzzleSolver:
    solve_instructions: list[PourInstruction]
    nodes: set[HashablePuzzleState]
    edges: set[tuple[HashablePuzzleState, HashablePuzzleState]]

    def __init__(self, initial_state: WaterPuzzleState):
        self.initial_state = initial_state

    def solve(self) -> None:
        queue = [self.initial_state]
        visited = {self.initial_state.hashable()}
        prev_nexts = {}

        solved_state = None
        self.nodes: set[HashablePuzzleState] = set()
        self.edges: set[tuple[HashablePuzzleState, HashablePuzzleState]] = set()
        self.solve_instructions = []

        self.hashable_to_original_unsorted = {}
        self.distance_to_hashables = {}

        if self.initial_state.is_solved():
            return

        while True:
            if len(queue) == 0:
                if solved_state is not None:
                    pass
                    #print("Solved!")
                else:
                    pass
                    #print("No solution found")
                break
            current_state = queue.pop(0)
            if current_state.hashable() not in self.hashable_to_original_unsorted:
                self.hashable_to_original_unsorted[current_state.hashable()] = current_state
            self.nodes.add(current_state.hashable())
            if current_state.distance not in self.distance_to_hashables:
                self.distance_to_hashables[current_state.distance] = set()
            self.distance_to_hashables[current_state.distance].add(current_state.hashable())
            if current_state.is_solved():
                solved_state = current_state
            for from_to, option in current_state.possible_options():
                self.edges.add((current_state.hashable(), option.hashable()))
                if option.hashable() not in visited:
                    visited.add(option.hashable())
                    queue.append(option)
                    prev_nexts[option.hashable()] = (current_state.hashable(), from_to)

        if solved_state is None:
            return

        backstep_state = solved_state.hashable()
        steps = []
        while True:
            try:
                backstep_state, from_to = prev_nexts[backstep_state]
            except KeyError:
                # Is not solvable (perhaps already solved?)
                return
            steps.append(from_to)
            if backstep_state == self.initial_state.hashable():
                break

        self.solve_instructions = list(reversed(steps))
