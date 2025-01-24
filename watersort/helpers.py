import dataclasses
import random


@dataclasses.dataclass
class PourInstruction:
    from_: int
    to: int
    pour_amount: int
    pour_color: int
    destination_empty: int


HashablePuzzleState = tuple[tuple[int, int, int, int], ...]


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

    def get_winnable_nodes(self) -> list[HashablePuzzleState]:
        winning_node = [e for e in self.nodes if all(len(set(f)) == 1 for f in e)][0]

        nodes_that_can_win = [winning_node]

        edges_by_e2 = {}
        for e1, e2 in self.edges:
            if e2 not in edges_by_e2:
                edges_by_e2[e2] = set()
            edges_by_e2[e2].add(e1)

        i = 0
        while i < len(nodes_that_can_win):
            node = nodes_that_can_win[i]
            for node in edges_by_e2.get(node, []):
                if node not in nodes_that_can_win:
                    nodes_that_can_win.append(node)

            i += 1

        return nodes_that_can_win

    def get_pour_instructions_into(self, backstep_state: HashablePuzzleState) -> list[PourInstruction]:
        steps = []
        while True:
            try:
                backstep_state, from_to = self.prev_nexts[backstep_state]
            except KeyError:
                assert 0  # TODO: Should this ever happen?
                # Is not solvable (perhaps already solved?)
                return []
            steps.append(from_to)
            if backstep_state == self.initial_state.hashable():
                break

        return list(reversed(steps))

    def winning_node(self) -> HashablePuzzleState:
        return [n for n in self.nodes if all([len(set(a)) == 1 for a in n])][0]

    def get_all_nodes_from_start_to(self, to: HashablePuzzleState) -> list[HashablePuzzleState]:
        nodes = []
        current = to
        while True:
            nodes.append(current)
            if current == self.initial_state.hashable():
                break
            distance_at_current = [(k, v) for k, v in self.distance_to_hashables.items() if current in v][0][0]
            possible_next_nodes = list(self.distance_to_hashables[distance_at_current - 1])
            next_node = [n for n in possible_next_nodes if (n, current) in self.edges][0]
            current = next_node

        return nodes

    def solve(self) -> None:
        queue = [self.initial_state]
        visited = {self.initial_state.hashable()}
        self.prev_nexts = {}

        solved_state = None
        self.nodes: set[HashablePuzzleState] = set()
        self.nodes_and_distance_that_have_no_moves: list[tuple[int, PourInstruction]] = []
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
            possible_options = current_state.possible_options()
            if len(possible_options) == 0:
                self.nodes_and_distance_that_have_no_moves.append((current_state.distance, current_state))

            for from_to, option in possible_options:
                self.edges.add((current_state.hashable(), option.hashable()))
                if option.hashable() not in visited:
                    visited.add(option.hashable())
                    queue.append(option)
                    self.prev_nexts[option.hashable()] = (current_state.hashable(), from_to)

        if solved_state is None:
            return

        backstep_state = solved_state.hashable()

        self.solve_instructions = self.get_pour_instructions_into(backstep_state)
