// Import rand
extern crate rand;
// Import shuffle
use rand::seq::SliceRandom;
// Import hash set
use std::collections::HashSet;
// Import serialize from serde (or serde_json?)
use serde::{Deserialize, Serialize};
use serde_json::Result;

type HashableState = Vec<[u32; 4]>;

#[derive(Clone, Debug)]
struct WaterSortState {
    // Tubes are vector of 4 long tuples unsigned integers
    // tubes: Vec<(u32, u32, u32, u32)>,
    // Tubes are vector of 4 long lists unsigned integers
    tubes: Vec<[u32; 4]>,
}

impl WaterSortState {
    pub fn new_random(num_colors: u32) -> WaterSortState {
        // First construct a vector of num_colors * 4 length that has numbers such as this:
        // 1,1,1,1,2,2,2,2,3,3,3,3 etc.
        let mut colors = Vec::new();
        for i in 0..num_colors {
            for _ in 0..4 {
                colors.push(i + 1);
            }
        }

        // Then random shuffle that vector
        let mut rng = rand::thread_rng();
        colors.shuffle(&mut rng);

        // Construct the tubes
        let mut tubes = Vec::new();
        for _ in 0..num_colors {
            let mut tube = [
                colors.pop().unwrap(),
                colors.pop().unwrap(),
                colors.pop().unwrap(),
                colors.pop().unwrap(),
            ];
            tubes.push(tube);
        }
        tubes.push([0, 0, 0, 0]);
        tubes.push([0, 0, 0, 0]);
        WaterSortState { tubes }
    }

    // Produce a hashable representation of the state
    // Should be same as the tubes but the tubes should be sorted to avoid duplicates
    // Not values within the tubes but the tubes themselves
    pub fn hashable(&self) -> HashableState {
        let mut sorted_tubes = self.tubes.clone();
        sorted_tubes.sort();
        sorted_tubes
    }

    pub fn is_solved(&self) -> bool {
        for tube in &self.tubes {
            if tube[0] != tube[1] || tube[1] != tube[2] || tube[2] != tube[3] {
                return false;
            }
        }
        true
    }

    pub fn possible_moves(&self) -> Vec<WaterSortState> {
        let mut moves = Vec::new();
        for from_i in 0..self.tubes.len() {
            for to_i in 0..self.tubes.len() {
                if from_i != to_i {
                    let tube_from = &self.tubes[from_i];
                    let tube_to = &self.tubes[to_i];
                    let tube_to_empty_count = tube_to.iter().filter(|&x| *x == 0).count();
                    if tube_to_empty_count == 0 {
                        continue;
                    }
                    let tube_from_empty_count = tube_from.iter().filter(|&x| *x == 0).count();
                    if tube_from_empty_count == 4 {
                        continue;
                    }
                    let from_color = tube_from[tube_from_empty_count];
                    if tube_to_empty_count < 4 {
                        let to_color = tube_to[tube_to_empty_count];
                        if to_color != from_color {
                            continue;
                        }
                    }
                    // Pour amount is how many of the same color the from tube has at top
                    let mut pour_amount = 1;
                    for i in tube_from_empty_count + 1..4 {
                        if tube_from[i] == from_color {
                            pour_amount += 1;
                        } else {
                            break;
                        }
                    }
                    // Pour amount caps at the free space in the to tube
                    pour_amount = pour_amount.min(tube_to_empty_count);
                    let mut new_tube = self.clone();

                    // Pour the water
                    for i in 0..pour_amount {
                        new_tube.tubes[to_i][tube_to_empty_count - i - 1] = from_color;
                        new_tube.tubes[from_i][tube_from_empty_count + i] = 0;
                    }

                    moves.push(new_tube);
                }
            }
        }
        moves
    }
}

struct WaterSortSearcher {
    puzzle: WaterSortState,
    nodes: HashSet<HashableState>,
    edges: HashSet<(HashableState, HashableState)>,
    is_solvable: bool,
}

impl WaterSortSearcher {
    pub fn new_random(num_colors: u32) -> WaterSortSearcher {
        let mut is_solvable = false;
        let puzzle = WaterSortState::new_random(num_colors);

        let mut seen_states = std::collections::HashSet::new();
        seen_states.insert(puzzle.hashable());

        let mut queue = std::collections::VecDeque::new();
        queue.push_back(puzzle.clone());

        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();

        while let Some(state) = queue.pop_front() {
            for mov in state.possible_moves() {
                nodes.insert(mov.hashable());
                edges.insert((state.hashable(), mov.hashable()));
                if seen_states.contains(&mov.hashable()) {
                    continue;
                }
                if mov.is_solved() {
                    is_solvable = true;
                    // break
                }
                seen_states.insert(mov.hashable());
                queue.push_back(mov);
            }
        }

        WaterSortSearcher{
            puzzle,
            is_solvable,
            nodes,
            edges,
        }
    }
}

// Derive serialize
#[derive(Serialize, Deserialize)]
struct JsonLineOutputSingle {
    pipes: Vec<Vec<u32>>,
    solvable: bool,
    nodes: usize,
    edges: usize,
}

#[derive(Serialize, Deserialize)]
struct JsonLineOutput {
    size: u32,
    puzzles: Vec<JsonLineOutputSingle>
}

fn main() {
    let sample_size = 100;

    // Clear out a file called output.json
    std::fs::write("output.json", "").expect("Unable to write file");

    for size in 4..50 {
        let mut solvable_count = 0;

        println!("Size: {}", size);

        let mut puzzles = Vec::new();

        // sample_size_real is sample_size if size is 10, otherwise it is 1000
        let sample_size_real = if size == 10 { 1000 } else { sample_size };

        for _ in 0..sample_size_real {
            let state = WaterSortSearcher::new_random(size);

            println!("  Nodes: {}, Edges: {}", state.nodes.len(), state.edges.len());

            if state.is_solvable {
                solvable_count += 1;
            }

            let json_line_output_single = JsonLineOutputSingle{
                pipes: state.puzzle.tubes.clone().iter().map(|x| x.to_vec()).collect(),
                solvable: state.is_solvable,
                nodes: state.nodes.len(),
                edges: state.edges.len(),
            };
            puzzles.push(json_line_output_single);
        }

        let json_line_output = JsonLineOutput{
            size,
            puzzles,
        };

        // Serialize and append to the file
        let serialized = serde_json::to_string(&json_line_output).unwrap();
        std::fs::write("output.json", serialized + "\n").expect("Unable to write file");

        println!("  Solvable: {}/{}", solvable_count, sample_size);
    }
}
