// Import rand
extern crate rand;
// Import shuffle
use rand::seq::SliceRandom;
// Import hash set
use std::collections::HashSet;
// Import serialize from serde (or serde_json?)
use serde::{Deserialize, Serialize};
use serde_json::Result;

use std::fs::OpenOptions;
use std::io::prelude::*;

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
    winnable_nodes: HashSet<HashableState>,
    moves_to_reach_winnable: i64,
}

impl WaterSortSearcher {
    pub fn new_random(num_colors: u32) -> WaterSortSearcher {
        let mut is_solvable = false;
        let puzzle = WaterSortState::new_random(num_colors);

        let mut seen_states = std::collections::HashSet::new();
        seen_states.insert(puzzle.hashable());

        let mut nodes = HashSet::new();
        let mut edges = HashSet::new();
        // solved_state is an optional WaterSortState that is empty/null/none by default
        let mut solved_state = None;

        // A mapping of a node to its distance from the start node
        let mut node_distances = std::collections::HashMap::new();
        node_distances.insert(puzzle.hashable(), 0);

        {
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(puzzle.clone());
            nodes.insert(puzzle.hashable());

            while let Some(state) = queue.pop_front() {
                let state_hashable = state.hashable();
                let distance_of_popped = node_distances.get(&state.hashable()).unwrap().clone();

                for mov in state.possible_moves() {
                    // Add node distance to a dict if not yet there
                    if !node_distances.contains_key(&mov.hashable()) {
                        node_distances.insert(mov.hashable(), distance_of_popped + 1);
                    }
                    // Add nodes and edges to the sets
                    nodes.insert(mov.hashable());
                    edges.insert((state_hashable.clone(), mov.hashable()));
                    if seen_states.contains(&mov.hashable()) {
                        continue;
                    }
                    if mov.is_solved() {
                        is_solvable = true;
                        solved_state = Some(mov.clone());
                    }
                    seen_states.insert(mov.hashable());
                    queue.push_back(mov);
                }
            }
        }

        let mut node_to_parents = std::collections::HashMap::new();
        for (from, to) in &edges {
            let parents = node_to_parents.entry(to.clone()).or_insert(Vec::new());
            parents.push(from.clone());
        }

        let mut winnable_nodes = HashSet::new();
        {
            let mut queue = std::collections::VecDeque::new();
            if let Some(solved_state) = solved_state.clone() {
                queue.push_back(solved_state.hashable());
            }
            while let Some(state) = queue.pop_front() {
                winnable_nodes.insert(state.clone());
                if let Some(parents) = node_to_parents.get(&state) {
                    for parent in parents {
                        if !winnable_nodes.contains(parent) {
                            winnable_nodes.insert(parent.clone()); // TODO: Should do this here or after popping or both?
                            queue.push_back(parent.clone());
                        }
                    }
                }
            }
        }

        let mut moves_to_reach_winnable = match solved_state {
            Some(solved_state) => *node_distances.get(&solved_state.hashable()).unwrap() as i64,
            None => -1,
        };

        WaterSortSearcher{
            puzzle,
            is_solvable,
            nodes,
            edges,
            winnable_nodes,
            moves_to_reach_winnable
        }
    }
}

// Derive serialize
#[derive(Serialize, Deserialize)]
struct JsonLineOutputSingle {
    pipes: Vec<Vec<u32>>,
    solvable: bool,
    nodes: usize,
    winnable_nodes: usize,
    edges: usize,
    distinct_color_parts: usize,
    random_move_probability_to_winnable: f64,
    winnable_nodes_vs_nodes: f64,
    moves_to_reach_winnable: i64,
}

#[derive(Serialize, Deserialize)]
struct JsonLineOutput {
    size: u32,
    puzzles: Vec<JsonLineOutputSingle>
}

fn main() {
    let sample_size = 2000;

    let file_name = "output3.json";

    // Clear out a file called output.json
    std::fs::write(file_name, "").expect("Unable to write file");

    let size = 8;
    {
    // for size in 4..50 {
        let mut solvable_count = 0;

        println!("Size: {}", size);

        let mut puzzles = Vec::new();

        // sample_size_real is sample_size if size is 10, otherwise it is 1000
        // let sample_size_real = if size == 10 { 1000 } else { sample_size };
        let sample_size_real = sample_size;

        for _ in 0..sample_size_real {
            let state = WaterSortSearcher::new_random(size);

            if state.is_solvable {
                solvable_count += 1;
            }

            // Count the distinct color parts of each pipe
            // So for an example:
            // [1, 1, 2, 2] -> 2
            // [1, 2, 3, 4] -> 4
            // [1, 1, 1, 1] -> 1
            // [0, 0, 0, 0] -> 0  // Empty pipe
            let mut distinct_color_parts = 0;
            for tube in &state.puzzle.tubes {
                let mut distinct_colors = HashSet::new();
                for color in tube {
                    if *color != 0 {
                        distinct_colors.insert(*color);
                    }
                }
                distinct_color_parts += distinct_colors.len();
            }

            // Calculate the probability of a random move leading to a winnable state
            let mut all_moves = 0;
            let mut winnable_moves = 0;
            // Construct a map from node into all of its childrens
            let mut node_to_children = std::collections::HashMap::new();
            for node in &state.nodes {
                // children is a vector of edges
                let children: Vec<HashableState> = state.edges.iter().filter(|(from, _)| from == node).map(|(_, to)| to.clone()).collect();
                node_to_children.insert(node.clone(), children);
            }
            // Iterate all of the nodes and the childrens (node_to_children)
            for (node, children) in &node_to_children {
                for child in children {
                    all_moves += 1;
                    if state.winnable_nodes.contains(child) {
                        winnable_moves += 1;
                    }
                }
            }

            let random_move_probability_to_winnable: f64  = winnable_moves as f64 / all_moves as f64;
            let winnable_nodes_vs_nodes = state.winnable_nodes.len() as f64 / state.nodes.len() as f64;

            let json_line_output_single = JsonLineOutputSingle{
                pipes: state.puzzle.tubes.clone().iter().map(|x| x.to_vec()).collect(),
                solvable: state.is_solvable,
                nodes: state.nodes.len(),
                edges: state.edges.len(),
                winnable_nodes: state.winnable_nodes.len(),
                distinct_color_parts,
                random_move_probability_to_winnable,
                winnable_nodes_vs_nodes,
                moves_to_reach_winnable: state.moves_to_reach_winnable,
            };
            puzzles.push(json_line_output_single);

            println!(
                "  Nodes: {}, Edges: {}, Winnable nodes: {}, Distinct color parts: {}, Random move probability to winnable: {}, Nodes vs winnables: {}, Moves to win: {}",
                state.nodes.len(), state.edges.len(), state.winnable_nodes.len(), distinct_color_parts,
                random_move_probability_to_winnable, winnable_nodes_vs_nodes, state.moves_to_reach_winnable
            );
        }

        let json_line_output = JsonLineOutput{
            size,
            puzzles,
        };

        // Serialize and append to the file
        let serialized = serde_json::to_string(&json_line_output).unwrap();
        // std::fs::write("output.json", serialized + "\n").expect("Unable to write file");
        // Append to the file (do not overwrite it)

        let mut file = OpenOptions::new()
            .write(true)
            .append(true)
            .open(file_name)
            .unwrap();

        writeln!(file, "{}", serialized).unwrap();

        println!("  Solvable: {}/{}", solvable_count, sample_size);
    }
}
