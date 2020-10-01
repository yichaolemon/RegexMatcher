use crate::parser::{CharacterClass, Regex, Boundary};
use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::hash::Hash;

#[derive(Debug, Clone)]
pub struct Node<T, U> {
  id: T,
  // edges are character classes
  transitions: Vec<(U, T)>,
}

#[derive(Debug, Clone, Default)]
pub struct Graph<T, U> {
  root: T,
  terminals: HashSet<T>,
  map: HashMap<T, Node<T, U>>,
}

#[derive(Debug, Clone)]
pub enum NfaTransition {
  Empty, // no op
  Character(CharacterClass),
  Boundary(Boundary),
}

#[derive(Debug, Clone, Eq)]
struct DfaTransition(CharacterClass);

/// reindexing the nodes when merging two graphs
fn graph_reindex<T: Hash + Eq, U, F: FnMut(&T) -> T>(graph: Graph<T, U>, mut f: F) -> Graph<T, U> {
  let mut map = HashMap::new();
  for (id, node) in graph.map.into_iter() {
    map.insert(f(&id), Node {
      id: f(&node.id),
      transitions: node.transitions.into_iter().map(|(u, t)| (u, f(&t))).collect(),
    });
  }
  Graph {
    root: f(&graph.root),
    terminals: graph.terminals.iter().map(f).collect(),
    map,
  }
}

fn nfa_with_one_transition(t: NfaTransition) -> Graph<i32, NfaTransition> {
  Graph{
    root: 0,
    terminals: hashset!{1},
    map: hashmap!{
      0 => Node{id: 0, transitions: vec!((t, 1))},
      1 => Node{id: 1, transitions: vec!()},
    },
  }
}

impl From<&Regex> for Graph<i32, NfaTransition> {
  fn from(r: &Regex) -> Self {
    return build_nfa(r)
  }
}

impl CharacterClass {
  pub fn example(&self) -> char {
    match self {
      CharacterClass::Char(c) => *c,
      CharacterClass::Any => '/',
      CharacterClass::Word => 'z',
      CharacterClass::Whitespace => ' ',
      CharacterClass::Digit => '7',
      CharacterClass::Negation(_) => '!',  // This is wrong but it's okay because the correct thing is hard.
      CharacterClass::Union(cc1, _) => cc1.example(),
      CharacterClass::Range(a, _) => *a,
    }
  }
}

trait Transition {
  fn example(&self, s: &String) -> String;
}

impl Transition for NfaTransition {
  fn example(&self, s: &String) -> String {
    match self {
      NfaTransition::Empty => s.clone(),
      NfaTransition::Character(cc) => format!("{}{}", s, cc.example()),
      NfaTransition::Boundary(_) => s.clone(),  // This is wrong but it's okay.
    }
  }
}

impl Transition for DfaTransition {
  fn example(&self, s: &String) -> String {
    format!("{}{}", s, self.0.example())
  }
}

impl<T: Eq + Hash, U: Transition> Graph<T, U> {
  pub fn example(&self) -> Option<String> {
    let mut queue = VecDeque::new();
    queue.push_back((&self.root, String::new()));
    loop {
      let (id, s) = queue.pop_front()?;
      if self.terminals.contains(id) {
        return Some(s)
      }
      for (transition, dest) in self.map.get(id)?.transitions.iter() {
        queue.push_back((dest, transition.example(&s)));
      }
    }
  }
}

/// construct non-deterministic finite automata from the parsed Regex
pub fn build_nfa(regex: &Regex) -> Graph<i32, NfaTransition> {
  match regex {
    Regex::Alternative(left, right) => {
      // Merge subgraphs, and create a new root with empty transitions to both subgraph-roots.
      let mut left_nfa = build_nfa(left);
      let right_nfa = graph_reindex(
        build_nfa(right),
        |id| *id + (left_nfa.map.len() as i32),
      );
      left_nfa.map.extend(right_nfa.map);
      left_nfa.terminals.extend(right_nfa.terminals);
      left_nfa.root = left_nfa.map.len() as i32;
      left_nfa.map.insert(left_nfa.root, Node{
        id: left_nfa.root,
        transitions: vec!((NfaTransition::Empty, left_nfa.root), (NfaTransition::Empty, right_nfa.root)),
      });
      left_nfa
    },
    Regex::Concat(left, right) => {
      // Merge subgraphs, and connect all terminals of subgraph 1 to the root of subgraph 2.
      let mut left_nfa = build_nfa(left);
      let right_nfa = graph_reindex(
        build_nfa(right),
        |id| *id + (left_nfa.map.len() as i32),
      );
      for terminal in left_nfa.terminals.into_iter() {
        left_nfa.map.get_mut(&terminal).unwrap()
          .transitions.push((NfaTransition::Empty, right_nfa.root));
      }
      left_nfa.map.extend(right_nfa.map);
      left_nfa.terminals = right_nfa.terminals;
      left_nfa
    },
    Regex::Optional(r) => {
      let mut nfa = build_nfa(r);
      nfa.terminals.insert(nfa.root);
      nfa
    },
    Regex::Plus(r) => {
      let mut nfa = build_nfa(r);
      for terminal in nfa.terminals.iter() {
        nfa.map.get_mut(terminal).unwrap()
          .transitions.push((NfaTransition::Empty, nfa.root));
      }
      nfa
    },
    Regex::Kleene(r) => {
      let mut nfa = build_nfa(r);
      for terminal in nfa.terminals.iter() {
        nfa.map.get_mut(terminal).unwrap()
          .transitions.push((NfaTransition::Empty, nfa.root));
      }
      nfa.terminals.insert(nfa.root);
      nfa
    },
    Regex::Boundary(b) => {
      nfa_with_one_transition(NfaTransition::Boundary(b.clone()))
    },
    Regex::Class(cc) => {
      nfa_with_one_transition(NfaTransition::Character(cc.clone()))
    },
    Regex::Group(r, _) => {
      build_nfa(r)
    },
    Regex::Char(c) => {
      nfa_with_one_transition(NfaTransition::Character(CharacterClass::Char(*c)))
    },
  }
}

/// find all nodes in the graph that are \epsilon away from given node set, mutate in place
fn bfs_epsilon<T, U, F: FnMut(&U) -> bool>(mut nodes: &BTreeSet<T>, graph: &Graph<T, U>, mut f: F) {
  let mut queue = VecDeque::new();
  nodes.iter().for_each(|node| queue.push_back(node));

  while !queue.is_empty() {
    let i = queue.pop_front().unwrap();
    let transitions = &graph.map.get(i).unwrap().transitions.iter()
      .filter_map(|(cc, j)| if f(cc) {Some(j)} else {None}).collect();
    for j in transitions {
      if !nodes.contains(j) {
        queue.push_back(j);
        nodes.insert(j);
      }
    }
  }
}

/// given a non-deterministic finite automata, construct its equivalent deterministic finite automata
pub fn nfa_to_dfa<T>(nfa: Graph<T, NfaTransition>) -> Graph<BTreeSet<T>, DfaTransition> {
  let mut iter_set: BTreeSet<T> = BTreeSet::new();
  iter_set.insert(nfa.root);
  let mut stack = VecDeque::new();
  stack.push_back(iter_set);
  let mut boo = true;

  let mut dfa = Graph::default();

  while !stack.is_empty() {
    iter_set = stack.pop_front().unwrap();
    // merge all the nodes that are \epsilon away
    bfs_epsilon(&iter_set, &nfa, |cc| cc == NfaTransition::Empty);
    if boo { dfa.root = iter_set; boo = false; }
    if dfa.map.contains_key(&iter_set) {
      continue
    }
    // construct the new edges
    let edges: Vec<(NfaTransition, T)> = iter_set.iter().flat_map(|i| nfa.map.get(i).unwrap().transitions.iter()
      .filter(|(cc, j)| !iter_set.contains(j))
      .collect())
      .collect();
    let mut new_edges = Vec::new();
    for (cc, i) in edges.iter() {
      new_edges.push(())
    }
    let new_node = Node {
      id: iter_set,
      transitions: new_edges,
    };
  }

  None
}

// returns cc1 intersect cc2 and cc1 setminus cc2
fn intersect_transitions(cc1: &DfaTransition, cc2: &DfaTransition) -> DfaTransition {

}

// returns the NFATransition that
fn subtract_intersection(cc1: &DfaTransition, cc2: &DfaTransition) -> NfaTransition {

}

fn cover_transitions<T>(transitions: Vec<(NfaTransition, T)>) -> HashMap<DfaTransition, BTreeSet<T>> {
}

trait MathSet: Eq + Clone {
  fn intersect(&self, other: &Self) -> Self;
  fn setminus(&self, other: &Self) -> Self;
  fn is_empty(&self) -> bool;
}

impl MathSet for DfaTransition {
  fn intersect(&self, other: &Self) -> Self {
    unimplemented!()
  }

  fn setminus(&self, other: &Self) -> Self {
    unimplemented!()
  }

  fn is_empty(&self) -> bool {
    unimplemented!()
  }
  // TODO
}

// Given some mathematical sets (given as id->set), find all intersections
fn set_covering<T, S: MathSet>(sets: HashMap<T, S>) -> HashMap<BTreeSet<T>, S> {
  let mut result = HashMap::new();
  let mut to_process = VecDeque::new();

  for (i, input_set) in sets.iter() {
    let mut ids = BTreeSet::new();
    ids.insert(i);
    to_process.push_back((input_set, ids));
  }

  while !to_process.is_empty() {
    let (s, ids) = to_process.pop_front();
    let mut leftover = s.clone();
    for (s2, ids2) in to_process.iter() {
      let intersection = s.intersect(s2);
      if !intersection.is_empty() {
        to_process.push_back((intersection, ids.union(ids2)));
      }
      leftover = leftover.setminus(s2);
    }
    if !leftover.is_empty() {
      result.insert(ids, leftover);
    }
  }
  result
}
