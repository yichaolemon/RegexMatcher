use crate::graph::{Graph, Node, EdgeLabel, Transition};
use std::hash::Hash;
use std::collections::{HashMap, BTreeSet, VecDeque, HashSet};
use crate::parser::{GroupId, Regex, Boundary};
use std::fmt::{Display, Formatter};
use std::fmt;
use crate::character_class::CharacterClass;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NfaTransition {
  Empty, // no op
  Character(CharacterClass),
  Boundary(Boundary),
}

impl Default for NfaTransition {
  fn default() -> Self {
    NfaTransition::Empty
  }
}

/// reindexing the nodes when merging two graphs
fn graph_reindex<T: Hash + Eq, U, F: FnMut(&T) -> T>(graph: Graph<T, U>, mut f: F) -> Graph<T, U> {
  let mut map = HashMap::new();
  for (id, node) in graph.map.into_iter() {
    map.insert(f(&id), Node {
      id: f(&node.id),
      transitions: node.transitions.into_iter().map(|(u, t)| (u, f(&t))).collect(),
      groups: node.groups,
    });
  }
  Graph {
    root: f(&graph.root),
    terminals: graph.terminals.iter().map(f).collect(),
    map,
  }
}

fn nfa_with_one_transition(t: NfaTransition, groups: BTreeSet<GroupId>) -> Graph<i32, NfaTransition> {
  Graph{
    root: 0,
    terminals: hashset!{1},
    map: hashmap!{
      0 => Node{id: 0, transitions: vec!((t, 1)), groups: groups.clone()},
      1 => Node{id: 1, transitions: vec!(), groups: groups},
    },
  }
}

impl Display for NfaTransition {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    match self {
      NfaTransition::Empty => write!(f, "ε"),
      NfaTransition::Character(cc) => write!(f, "{}", cc),
      NfaTransition::Boundary(b) => write!(f, "{}", b),
    }
  }
}

impl EdgeLabel for NfaTransition {
  fn display(&self) -> String {
    match self {
      NfaTransition::Empty => format!("ε"),
      NfaTransition::Character(cc) => cc.canonical_form().display(),
      NfaTransition::Boundary(b) => b.display(),
    }
  }
}

impl From<&Regex> for Graph<i32, NfaTransition> {
  fn from(r: &Regex) -> Self {
    return build_nfa(r, BTreeSet::new())
  }
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

/// construct non-deterministic finite automata from the parsed Regex
pub fn build_nfa(regex: &Regex, mut groups: BTreeSet<GroupId>) -> Graph<i32, NfaTransition> {
  match regex {
    Regex::Alternative(left, right) => {
      // Merge subgraphs, and create a new root with empty transitions to both subgraph-roots.
      let mut left_nfa = build_nfa(left, groups.clone());
      let right_nfa = graph_reindex(
        build_nfa(right, groups.clone()),
        |id| *id + (left_nfa.map.len() as i32),
      );
      left_nfa.map.extend(right_nfa.map);
      left_nfa.terminals.extend(right_nfa.terminals);
      let tmp_left_root = left_nfa.root;
      left_nfa.root = left_nfa.map.len() as i32;
      left_nfa.map.insert(left_nfa.root, Node{
        id: left_nfa.root,
        transitions: vec!((NfaTransition::Empty, tmp_left_root), (NfaTransition::Empty, right_nfa.root)),
        groups,
      });
      left_nfa
    },
    Regex::Concat(left, right) => {
      // Merge subgraphs, and connect all terminals of subgraph 1 to the root of subgraph 2.
      let mut left_nfa = build_nfa(left, groups.clone());
      let right_nfa = graph_reindex(
        build_nfa(right, groups),
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
      let mut nfa = build_nfa(r, groups);
      nfa.terminals.insert(nfa.root);
      nfa
    },
    Regex::Plus(r) => {
      let mut nfa = build_nfa(r, groups);
      for terminal in nfa.terminals.iter() {
        nfa.map.get_mut(terminal).unwrap()
          .transitions.push((NfaTransition::Empty, nfa.root));
      }
      nfa
    },
    Regex::Kleene(r) => {
      let mut nfa = build_nfa(r, groups);
      for terminal in nfa.terminals.iter() {
        nfa.map.get_mut(terminal).unwrap()
          .transitions.push((NfaTransition::Empty, nfa.root));
      }
      nfa.terminals.insert(nfa.root);
      nfa
    },
    Regex::Boundary(b) => {
      nfa_with_one_transition(NfaTransition::Boundary(b.clone()), groups)
    },
    Regex::Class(cc) => {
      nfa_with_one_transition(NfaTransition::Character(cc.clone()), groups)
    },
    Regex::Group(r, group_id) => {
      groups.insert(*group_id);
      build_nfa(r, groups)
    },
    Regex::Char(c) => {
      nfa_with_one_transition(NfaTransition::Character(CharacterClass::Char(*c)), groups)
    },
  }
}

/// find all nodes in the graph that are \epsilon away from given node set, mutate in place
pub fn bfs_epsilon<T: Hash + Ord + Clone, U, V: From<T> + Clone + Ord + Project<T>, F: FnMut(&U) -> bool>(
  nodes: &mut BTreeSet<V>, graph: &Graph<T, U>, mut f: F,
) {
  let mut queue = VecDeque::new();
  for node in nodes.iter() {
    // clone is necessary because we will be inserting into nodes, so we can't take a
    // reference to a node. A reference is a pointer that will be invalidated when the BTreeSet
    // gets bigger.
    queue.push_back(node.clone());
  }

  while !queue.is_empty() {
    let maybe_i = queue.pop_front().unwrap().project();
    let i = match maybe_i {
      Some(i) => i,
      _ => {continue},
    };
    let epsilon_dests = graph.map.get(&i).unwrap().transitions.iter()
      .filter_map(|(cc, j)| if f(cc) {Some(j)} else {None});
    for j in epsilon_dests {
      let v: V = j.clone().into();
      if !nodes.contains(&v) {
        queue.push_back(v.clone());
        nodes.insert(v.clone());
      }
    }
  }
}

// Like TryInto but custom so it doesn't interfere with the standard library.
pub trait Project<T> {
  fn project(&self) -> Option<T>;
}

impl<T: Eq + Hash> Graph<T, NfaTransition> {
  /// Finds a path through the NFA that traverses any number of epsilon and allowed_boundary edges.
  pub(crate) fn find_path_to<'a>(&'a self, dst: &T, sources: HashMap<&'a T, Vec<&'a T>>, allowed_boundary: &HashSet<&Boundary>) -> Option<Vec<&'a T>> {
    let mut queue = VecDeque::new();
    let mut backedges = HashMap::new();
    for source in sources.keys() {
      queue.push_back(*source);
      backedges.insert(*source, None);
    }
    while !queue.is_empty() {
      let n = queue.pop_front().unwrap();

      if *n == *dst {
        // Found it! Determine the path by walking backedges.
        let mut path = Vec::new();
        let mut path_node = Some(n);
        loop {
          if let Some(n) = path_node {
            path_node = *backedges.get(n).unwrap();
            if path_node == None {
              path.reverse();
              let mut full_path = sources.get(n).unwrap().clone();
              full_path.extend(path);
              return Some(full_path);
            }
            path.push(n);
          }
        }
      }

      for (transition, next) in self.map.get(n).unwrap().transitions.iter() {
        if backedges.contains_key(&next) {
          continue;
        }
        if match transition {
          NfaTransition::Empty => true,
          NfaTransition::Character(_) => false,
          NfaTransition::Boundary(b) => allowed_boundary.contains(b),
        } {
          backedges.insert(next, Some(n));
          queue.push_back(next);
        }
      }
    }
    None
  }

  pub(crate) fn extend_paths_by_char<'a>(&'a self, sources: HashMap<&T, Vec<&'a T>>, c: char) -> HashMap<&'a T, Vec<&'a T>> {
    let mut results = HashMap::new();
    for (source, path) in sources.into_iter() {
      for (transition, next) in self.map.get(source).unwrap().transitions.iter() {
        match transition {
          NfaTransition::Character(cc) => if cc.matches_char(c) {
            let mut new_path = path.clone();
            new_path.push(next);
            results.insert(next, new_path);
          }
          _ => {},
        }
      }
    }
    results
  }

  pub(crate) fn find_groups_by_path(&self, path: Vec<&T>, matched_string: &str, num_groups: i32) -> HashMap<GroupId, String> {
    if **path.get(0).unwrap() != self.root { panic!("path should start at NFA root!") }
    if !self.terminals.contains(path.last().unwrap()) { panic!("path should end at a terminal node of the NFA!") }
    let mut results: HashMap<GroupId, String> = (1..=num_groups).into_iter().map(|i| (i, String::new())).collect();
    let matched_chars: Vec<char> = matched_string.chars().collect();
    let mut ind_char = 0;

    for i in 0..path.len()-1 {
      let node = *path.get(i).unwrap();
      let mut transition = None;
      let next_node = *path.get(i+1).unwrap();

      for (tran, dst) in self.map.get(node).unwrap().transitions.iter() {
        if *dst == *next_node { transition = Some(tran); break; }
      }
      let consumed_char = match transition.unwrap() {
        NfaTransition::Character(_) => {
          ind_char += 1;
          Some(matched_chars.get(ind_char-1).unwrap())
        },
        _ => None,
      };
      // if transition consumes some character, then do group matches
      if let Some(c) = consumed_char {
        let src_groups = &self.map.get(node).unwrap().groups;
        let dst_groups = &self.map.get(next_node).unwrap().groups;
        for group in src_groups.intersection(dst_groups) {
          results.get_mut(group).unwrap().push(*c);
        }
      };
    }

    results
  }
}
