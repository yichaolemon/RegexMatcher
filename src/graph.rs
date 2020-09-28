use std::rc::Rc;
use crate::parser::{CharacterClass, Regex, Boundary};
use std::collections::{HashMap, HashSet};
use std::ops::Deref;
use std::hash::Hash;

struct Node<T, U> {
  id: T,
  // edges are character classes
  transitions: Vec<(U, T)>,
}

struct Graph<T, U> {
  root: T,
  terminals: HashSet<T>,
  map: HashMap<T, Node<T, U>>,
}

pub enum NfaTransition {
  Empty, // no op
  Character(CharacterClass),
  Boundary(Boundary),
}

/// reindexing the nodes when merging two graphs
fn graph_reindex<T: Hash + Eq, U: Clone, F: FnMut(&T) -> T>(graph: Graph<T, U>, mut f: F) -> Graph<T, U>{
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

// construct non-deterministic finite automata from the parsed Regex
// pub fn build_nfa(regex: Regex) -> Graph<i32, NfaTransition> {
//   match regex {
//
//   }
// }
