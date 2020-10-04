use crate::parser::{CharacterClass, Regex, Boundary};
use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::hash::Hash;
use std::cmp;
use crate::graph::NfaTransition::Character;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NfaTransition {
  Empty, // no op
  Character(CharacterClass),
  Boundary(Boundary),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct DfaTransition(CharacterClass);

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

pub trait Transition {
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
fn bfs_epsilon<T: Hash + Ord + Clone, U, F: FnMut(&U) -> bool>(
  nodes: &mut BTreeSet<T>, graph: &Graph<T, U>, mut f: F
) {
  let mut queue = VecDeque::new();
  for node in nodes.iter() {
    // clone is necessary because we will be inserting into nodes, so we can't take a
    // reference to a node. A reference is a pointer that will be invalidated when the BTreeSet
    // gets bigger.
    queue.push_back(node.clone());
  }

  while !queue.is_empty() {
    let i = queue.pop_front().unwrap();
    let transitions = graph.map.get(&i).unwrap().transitions.iter()
      .filter_map(|(cc, j)| if f(cc) {Some(j)} else {None});
    for j in transitions {
      if !nodes.contains(j) {
        queue.push_back(j.clone());
        nodes.insert(j.clone());
      }
    }
  }
}

/// given a non-deterministic finite automata, construct its equivalent deterministic finite automata
pub fn nfa_to_dfa<T: Hash + Ord + Clone>(nfa: Graph<T, NfaTransition>) -> Graph<BTreeSet<T>, DfaTransition> {
  let mut iter_set: BTreeSet<T> = BTreeSet::new();
  iter_set.insert(nfa.root.clone());
  let mut stack = VecDeque::new();
  stack.push_back(iter_set);
  let mut boo = true;

  let mut dfa = Graph::default();

  // TODO: push onto stack
  while !stack.is_empty() {
    let mut iter_set = stack.pop_front().unwrap();
    // merge all the nodes that are \epsilon away
    bfs_epsilon(&mut iter_set, &nfa, |cc| *cc == NfaTransition::Empty);
    if boo { dfa.root = iter_set.clone(); boo = false; }
    if dfa.map.contains_key(&iter_set) {
      continue
    }
    // construct the new edges
    let edges: HashMap<T, DfaTransition> = iter_set.iter().flat_map(
      |i| nfa.map.get(i).unwrap().transitions.iter()
        .filter_map(|(cc, j)| if iter_set.contains(j) { None } else {
          match cc {
            NfaTransition::Empty => None,
            NfaTransition::Character(cc) => Some((j.clone(), DfaTransition(cc.clone()))),
            NfaTransition::Boundary(_) => unimplemented!(),
          }
        }).collect::<Vec<(T, DfaTransition)>>()
    ).collect();
    let mut new_edges = Vec::new();
    for (ids, cc) in set_covering(edges).into_iter() {
      new_edges.push((cc, ids))
    }
    let new_node = Node {
      id: iter_set.clone(),
      transitions: new_edges,
    };
    for i in iter_set.iter() {
      if nfa.terminals.contains(i) {
        dfa.terminals.insert(iter_set.clone());
        break
      }
    }
    dfa.map.insert(iter_set, new_node);
  }
  dfa
}

trait MathSet: Hash + Eq + Clone {
  fn intersect(&self, other: &Self) -> Self;
  fn setminus(&self, other: &Self) -> Self;
  fn is_empty(&self) -> bool;
}

impl CharacterClass {
  pub fn matches_char(&self, c: char) -> bool {
    match self {
      CharacterClass::Char(c1) => c == *c1,
      CharacterClass::Any => true,
      CharacterClass::Word => c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_',
      CharacterClass::Whitespace => c == ' ' || c == '\n' || c == '\r' || c == '\t',
      CharacterClass::Digit => c >= '0' && c <= '9',
      CharacterClass::Negation(cc) => !cc.matches_char(c),
      CharacterClass::Union(cc1, cc2) => cc1.matches_char(c) || cc2.matches_char(c),
      CharacterClass::Range(a, b) => c >= *a && c <= *b,
    }
  }
}

impl MathSet for CharacterClass {
  fn intersect(&self, other: &Self) -> Self {
    match self {
      CharacterClass::Char(c) => if other.matches_char(*c) { self.clone() } else { CharacterClass::default() },
      CharacterClass::Any => other.clone(),
      CharacterClass::Word => match other {
        CharacterClass::Word => self.clone(),
        CharacterClass::Whitespace => CharacterClass::default(),
        CharacterClass::Digit => other.clone(),
        _ => other.intersect(self),
      },
      CharacterClass::Whitespace => match other {
        CharacterClass::Word => CharacterClass::default(),
        CharacterClass::Whitespace => self.clone(),
        CharacterClass::Digit => CharacterClass::default(),
        _ => other.intersect(self),
      },
      CharacterClass::Digit => match other {
        CharacterClass::Word => self.clone(),
        CharacterClass::Whitespace => CharacterClass::default(),
        CharacterClass::Digit => self.clone(),
        _ => other.intersect(self),
      },
      CharacterClass::Negation(cc) =>
        if cc.is_empty() {
          other.clone()
        } else {
          // de morgan's law. intersection is inverse of union of inverses.
          CharacterClass::Negation(CharacterClass::Union(CharacterClass::Negation(other.into()).into(), cc.clone()).into())
        },
      CharacterClass::Union(cc1, cc2) =>
        CharacterClass::Union(cc1.intersect(other).into(), cc2.intersect(other).into()),
      CharacterClass::Range(a, b) =>
        if (*a <= *b) {
          match other {
            CharacterClass::Any => self.clone(),
            CharacterClass::Digit => {
              let mut new_range = CharacterClass::default();
              if '0' <= *a && *a <= '9' {
                new_range = CharacterClass::Range(a.clone(), cmp::min(*b, '9'));
              } else if '0' <= *b && *b <= '9' {
                new_range = CharacterClass::Range(cmp::max(*a, '0'), b.clone());
              }
              if new_range == CharacterClass::Range('0', '9') { CharacterClass::Digit } else { new_range }
            },
            CharacterClass::Char(c) =>
              if *a <= *c && *c <= *b { other.clone() } else { CharacterClass::default() },
            CharacterClass::Whitespace =>
              if *a <= ' ' && ' ' <= *b || *a <= '\n' && '\n' <= *b || *a <= '\r' && '\r' <= *b
                || *a <= '\t' && '\t' <= *b {
                CharacterClass::Whitespace
              },
            CharacterClass::Word => {}
            CharacterClass::Negation(_) => {}
            CharacterClass::Union(_, _) => {}
            CharacterClass::Range(_, _) => {}
          }
        }
    }
  }

  fn setminus(&self, other: &Self) -> Self {
    unimplemented!()
  }

  fn is_empty(&self) -> bool {
    unimplemented!()
  }
}

impl MathSet for DfaTransition {
  fn intersect(&self, other: &Self) -> Self {
    return DfaTransition(self.0.intersect(&(*other).0));
  }

  fn setminus(&self, other: &Self) -> Self {
    return DfaTransition(self.0.setminus(&(*other).0));
  }

  fn is_empty(&self) -> bool {
    return self.0.is_empty()
  }
}

// Given some mathematical sets (given as id->set), find all intersections
fn set_covering<T: Ord + Clone + Hash, S: MathSet>(sets: HashMap<T, S>) -> HashMap<BTreeSet<T>, S> {
  let mut result = HashMap::new();
  let mut to_process = HashMap::new();

  for (i, input_set) in sets.into_iter() {
    let mut ids = BTreeSet::new();
    ids.insert(i);
    to_process.insert(input_set, ids);
  }

  while !to_process.is_empty() {
    let (s, ids) = to_process.iter().next().unwrap();
    let (s, ids) = (s.clone(), ids.clone());
    to_process.remove(&s);
    let mut leftover = s.clone();
    let mut new_to_process = HashMap::new();
    for (s2, ids2) in to_process.iter() {
      let intersection = s.intersect(s2);
      let id_union: BTreeSet<T> = ids.union(ids2).cloned().collect();
      let merged_id_union = match to_process.get(&intersection) {
        Some(ids3) => id_union.union(ids3).cloned().collect(),
        None => id_union,
      };
      new_to_process.insert(intersection, merged_id_union);
      leftover = leftover.setminus(s2);
    }
    to_process.extend(new_to_process);
    if !leftover.is_empty() {
      result.insert(ids.clone(), leftover);
    }
  }
  result
}
