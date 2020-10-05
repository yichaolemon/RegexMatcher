use crate::parser::{CharacterClass, Regex, Boundary};
use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::hash::Hash;
use std::{cmp, fmt};
use std::ops::Add;
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io::Write;

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
pub enum DfaTransition {
  Character(CharacterClass),
  Boundary(Boundary),
}

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

pub trait EdgeLabel {
  fn display(&self) -> String;
}

impl<T: Hash + Eq + EdgeLabel, U: Display> Display for Graph<T, U> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    write!(f, "digraph Regex {{\n")?;
    for (label, node) in self.map.iter() {
      write!(
        f,
        "{} [{},{}];\n",
        label.display(),
        if self.root == *label { "style=filled,color=\"0 0 .9\"" } else {"shape=ellipse"},
        if self.terminals.contains(label) { "peripheries=2" } else {"peripheries=1"},
      )?;
      for (transition, dst) in node.transitions.iter() {
        write!(f, "{} -> {} [label=\"{}\"];\n", label.display(), dst.display(), transition)?;
      }
    }
    write!(f, "}}\n")
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

impl Display for DfaTransition {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    write!(f, "{}", self.0)
  }
}

impl EdgeLabel for i32 {
  fn display(&self) -> String {
    return format!("{}", self)
  }
}

impl<T: EdgeLabel> EdgeLabel for BTreeSet<T> {
  fn display(&self) -> String {
    let mut s = String::new();
    for elt in self.iter() {
      s = format!("{}_{}", s, elt.display());
    }
    s
  }
}

pub fn write_graph_to_file<T: Hash + Eq + EdgeLabel, U: Display>(f: &str, g: &Graph<T, U>) {
  let mut file = File::create(f).unwrap();
  write!(&mut file, "{}", g).unwrap();
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
      let tmp_left_root = left_nfa.root;
      left_nfa.root = left_nfa.map.len() as i32;
      left_nfa.map.insert(left_nfa.root, Node{
        id: left_nfa.root,
        transitions: vec!((NfaTransition::Empty, tmp_left_root), (NfaTransition::Empty, right_nfa.root)),
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
pub fn nfa_to_dfa<T: Hash + Ord + Clone + Debug>(nfa: Graph<T, NfaTransition>) -> Graph<BTreeSet<T>, DfaTransition> {
  let mut iter_set: BTreeSet<T> = BTreeSet::new();
  iter_set.insert(nfa.root.clone());
  bfs_epsilon(&mut iter_set, &nfa, |cc| *cc == NfaTransition::Empty);
  let mut stack = VecDeque::new();
  stack.push_back(iter_set);
  let mut boo = true;

  let mut dfa = Graph::default();

  while !stack.is_empty() {
    let iter_set = stack.pop_front().unwrap();
    // merge all the nodes that are \epsilon away
    if boo { dfa.root = iter_set.clone(); boo = false; }
    if dfa.map.contains_key(&iter_set) {
      continue
    }
    // construct the new edges
    let edges: HashMap<T, DfaTransition> = iter_set.iter().flat_map(
      |i| nfa.map.get(i).unwrap().transitions.iter()
        .filter_map(|(cc, j)|
          match cc {
            NfaTransition::Empty =>
              None,
            NfaTransition::Character(cc) =>
              Some((j.clone(), DfaTransition::Character(cc.canonical_form()))),
            NfaTransition::Boundary(b) =>
              unimplemented!(),
          }
        ).collect::<Vec<(T, DfaTransition)>>()
    ).collect();
    let mut new_edges = Vec::new();
    for (ids, cc) in set_covering(edges).into_iter() {
      let mut ids = ids.clone();
      bfs_epsilon(&mut ids, &nfa, |cc| *cc == NfaTransition::Empty);
      new_edges.push((cc, ids.clone()));
      // push ids onto the stack
      stack.push_back(ids)
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

const MIN_CHAR: char = '\x00';
const MAX_CHAR: char = '\x7F';

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

  // The canonical form of any character class is a left-associative union of sorted disjoint nonempty ranges.
  // Special cases: empty character class is Range('b', 'a').
  // All characters are ascii, from '\x00' to '\x7F'.
  pub fn canonical_form(&self) -> CharacterClass {
    match self {
      CharacterClass::Char(c) => (*c).into(),
      CharacterClass::Any => range('\x00', '\x7F'),
      CharacterClass::Word => range('a', 'z')+range('A', 'Z')+range('0', '9')+'_'.into(),
      CharacterClass::Whitespace => range(' ', ' ')+'\t'.into()+'\n'.into()+'\r'.into(),
      CharacterClass::Digit => range('0', '9'),
      CharacterClass::Negation(cc) => {
        let cc = cc.canonical_form();
        if cc.is_empty() {
          return CharacterClass::Range(MIN_CHAR, MAX_CHAR);
        }
        let mut result = CharacterClass::default();
        let mut end_of_last_range = None;
        for cc in cc.iter() {
          if *cc == CharacterClass::default() { continue }
          let (a, b) = match cc {
            CharacterClass::Range(a, b) => (*a, *b),
            _ => panic!("canonical form can only contain ranges"),
          };
          match end_of_last_range {
            Some(e) => {
              result = result + range(char_add(e, 1), char_add(a, -1));
            },
            None => if a > MIN_CHAR {
              result = result + range(MIN_CHAR, char_add(a, -1));
            },
          }
          end_of_last_range = Some(b);
        }
        match end_of_last_range {
          Some(e) => if e < MAX_CHAR {
            result = result + range(char_add(e, 1), MAX_CHAR);
          }
          None => {},
        }
        result
      }
      CharacterClass::Union(cc1, cc2) => {
        // must be left associative, sorted, and disjoint, so we can't just union them.
        let mut ranges = BTreeSet::new();
        for cc in cc1.canonical_form().iter() {
          if *cc == CharacterClass::default() { continue }
          ranges.insert(match cc {
            CharacterClass::Range(a, b) => (*a, *b),
            _ => panic!("canonical form can only contain ranges"),
          });
        }
        for cc in cc2.canonical_form().iter() {
          if *cc == CharacterClass::default() { continue }
          ranges.insert(match cc {
            CharacterClass::Range(a, b) => (*a, *b),
            _ => panic!("canonical form can only contain ranges"),
          });
        }
        // ranges are sorted and nonempty, but we still may need to merge them
        let mut previous_range = None;
        let mut result = CharacterClass::default();
        for (a, b) in ranges.into_iter() {
          match previous_range {
            None => { previous_range = Some((a, b)) }
            Some((a0, b0)) => {
              if a <= char_add(b0, 1) {
                previous_range = Some((a0, cmp::max(b, b0)));
              } else {
                result = result + range(a0, b0);
                previous_range = Some((a, b));
              }
            }
          }
        }
        match previous_range {
          None => {},
          Some((a0, b0)) => {
            result = result + range(a0, b0);
          }
        }
        result
      }
      CharacterClass::Range(a, b) => if *b < *a {
        CharacterClass::default()
      } else {
        self.clone()
      },
    }
  }

  fn iter(&self) -> CharacterClassIterator {
    match self {
      CharacterClass::Union(cc1, cc2) =>
        CharacterClassIterator{
          cc: self,
          sub_iters: Some((cc1.iter().into(), cc2.iter().into())),
          done: false,
        },
      _ => CharacterClassIterator{
        cc: self,
        sub_iters: None,
        done: false,
      },
    }
  }
}

// wrap each node with the metadata we need to traverse the leaves in order.
struct CharacterClassIterator<'a> {
  cc: &'a CharacterClass,
  sub_iters: Option<(Box<CharacterClassIterator<'a>>, Box<CharacterClassIterator<'a>>)>,
  done: bool,
}

// iterate over unioned items in order
impl<'a> Iterator for CharacterClassIterator<'a> {
  type Item = &'a CharacterClass;

  fn next(&mut self) -> Option<Self::Item> {
    if self.done {
      return None
    }
    match self.sub_iters {
      Some((ref mut l, ref mut r)) => match l.next() {
        Some(cc) => { Some(cc) },
        None => match r.next() {
          Some(cc) => { Some(cc) },
          None => { self.done = true; None },
        },
      }
      None => { self.done = true; Some(self.cc) },
    }
  }
}

impl From<char> for CharacterClass {
  fn from(c: char) -> Self {
    CharacterClass::Range(c, c)
  }
}

impl Add<CharacterClass> for CharacterClass {
  type Output = CharacterClass;

  fn add(self, rhs: CharacterClass) -> Self::Output {
    if self == CharacterClass::default() {
      rhs
    } else if rhs == CharacterClass::default() {
      self
    } else {
      CharacterClass::Union(self.into(), rhs.into())
    }
  }
}

fn range(c1: char, c2: char) -> CharacterClass {
  CharacterClass::Range(c1, c2)
}

fn char_add(c: char, i: i32) -> char {
  ((c as i32) + i) as u8 as char
}

impl MathSet for CharacterClass {
  fn intersect(&self, other: &Self) -> Self {
    // wheeeeee
    CharacterClass::Negation((
      CharacterClass::Negation(self.clone().into())
        + CharacterClass::Negation(other.clone().into())
    ).into()).canonical_form()
  }

  fn setminus(&self, other: &Self) -> Self {
    self.intersect(&CharacterClass::Negation(other.clone().into()))
  }

  fn is_empty(&self) -> bool {
    self.canonical_form() == CharacterClass::default()
  }
}

#[cfg(test)]
mod intersect_tests {
  use super::*;

  #[test]
  fn test_empty_intersect() {
    let l = CharacterClass::Range('0', 'a');
    let r = CharacterClass::default();
    assert_eq!(l.intersect(&r), CharacterClass::default());
  }
}

impl MathSet for DfaTransition {
  fn intersect(&self, other: &Self) -> Self {
    match (self, other) {
      (DfaTransition::Character(cc1), DfaTransition::Character(cc2)) =>
        DfaTransition::Character(cc1.intersect(cc2)),
      (_, _) => panic!("Doesn't make sense to intersect boundary and character in dfa transition"),
    }
  }

  fn setminus(&self, other: &Self) -> Self {
    match (self, other) {
      (DfaTransition::Character(cc1), DfaTransition::Character(cc2)) =>
        DfaTransition::Character(cc1.setminus(cc2)),
      (_, _) => panic!("Doesn't make sense to setminus boundary and character in dfa transition"),
    }
  }

  fn is_empty(&self) -> bool {
    match self {
      DfaTransition::Character(cc) => cc.is_empty(),
      DfaTransition::Boundary(_) => false, // doesn't really make sense, since boundary doens't consume chars
    }
  }
}

// Given some mathematical sets (given as id->set), find all intersections
fn set_covering<T: Ord + Clone + Hash + Debug, S: MathSet + Debug>(sets: HashMap<T, S>) -> HashMap<BTreeSet<T>, S> {
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
      if ids2.is_subset(&ids) { continue }
      let intersection = s.intersect(s2);
      if intersection.is_empty() {
        continue
      }
      let id_union: BTreeSet<T> = ids.union(ids2).cloned().collect();
      let merged_id_union = match to_process.get(&intersection) {
        Some(ids3) => id_union.union(ids3).cloned().collect(),
        None => id_union,
      };
      new_to_process.insert(intersection, merged_id_union);
      leftover = leftover.setminus(s2);
    }
    for (_, s2) in result.iter() {
      leftover = leftover.setminus(s2);
    }
    to_process.extend(new_to_process);
    if !leftover.is_empty() {
      result.insert(ids.clone(), leftover);
    }
  }
  result
}

impl<T: Eq + Hash + Debug + EdgeLabel> Matcher for Graph<T, DfaTransition> {
  /// decide if the given string is part of the language defined by this DFA
  fn match_string(&self, s: &str) -> bool {
    let mut node = &self.root;
    let c_list: Vec<char> = s.chars().into_iter().collect();
    let mut i = 0;

    while i < c_list.len() {
      let c = c_list[i];
      let transitions = &self.map.get(node).unwrap().transitions;
      let mut found_match = false;

      for (transition, dst) in transitions {
        match transition {
          DfaTransition::Character(cc) => {
            if cc.matches_char(c) {
              if found_match {
                panic!("invalid dfa has two output edges for char '{:?}' at node {:?}. dfa is {:?}", c, node, self);
              }
              node = dst;
              found_match = true;
              i += 1;
            }
          },
          DfaTransition::Boundary(b) => {
            let c_before = c_list.get(i-1).copied();
            let c_after = Some(c);
            if b.matches(c_before, c_after) {
              if found_match {
                panic!("invalid dfa has two output edges for char '{:?}' at node {:?}. dfa is {:?}", c, node, self);
              }
              node = dst;
              found_match = true;
            }
          }
        }
      }
      if !found_match { return false }
    }

    self.terminals.contains(node)
  }

  fn print_to_file(&self, f: &str) {
    write_graph_to_file(f, self);
  }
}

pub trait Matcher {
  fn match_string(&self, s: &str) -> bool;
  fn print_to_file(&self, f: &str);
}

impl Regex {
  pub fn matcher(&self) -> Graph<BTreeSet<i32>, DfaTransition> {
    let nfa: Graph<i32, NfaTransition> = self.into();
    let dfa = nfa_to_dfa(nfa);
    dfa
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::convert::TryInto;

  #[test]
  fn test_simple() {
    let r: Regex = "a".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("a"));
    assert!(!m.match_string("b"));
    assert!(!m.match_string("aa"));
  }

  #[test]
  fn test_character_class() {
    let r: Regex = "a([ab]|[ac])c[abc]".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("abca"));
    assert!(!m.match_string("abbc"));
    assert!(!m.match_string("aaccc"));
  }

  #[test]
  fn test_character_class_intersection() {
    let r: Regex = "[0-a]+(\\d)*[abc|c]+".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("90736464ZLKHAHFUU``223|"));
  }

  #[test]
  fn test_any() {
    let r: Regex = ".[-a0-8]+[0-9]?[.]+".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("a-123..."));
    assert!(m.match_string("-19."));
    assert!(!m.match_string("a9."));
    assert!(!m.match_string("00000"));
    assert!(!m.match_string("x012-90."));
    assert!(m.match_string("00000."));
    m.print_to_file("out/any.dot");
  }

  #[test]
  fn test_stanford_loop() {
    // example from stanford pset that can cause a loop if you're not careful.
    let r: Regex = "(x?)*y".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("xxxxxy"));
    assert!(!m.match_string("x"));
    assert!(!m.match_string("xxxxxxxx"));
    assert!(!m.match_string("xxxyx"));
    assert!(m.match_string("y"));
    m.print_to_file("out/loop.dot");
  }
}
