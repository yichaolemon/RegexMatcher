use std::fmt::{Display, Formatter, Debug};
use std::fmt;
use crate::graph::{EdgeLabel, Transition, Graph, Node, write_graph_to_file};
use std::collections::{BTreeSet, HashSet, HashMap, VecDeque};
use crate::parser::{Boundary, GroupId, Regex};
use std::hash::Hash;
use crate::nfa::{NfaTransition, Project, bfs_epsilon};
use crate::character_class::{CharacterClass, MathSet, set_covering};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DfaTransition {
  Character(CharacterClass),
  Boundary(Boundary),
  NegBoundary(Boundary),
}

impl Default for DfaTransition {
  fn default() -> Self {
    DfaTransition::Character(CharacterClass::default())
  }
}

impl Display for DfaTransition {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    match self {
      DfaTransition::Character(cc) => write!(f, "{}", cc),
      DfaTransition::Boundary(b) => write!(f, "{}", b),
      DfaTransition::NegBoundary(b) => write!(f, "~{}", b),
    }
  }
}

impl<T: EdgeLabel, B: EdgeLabel> EdgeLabel for DfaIdentifier<T, B> {
  fn display(&self) -> String {
    match self {
      DfaIdentifier::Plain(t) => t.display(),
      DfaIdentifier::Bound(b) => b.display(),
      DfaIdentifier::InverseBound(b) => format!("¬{}", b.display()),
    }
  }
}

// In the DFA each node is identified by a set of these DfaIdentifiers.
// The DFA node should be thought of as being on any of the corresponding NFA nodes,
// with all Bound and InverseBound conditions satisfied.
#[derive(Hash, PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
pub enum DfaIdentifier<T, B> {
  Plain(T),
  Bound(B),
  InverseBound(B),
}

impl EdgeLabel for DfaTransition {
  fn display(&self) -> String {
    match self {
      DfaTransition::Character(cc) => cc.display(),
      DfaTransition::Boundary(b) => b.display(),
      DfaTransition::NegBoundary(b) => format!("¬{}", b.display()),
    }
  }
}

impl Transition for DfaTransition {
  fn example(&self, s: &String) -> String {
    match self {
      DfaTransition::Character(cc) =>
        format!("{}{}", s, cc.example()),
      DfaTransition::Boundary(_) =>
        unimplemented!(),
      DfaTransition::NegBoundary(_) =>
        unimplemented!()
    }
  }
}

impl<T: Clone, B> Project<T> for DfaIdentifier<T, B> {
  fn project(&self) -> Option<T> {
    match self {
      DfaIdentifier::Plain(t) => Some(t.clone()),
      _ => None,
    }
  }
}

impl<T, B> From<T> for DfaIdentifier<T, B> {
  fn from(value: T) -> Self {
    DfaIdentifier::Plain(value)
  }
}

type DfaNode<T, B> = BTreeSet<DfaIdentifier<T, B>>;

/// given a non-deterministic finite automata, construct its equivalent deterministic finite automata
pub fn nfa_to_dfa<T: Hash + Ord + Clone + Debug>(nfa: Graph<T, NfaTransition>)
  -> Graph<DfaNode<T, Boundary>, DfaTransition> {
  let mut iter_set = BTreeSet::new();
  iter_set.insert(DfaIdentifier::Plain(nfa.root.clone()));
  bfs_epsilon(&mut iter_set, &nfa, |cc| *cc == NfaTransition::Empty);
  let mut stack = VecDeque::new();
  stack.push_back(iter_set);
  let mut first = true;

  let mut dfa = Graph::default();

  while !stack.is_empty() {
    let iter_set = stack.pop_front().unwrap();
    // merge all the nodes that are \epsilon away
    if first { dfa.root = iter_set.clone(); first = false; }
    if dfa.map.contains_key(&iter_set) {
      continue
    }
    // find the existing edges so we can construct new edges.
    let mut new_edges = Vec::new();

    // if there are any boundary edges, split the current node.
    // TODO: this doesn't have to be a flatmap, since we only want one it can be a loop.
    let boundary_edges: HashSet<Boundary> = iter_set.iter().flat_map(
      |i| match i {
        DfaIdentifier::Plain(i) =>
          nfa.map.get(i).unwrap().transitions.iter()
            .filter_map(|(cc, _)|
              match cc {
                NfaTransition::Boundary(b) =>
                  if iter_set.contains(&DfaIdentifier::Bound(b.clone())) || iter_set.contains(&DfaIdentifier::InverseBound(b.clone())) {
                    None
                  } else {
                    Some(b.clone())
                  },
                _ => None,
              }
            ).collect::<Vec<Boundary>>(),
        _ => vec!(),
      }
    ).collect();

    if boundary_edges.len() > 0 {
      // split the node to make a deterministic choice based on the boundary.
      // TODO: short circuit Any boundary.
      let b = boundary_edges.into_iter().next().unwrap();
      let mut iter_set_bound = iter_set.clone();
      iter_set_bound.insert(DfaIdentifier::Bound(b.clone()));
      bfs_epsilon(&mut iter_set_bound, &nfa, |nfa_transition| match nfa_transition {
        NfaTransition::Empty => true,
        NfaTransition::Boundary(b_transition) => *b_transition == b,
        _ => false,
      });
      new_edges.push((DfaTransition::Boundary(b.clone()), iter_set_bound.clone()));
      stack.push_back(iter_set_bound);
      let mut iter_set_bound_inv = iter_set.clone();
      iter_set_bound_inv.insert(DfaIdentifier::InverseBound(b.clone()));
      new_edges.push((DfaTransition::NegBoundary(b), iter_set_bound_inv.clone()));
      stack.push_back(iter_set_bound_inv);
    } else {
      // no boundary edges. find edges by taking a set covering of character edges.
      let character_edges: HashMap<DfaIdentifier<T, Boundary>, DfaTransition> = iter_set.iter().flat_map(
        |i| match i {
          DfaIdentifier::Plain(i) =>
            nfa.map.get(i).unwrap().transitions.iter()
              .filter_map(|(cc, j)|
                match cc {
                  NfaTransition::Character(cc) =>
                    Some((DfaIdentifier::Plain(j.clone()), DfaTransition::Character(cc.canonical_form()))),
                  _ => None,
                }
              ).collect::<Vec<(DfaIdentifier<T, Boundary>, DfaTransition)>>(),
          _ => vec!(),
        }
      ).collect();

      for (ids, cc) in set_covering(character_edges).into_iter() {
        let mut ids = ids.clone();
        bfs_epsilon(&mut ids, &nfa, |cc| *cc == NfaTransition::Empty);
        new_edges.push((cc, ids.clone()));
        // push ids onto the stack
        stack.push_back(ids)
      }
    }
    let new_node = Node {
      id: iter_set.clone(),
      transitions: new_edges,
      groups: BTreeSet::new(), // we don't populate this for DFA nodes
    };
    for i in iter_set.iter() {
      match i {
        DfaIdentifier::Plain(i) =>
          if nfa.terminals.contains(i) {
            dfa.terminals.insert(iter_set.clone());
            break
          },
        _ => {},
      }
    }
    dfa.map.insert(iter_set, new_node);
  }
  dfa
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
      DfaTransition::NegBoundary(_) => false,
    }
  }
}

#[derive(Debug, Clone, Default)]
pub struct RegexMatcher<T: Ord> {
  nfa: Graph<T, NfaTransition>,
  dfa: Graph<BTreeSet<DfaIdentifier<T, Boundary>>, DfaTransition>,
}

#[derive(Default, Debug, Clone)]
pub struct Match {
  is_match: bool,
  groups: HashMap<GroupId, String>,
  debug: String,
}

impl Match {
  pub fn group(&self, id: GroupId) -> Option<&str> {
    self.groups.get(&id).map(|s| &**s)
  }
}

pub trait Matcher {
  fn match_string(&self, s: &str) -> Match;
  fn print_to_file(&self, f: &str);
}

/// main function that matches a string against the DFA constructed from the regex
impl<T: Eq + Hash + Debug + EdgeLabel + Ord> Matcher for RegexMatcher<T> {
  /// decide if the given string is part of the language defined by this DFA
  fn match_string(&self, s: &str) -> Match {
    let mut dfa_node = &self.dfa.root;
    let c_list: Vec<char> = s.chars().into_iter().collect();
    let mut i = 0;
    // For each NFA node in the current DFA node,
    // keep track of 1 possible path in the NFA that goes from the root to this node.
    let mut possible_paths: HashMap<&T, Vec<&T>> = HashMap::new();
    for node_id in dfa_node.iter() {
      match node_id {
        DfaIdentifier::Plain(id) => {
          possible_paths.insert(id, self.nfa.find_path_to(
            id,
            hashmap!(&self.nfa.root => vec!(&self.nfa.root)),
            &HashSet::new(),
          ).unwrap());
        },
        _ => {},
      }
    }

    while i <= c_list.len() {
      let c = if i < c_list.len() { Some(c_list[i]) } else { None };
      let transitions = &self.dfa.map.get(dfa_node).unwrap().transitions;
      let mut found_match = false;

      for (transition, dst) in transitions {
        let mut match_consumes_char = false;
        let new_found_match = match transition {
          DfaTransition::Character(cc) => {
            match c {
              None => false,
              Some(c) => if cc.matches_char(c) {
                i += 1;
                match_consumes_char = true;
                true
              } else { false }
            }
          },
          DfaTransition::Boundary(b) => {
            let c_before = if i == 0 { None } else {
              c_list.get(i-1).copied()
            };
            b.matches(c_before, c)
          }
          DfaTransition::NegBoundary(b) => {
            let c_before = if i == 0 { None } else {
              c_list.get(i-1).copied()
            };
            !b.matches(c_before, c)
          }
        };

        if new_found_match {
          if found_match {
            panic!("invalid dfa has two output edges for char '{:?}' at node {:?}. dfa is {:?}", c, dfa_node, self);
          }
          // We have transitioned from dfa_node to dst by consuming c or by consuming a boundary.
          // Now we find paths in the NFA that could have gotten us here.
          if match_consumes_char {
            if let Some(c) = c {
              possible_paths = self.nfa.extend_paths_by_char(possible_paths, c);
            }
          }
          let boundaries = dst.iter().filter_map(|id| match id {
            DfaIdentifier::Bound(b) => Some(b),
            _ => None,
          }).collect();
          let mut new_possible_paths = HashMap::new();
          for nfa_dst in dst.iter().filter_map(|id| match id {
            DfaIdentifier::Plain(id) => Some(id),
            _ => None,
          }) {
            new_possible_paths.insert(
              nfa_dst,
              self.nfa.find_path_to(nfa_dst, possible_paths.clone(), &boundaries).unwrap(),
            );
          }
          possible_paths = new_possible_paths;

          // Move on to the next character.
          dfa_node = dst;
          found_match = true;
        }
      }
      if !found_match {
        if i < c_list.len() {
          return Match::default()
        } else {
          break
        }
      }
    }

    if self.dfa.terminals.contains(dfa_node) {
      let mut nfa_path: Vec<&T> = Vec::new();
      for nfa_node in dfa_node.iter().filter_map(|id| match id {
        DfaIdentifier::Plain(id) => Some(id),
        _ => None,
      }) {
        if self.nfa.terminals.contains(nfa_node) {
          nfa_path = possible_paths.get(nfa_node).unwrap().to_vec();
          break;
        }
      }
      Match{
        is_match: true,
        groups: hashmap!(),
        debug: format!("{:?}", nfa_path),
      }
    } else { Match::default() }
  }

  fn print_to_file(&self, f: &str) {
    write_graph_to_file(f, &self.dfa);
  }
}

impl Regex {
  pub fn matcher(&self) -> RegexMatcher<i32> {
    let nfa: Graph<i32, NfaTransition> = self.into();
    let dfa = nfa_to_dfa(nfa.clone());
    RegexMatcher{
      nfa,
      dfa,
    }
  }
}

#[cfg(test)]
mod group_tests {
  use super::*;
  use std::convert::TryInto;

  #[test]
  fn test_simple() {
    let r: Regex = "(a)".try_into().unwrap();
    let m = r.matcher();
    assert_eq!(m.match_string("a").group(1), Some("a"));
    assert_eq!(m.match_string("b").group(1), None);
    assert_eq!(m.match_string("a").group(2), None);
  }

  #[test]
  fn test_multiple() {
    let r: Regex = "([0-9]*),([0-9]*)".try_into().unwrap();
    let m = r.matcher();
    let my_match = m.match_string("123,45");
    assert_eq!(my_match.group(1), Some("123"));
    assert_eq!(my_match.group(2), Some("45"));
    assert_eq!(m.match_string("123").group(1), None);
  }

  #[test]
  fn test_greedy() {
    let r: Regex = "(.*)a".try_into().unwrap();
    let m = r.matcher();
    m.print_to_file("out/greedy.dot");
    assert_eq!(m.match_string("ba").group(1), Some("b"));
  }

  #[test]
  fn test_nested_overlap() {
    let r: Regex = "My name is ((.*\\.) (\\w+))".try_into().unwrap();
    let m = r.matcher();
    let my_match = m.match_string("My name is Mr. Wallace");
    assert_eq!(my_match.group(1), Some("Mr. Wallace"));
    assert_eq!(my_match.group(2), Some("Mr."));
    assert_eq!(my_match.group(3), Some("Wallace"));
  }
}

#[cfg(test)]
mod match_tests {
  use super::*;
  use std::convert::TryInto;

  #[test]
  fn test_simple() {
    let r: Regex = "a".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("a").is_match);
    assert!(!m.match_string("b").is_match);
    assert!(!m.match_string("aa").is_match);
    m.print_to_file("out/simple.dot")
  }

  #[test]
  fn test_character_class() {
    let r: Regex = "a([ab]|[ac])c[abc]".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("abca").is_match);
    assert!(!m.match_string("abbc").is_match);
    assert!(!m.match_string("aaccc").is_match);
  }

  #[test]
  fn test_character_class_intersection() {
    let r: Regex = "[0-a]+(\\d)*[abc|c]+".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("90736464ZLKHAHFUU``223|").is_match);
  }

  #[test]
  fn test_any() {
    let r: Regex = ".[-a0-8]+[0-9]?[.]+".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("a-123...").is_match);
    assert!(m.match_string("-19.").is_match);
    assert!(!m.match_string("a9.").is_match);
    assert!(!m.match_string("00000").is_match);
    assert!(!m.match_string("x012-90.").is_match);
    assert!(m.match_string("00000.").is_match);
    m.print_to_file("out/any.dot");
  }

  #[test]
  fn test_stanford_loop() {
    // example from stanford pset that can cause a loop if you're not careful.
    let r: Regex = "(x?)*y".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("xxxxxy").is_match);
    assert!(!m.match_string("x").is_match);
    assert!(!m.match_string("xxxxxxxx").is_match);
    assert!(!m.match_string("xxxyx").is_match);
    assert!(m.match_string("y").is_match);
    m.print_to_file("out/loop.dot");
  }

  #[test]
  fn test_cat_word() {
    let r: Regex = ".*\\bcat\\b.*".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("hello cat!").is_match);
    assert!(m.match_string("cat").is_match);
    assert!(!m.match_string("scatterbrained").is_match);
    assert!(!m.match_string("catatonic cats").is_match);
    assert!(m.match_string("catatonic cat").is_match);
    m.print_to_file("out/cat.dot");
  }

  #[test]
  fn test_impossible_word_boundary() {
    let r: Regex = "\\w+\\b\\w+".try_into().unwrap();
    let m = r.matcher();
    assert!(!m.match_string("eeeeee").is_match);
    assert!(!m.match_string("eee.eee").is_match);
    assert!(!m.match_string("cc").is_match);
  }

  #[test]
  fn test_start_end() {
    let r: Regex = "^id:[0-9]+$".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("id:123").is_match);
    assert!(!m.match_string("hid:1").is_match);
    assert!(!m.match_string("id:123 ").is_match);
  }

  #[test]
  fn test_bad_start_end() {
    let r: Regex = "a^b$c".try_into().unwrap();
    let m = r.matcher();
    assert!(!m.match_string("abc").is_match);
  }

  #[test]
  fn test_multi_boundary() {
    let r: Regex = "^\\b()hi\\b()$".try_into().unwrap();
    let m = r.matcher();
    assert!(m.match_string("hi").is_match);
    assert!(!m.match_string(" hi").is_match);
    assert!(!m.match_string("hi ").is_match);
    m.print_to_file("out/multi.dot");
  }
}
