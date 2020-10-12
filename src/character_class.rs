use crate::graph::EdgeLabel;
use std::collections::{BTreeSet, HashMap};
use std::fmt::{Formatter, Debug};
use std::{fmt, cmp};
use std::ops::{Deref, Add};
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CharacterClass {
  Char(char), // [a]
  Any, // .
  Word, // \w
  Whitespace, // \s
  Digit, // \d
  Negation(Box<CharacterClass>), // [^a], \W
  Union(Box<CharacterClass>, Box<CharacterClass>), // [ab]
  Range(char, char), // a-z
}

impl Default for CharacterClass {
  fn default() -> Self {
    CharacterClass::Range('b', 'a')  // empty
  }
}

impl fmt::Display for CharacterClass {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {

    fn inner_fmt(cc: &CharacterClass, f: &mut Formatter<'_>) -> fmt::Result {
      match cc {
        CharacterClass::Char(c) => write!(f, "{}", c.escape_default().collect::<String>()),
        CharacterClass::Any => panic!("Any not allowed inside []"),
        CharacterClass::Range(a, b) => write!(f, "{}-{}", a.escape_default().collect::<String>(), b.escape_default().collect::<String>()),
        CharacterClass::Union(cc1, cc2) => {
          inner_fmt(cc1, f)?;
          inner_fmt(cc2, f)
        },
        CharacterClass::Word => write!(f, "\\w"),
        CharacterClass::Digit => write!(f, "\\d"),
        CharacterClass::Whitespace => write!(f, "\\s"),
        CharacterClass::Negation(cc) =>
          match cc.deref() {
            CharacterClass::Word => write!(f, "\\w"),
            CharacterClass::Digit => write!(f, "\\d"),
            CharacterClass::Whitespace => write!(f, "\\s"),
            _ => panic!("Negation must be at the top level of character class"),
          },
      }
    }

    match self {
      CharacterClass::Any => write!(f, "."),
      CharacterClass::Negation(cc) => { write!(f, "[^")?; inner_fmt(cc, f)?; write!(f, "]") },
      _ => { write!(f, "[")?; inner_fmt(self, f)?; write!(f, "]") },
    }
  }
}

impl EdgeLabel for CharacterClass {
  fn display(&self) -> String {
    let mut complex = false;
    let inner = {
      let mut s = String::new();
      for range in self.iter() {
        let r = match range {
          CharacterClass::Range(a, b) => if *a == *b {
            (*a).display()
          } else {
            complex = true;
            format!("{}-{}", (*a).display(), (*b).display())
          },
          _ => panic!("expected canonical form"),
        };
        if s.len() > 0 {
          complex = true;
          s = format!("{},{}", s, r);
        } else {
          s = r;
        }
      }
      s
    };
    if !complex {
      inner
    } else {
      format!("[{}]", inner)
    }
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

pub trait MathSet: Hash + Eq + Clone {
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
mod set_covering_tests {
  use super::*;
  use crate::dfa::DfaIdentifier;
  use crate::parser::Boundary;

  #[test]
  fn test_full_intersection() {
    let word = CharacterClass::Word;
    let plain_0: DfaIdentifier<i32, Boundary> = DfaIdentifier::Plain(0);
    let plain_1: DfaIdentifier<i32, Boundary> = DfaIdentifier::Plain(5);
    let covering = set_covering(hashmap!(
      plain_0.clone() => word.clone(),
      plain_1.clone() => word.clone(),
    ));
    assert_eq!(covering, hashmap!(
      btreeset!(plain_0, plain_1) => word.clone(),
    ))
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

// Given some mathematical sets (given as id->set), find all intersections
pub fn set_covering<T: Ord + Clone + Hash + Debug, S: MathSet + Debug>(sets: HashMap<T, S>) -> HashMap<BTreeSet<T>, S> {
  let mut result = HashMap::new();
  let mut to_process = HashMap::new();

  for (i, input_set) in sets.into_iter() {
    let mut ids = BTreeSet::new();
    ids.insert(i);
    let unioned_ids = match to_process.get(&input_set) {
      Some(ids2) => ids.union(ids2).cloned().collect(),
      None => ids,
    };
    to_process.insert(input_set, unioned_ids);
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
