use crate::parser::{Boundary, GroupId};
use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::hash::Hash;
use std::{fmt};
use std::fmt::{Debug, Display, Formatter};
use std::fs::File;
use std::io::Write;
use hex::ToHex;

#[derive(Debug, Clone)]
pub struct Node<T, U> {
  pub id: T,
  // edges are character classes
  pub transitions: Vec<(U, T)>,

  // which capture groups contain this node.
  pub groups: BTreeSet<GroupId>,
}

#[derive(Debug, Clone, Default)]
pub struct Graph<T, U> {
  pub root: T,
  pub terminals: HashSet<T>,
  pub map: HashMap<T, Node<T, U>>,
}


pub trait EdgeLabel {
  fn display(&self) -> String;
}

fn encoded_label<L: EdgeLabel>(l: &L) -> String {
  format!("_{}", l.display().encode_hex::<String>())
}

fn display_groups(groups: &BTreeSet<GroupId>) -> String {
  let mut s = String::new();
  if groups.len() == 0 {
    return s
  }
  s.push_str(" (");
  let mut first = true;
  for group in groups.iter() {
    if first {
      first = false;
    } else {
      s.push_str(", ");
    }
    s.push_str(&*format!("{}", *group));
  }
  s.push(')');
  s
}

impl<T: Hash + Eq + EdgeLabel, U: EdgeLabel> Display for Graph<T, U> {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    write!(f, "digraph Regex {{\n")?;
    for (label, node) in self.map.iter() {
      write!(
        f,
        "{} [{},{},label=\"{}{}\"];\n",
        encoded_label(label),
        if self.root == *label { "style=filled,color=\"0 0 .9\"" } else {"shape=ellipse"},
        if self.terminals.contains(label) { "peripheries=2" } else {"peripheries=1"},
        label.display(),
        display_groups(&node.groups),
      )?;
      for (transition, dst) in node.transitions.iter() {
        write!(f, "{} -> {} [label=\"{}\"];\n", encoded_label(label), encoded_label(dst), transition.display())?;
      }
    }
    write!(f, "}}\n")
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
      if s.len() > 0 {
        s = format!("{},{}", s, elt.display());
      } else {
        s = elt.display();
      }
    }
    s
  }
}

impl EdgeLabel for Boundary {
  fn display(&self) -> String {
    match self {
      Boundary::Any => format!("()"),
      Boundary::Word => format!("\\\\b"),
      Boundary::Start => format!("^"),
      Boundary::End => format!("$"),
    }
  }
}

impl EdgeLabel for char {
  fn display(&self) -> String {
    (*self).escape_default().to_string().replace("\\u", "\\x")
      .replace("\\", "\\\\").replace("{", "").replace("}", "")
  }
}

pub fn write_graph_to_file<T: Hash + Eq + EdgeLabel, U: EdgeLabel>(f: &str, g: &Graph<T, U>) {
  let mut file = File::create(f).unwrap();
  write!(&mut file, "{}", g).unwrap();
}

pub trait Transition {
  fn example(&self, s: &String) -> String;
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
