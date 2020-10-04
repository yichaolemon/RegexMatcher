use std::convert::TryInto;
use crate::graph::{nfa_to_dfa, Matcher};

#[macro_use] extern crate maplit;

mod parser;
mod graph;

fn main() {
  let r = "a([ab]|[ac])c[abc]";
  let regex: parser::Regex = r.try_into().unwrap();
  println!("regex is {:?}", regex);
  let nfa: graph::Graph<i32, graph::NfaTransition> = (&regex).into();
  println!("nfa is {:?}", nfa);
  let example = nfa.example();
  println!("example is {}", example.unwrap());
  let dfa = nfa_to_dfa(nfa);
  println!("dfa is {:?}", dfa);
  let s = "abca";
  let is_match = dfa.match_string(s);
  println!("Does {} match `{}`? {}", s, r, is_match)
}
