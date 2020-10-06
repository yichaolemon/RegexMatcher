use std::convert::TryInto;
use crate::graph::{nfa_to_dfa, Matcher, write_graph_to_file};
use std::fs;

#[macro_use] extern crate maplit;

mod parser;
mod graph;

fn main() {
  let r = "a";
  let regex: parser::Regex = r.try_into().unwrap();
  println!("regex is {:?}", regex);
  let nfa: graph::Graph<i32, graph::NfaTransition> = (&regex).into();
  println!("nfa is {:?}", nfa);
  fs::create_dir_all("out").unwrap();
  write_graph_to_file("out/nfa.dot", &nfa);
  // let example = nfa.example();
  // println!("example is {}", example.unwrap());
  let dfa = nfa_to_dfa(nfa);
  write_graph_to_file("out/dfa.dot", &dfa);
  println!("dfa is {:?}", dfa);
  let s = "a";
  let is_match = dfa.match_string(s);
  println!("Does {} match `{}`? {}", s, r, is_match)

}
