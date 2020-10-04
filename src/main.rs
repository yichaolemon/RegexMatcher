use std::convert::TryInto;
use crate::graph::nfa_to_dfa;

#[macro_use] extern crate maplit;

mod parser;
mod graph;

fn main() {
  // let regex: parser::Regex = "a?([bc]|[ac])".try_into().unwrap();
  let regex: parser::Regex = "a?([ab]|[ac])".try_into().unwrap();
  println!("regex is {:?}", regex);
  let nfa: graph::Graph<i32, graph::NfaTransition> = (&regex).into();
  println!("nfa is {:?}", nfa);
  let example = nfa.example();
  println!("example is {}", example.unwrap());
  let dfa = nfa_to_dfa(nfa);
  println!("dfa is {:?}", dfa);
  let s = "abcdefg";
  let is_match = dfa.match_string(s);
  println!("Does dfa match `{}`? {}", s, is_match)
}
