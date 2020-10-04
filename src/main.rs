use std::convert::TryInto;
use crate::graph::nfa_to_dfa;

#[macro_use] extern crate maplit;

mod parser;
mod graph;

fn main() {
  // let regex: parser::Regex = "a?([bc]|[ac])".try_into().unwrap();
  let regex: parser::Regex = ".*".try_into().unwrap();
  println!("regex is {:?}", regex);
  let nfa: graph::Graph<i32, graph::NfaTransition> = (&regex).into();
  println!("nfa is {:?}", nfa);
  let example = nfa.example();
  println!("example is {}", example.unwrap());
  let dfa = nfa_to_dfa(nfa);
  println!("dfa is {:?}", dfa);
  let is_match = dfa.match_string("a f 1 3fdas");
  println!("Does dfa match `a f 1 3fdas`? {}", is_match)
}
