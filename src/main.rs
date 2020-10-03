use std::convert::TryInto;
use crate::graph::nfa_to_dfa;

#[macro_use] extern crate maplit;

mod parser;
mod graph;

fn main() {
    let regex: parser::Regex = "^\\((\\d\\d\\d)\\)[-.]([0-9][0-9][0-9])[-.]([0-9][0-9][0-9][0-9])$".try_into().unwrap();
    println!("regex is {:?}", regex);
    let nfa: graph::Graph<i32, graph::NfaTransition> = (&regex).into();
    println!("nfa is {:?}", nfa);
    let example = nfa.example();
    println!("example is {}", example.unwrap());
    let dfa = nfa_to_dfa(nfa);
    println!("dfa is {:?}", dfa);
}
