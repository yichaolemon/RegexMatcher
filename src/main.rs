use std::convert::TryInto;
use crate::graph::{nfa_to_dfa, Matcher, write_graph_to_file};
use std::fs;
use std::io;

#[macro_use] extern crate maplit;

mod parser;
mod graph;

fn main() {
  loop {
    let mut expr = String::new();
    println!("Enter a regex: ");
    io::stdin()
      .read_line(&mut expr)
      .expect("Failed to read expression");

    let regex: parser::Regex = (&*expr.trim()).try_into().unwrap();
    // println!("regex is {:?}", regex);
    let nfa: graph::Graph<i32, graph::NfaTransition> = (&regex).into();
    // println!("nfa is {:?}", nfa);
    fs::create_dir_all("out").unwrap();
    write_graph_to_file("examples/nfa.dot", &nfa);
    // let example = nfa.example();
    // println!("example is {}", example.unwrap());
    let dfa = nfa_to_dfa(nfa);
    write_graph_to_file("examples/dfa.dot", &dfa);
    println!("dfa is {:?}", dfa);
    loop {
      let mut match_str = String::new();
      println!("Enter a string to be matched: ");
      io::stdin()
        .read_line(&mut match_str)
        .expect("Failed to read expression");
      if match_str.trim() == "quit" { break; }
      let is_match = dfa.match_string(&*match_str.trim());
      println!("Match result: [{}]", is_match)
    }
  }
}
