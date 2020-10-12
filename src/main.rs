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
    let nfa: graph::Graph<i32, graph::NfaTransition> = (&regex).into();
    // println!("nfa is {:?}", nfa);
    fs::create_dir_all("out").unwrap();
    write_graph_to_file("out/nfa.dot", &nfa);
    let dfa = nfa_to_dfa(nfa);
    write_graph_to_file("out/dfa.dot", &dfa);
    let matcher = regex.matcher();
    // println!("dfa is {:?}", dfa);
    loop {
      let mut match_str = String::new();
      println!("Enter a string to be matched: ");
      io::stdin()
        .read_line(&mut match_str)
        .expect("Failed to read expression");
      if match_str.trim() == "quit" { break; }
      let is_match = matcher.match_string(&*match_str.trim());
      println!("Match result: [{:?}]", is_match);
      let mut group_id = 1;
      loop {
        if let Some(group) = is_match.group(group_id) {
          println!("Group {} is \"{}\"", group_id, group);
          group_id+=1;
        } else {
          break;
        }
      }
    }
  }
}
