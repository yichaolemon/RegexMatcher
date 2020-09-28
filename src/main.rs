use std::convert::TryInto;
#[macro_use] extern crate maplit;

mod parser;
mod graph;

fn main() {
    let regex: parser::Regex = "(a)".try_into().unwrap();
    println!("regex is {:?}", regex);
    let nfa: graph::Graph<i32, graph::NfaTransition> = (&regex).into();
    println!("nfa is {:?}", nfa);
}
