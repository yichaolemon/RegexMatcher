use std::convert::TryInto;

mod parser;
mod graph;

fn main() {
    let regex: parser::Regex = "(a)".try_into().unwrap();
    println!("regex is {:?}", regex);
}
