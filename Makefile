build:
	cargo build

run:
	cargo run

dot:
	cargo run ; dot -Tpng out/nfa.dot > out/nfa.png && dot -Tpng out/dfa.dot > out/dfa.png
