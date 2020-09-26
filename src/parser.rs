
enum CharacterClass {
  Char(char), // [a]
  Word, // \w
  Whitespace, // \s
  Negation(Box<CharacterClass>), // [^a], \W
  Union(Box<CharacterClass>, Box<CharacterClass>), // [ab]
}

enum Regex {
  Char(char), // a, \n
  Concat(Box<Regex>, Box<Regex>), // ab
  Group(Box<Regex>, i32), // (a) with a group index
  Kleene(Box<Regex>), // a*
  Plus(Box<Regex>), // a+
  Optional(Box<Regex>), // a?
  Class(CharacterClass), // \w , [abc], [a-z], [$#.\w], [^abc], [^abc\w]
  Alternative(Box<Regex>, Box<Regex>), // a|b
}
