use std::convert::{TryFrom, TryInto};
use std::fmt;

#[derive(Debug)]
pub struct ParseError {
  msg: String,
}

impl fmt::Display for ParseError {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    write!(f, "Parse Error: {}", self.msg)
  }
}

type ParseResult<'a> = Result<(Regex, &'a str), ParseError>;

enum CharacterClass {
  Char(char), // [a]
  Word, // \w
  Whitespace, // \s
  Digit, // \d
  Negation(Box<CharacterClass>), // [^a], \W
  Union(Box<CharacterClass>, Box<CharacterClass>), // [ab]
  Range(char, char), // a-z
}

enum Regex {
  Char(char), // a, \n
  Group(Box<Regex>, i32), // (a) with a group index
  Class(CharacterClass), // \w , [abc], [a-z], [$#.\w], [^abc], [^abc\w]
  Kleene(Box<Regex>), // a*
  Plus(Box<Regex>), // a+
  Optional(Box<Regex>), // a?
  Concat(Box<Regex>, Box<Regex>), // ab
  Alternative(Box<Regex>, Box<Regex>), // a|b
}

/// turn whole string into regex
impl TryFrom<&str> for Regex {
  type Error = ParseError;

  fn try_from(value: &str) -> Result<Self, Self::Error> {
    let (regex, str_remaining) = parse_regex(value)?;

    match str_remaining.is_empty() {
      true => Ok(regex),
      false => Err(ParseError {msg: "Invalid regex expression".into_string()})
    }
  }
}

/// turn regex into the underlying character class
impl TryFrom<Regex> for CharacterClass {
  type Error = ParseError;

  fn try_from(value: Regex) -> Result<Self, Self::Error> {
    match value {
      Regex::Char(c) => Ok(CharacterClass::Char(c)),
      Regex::Class(class) => Ok(class),
      _ => Err(ParseError{msg : "unreachable".into_string() })
    }
  }
}

/// turn char class into regex character class
impl From<CharacterClass> for Regex {
  fn from(char_class: CharacterClass) -> Self {
    Regex::Class(char_class)
  }
}

/// turn characters into regex character class
impl From<char> for Regex {
  fn from(c: char) -> Self {
    Regex::Char(c)
  }
}

/// turn string into parse errors
impl From<&str> for ParseError {
  fn from(s: &str) -> Self {
    ParseError { msg : s.to_string() }
  }
}

fn parse_regex(input_str: &str) -> ParseResult {
  unimplemented!()
}

fn parse_alternative(input_str: &str) -> ParseResult {
  let (mut regex, mut str_remaining) = parse_concat(input_str)?;
  while str_remaining.starts_with("|") {
    let (regex2, str_remaining2) = parse_concat(str_remaining.get(1..).unwrap())?;
    regex = Regex::Alternative(regex.into(), regex2.into());
    str_remaining = str_remaining2;
  }

  Ok((regex, str_remaining))
}

fn parse_concat(input_str: &str) -> ParseResult {
  let (mut regex, mut str_remaining) = parse_quantifiers(input_str)?;
  while !str_remaining.starts_with("|") && !str_remaining.is_empty() {
    let (regex2, str_remaining2) = parse_quantifiers(str_remaining.get(1..).unwrap())?;
    regex = Regex::Concat(regex.into(), regex2.into());
    str_remaining = str_remaining2;
  }

  Ok((regex, str_remaining))
}

fn parse_quantifiers(input_str: &str) -> ParseResult {
  let (mut regex, mut str_remaining) = parse_atom(input_str)?;
  if str_remaining.starts_with("*") {
    regex = Regex::Kleene(regex.into());
    str_remaining = str_remaining.get(1..).unwrap();
  } else if str_remaining.starts_with("?") {
    regex = Regex::Optional(regex.into());
    str_remaining = str_remaining.get(1..).unwrap();
  } else if str_remaining.starts_with("+") {
    regex = Regex::Plus(regex.into());
    str_remaining = str_remaining.get(1..).unwrap();
  }

  Ok((regex, str_remaining))
}

// Char, Group, Class
fn parse_atom(input_str: &str) -> ParseResult {
  if input_str.starts_with("(") {
    let (regex, str_remaining) = parse_alternative(input_str.get(1..).unwrap())?;
    if str_remaining.starts_with(")") {
      Ok((Regex::Group(regex.into(), 0), str_remaining.get(1..).unwrap()))
    } else {
      Err(ParseError { msg: "Unbalanced parenthesis in regex".into_string() })
    }
  } else if input_str.starts_with("[") {
    let (regex, )
  }
  Ok
}

// parse [...]
fn parse_character_class(input_str: &str) -> ParseResult {
  let mut negation = false;
  let mut str_remaining = input_str;

  // check if it's negation
  if str_remaining.starts_with("^") {
    negation = true;
    str_remaining = str_remaining.get(1..).unwrap();
  }

  // start out with a character class.
  let (regex, mut str_remaining) = parse_character_class_atom(str_remaining)?;
  let mut char_class = regex.try_into()?;

  while !str_remaining.starts_with("]") && !str_remaining.is_empty() {
    let (regex2, str_remaining2) = parse_character_class_atom(str_remaining)?;
    char_class = CharacterClass::Union(char_class.into(), regex2.try_into()?);
    str_remaining = str_remaining2;
  }

  if negation {
    char_class = CharacterClass::Negation(char_class.into());
  }

  Ok((Regex::Class(char_class), str_remaining))
}

fn parse_character_class_atom(input_str: &str) -> ParseResult {
  let mut str_remaining = input_str;
  let (regex, mut str_remaining) = parse_single_char(str_remaining)?;
  let mut char_class = regex.try_into()?;

  if (str_remaining.starts_with("-")) {
    let start_char = match char_class {
      CharacterClass::Char(c) => c,
      _ => return Err(ParseError { msg: "Invalid `-` range expression".into_string() })
    };

    match parse_single_char(str_remaining.get(1..).unwrap())? {
      Ok((Regex::Char(c), s)) => {
        str_remaining = s;
        char_class = CharacterClass::Range(start_char, c);
      },
      _ => return Err(ParseError { msg: "Invalid `-` range expression".into_string() })
    }
  }

  Ok((Regex::Class(char_class), str_remaining))
}

fn parse_single_char(input_str: &str) -> ParseResult {
  let mut str_remaining = input_str;
  if str_remaining.starts_with("\\") {
    if str_remaining.is_empty() {
      Err(ParseError{msg: "lonely backslash wants to escape something".to_string()})
    } else {
      // We only care about ascii.
      let regex = escaped_char(str_remaining.chars().next().unwrap())?;
      Ok((regex, str_remaining.get(1..).unwrap()))
    }
  } else if str_remaining.is_empty() {
    Err(ParseError { msg: "Error".into_string() })
  } else {
    Ok((Regex::Char(str_remaining.chars().next().unwrap()), str_remaining.get(1..).unwrap()))
  }
}

fn escaped_char(c: char) -> Result<Regex, ParseError> {
  match c {
    'n' => Ok('\n'.into()),
    't' => Ok('\t'.into()),
    'r' => Ok('\r'.into()),
    'w' => Ok(CharacterClass::Word.into()),
    'W' => Ok(CharacterClass::Negation(CharacterClass::Word.into()).into()),
    's' => Ok(CharacterClass::Whitespace.into()),
  }
}