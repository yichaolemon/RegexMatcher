use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::fmt::Formatter;
use std::ops::Deref;

const SPECIAL_CHARS: &str = "[\\^$.|?*+()";

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

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CharacterClass {
  Char(char), // [a]
  Any, // .
  Word, // \w
  Whitespace, // \s
  Digit, // \d
  Negation(Box<CharacterClass>), // [^a], \W
  Union(Box<CharacterClass>, Box<CharacterClass>), // [ab]
  Range(char, char), // a-z
}

impl Default for CharacterClass {
  fn default() -> Self {
    CharacterClass::Range('b', 'a')  // empty
  }
}

/// zero-width matcher
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Boundary {
  Any, // matches any boundary
  Word, // \b
  Start, // ^
  End, // $
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Regex {
  Char(char), // a, \n
  Group(Box<Regex>, i32), // (a) with a group index
  Class(CharacterClass), // \w , [abc], [a-z], [$#.\w], [^abc], [^abc\w]
  Boundary(Boundary),
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
      false => Err("Invalid regex expression".into())
    }
  }
}

impl fmt::Display for Regex {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    match self {
      Regex::Char(c) => {
        if SPECIAL_CHARS.contains(*c) {
          write!(f, "\\")?
        };
        write!(f, "{}", c)
      }
      Regex::Class(char_class) => write!(f, "{}", char_class),
      Regex::Group(r, _) => write!(f, "({})", r.deref()),
      Regex::Boundary(b) => write!(f, "{}", b),
      Regex::Kleene(r) => write!(f, "{}*", r),
      Regex::Plus(r) => write!(f, "{}+", r),
      Regex::Optional(r) => write!(f, "{}?", r),
      Regex::Concat(r1, r2) => write!(f, "{}{}", r1, r2),
      Regex::Alternative(r1, r2) => write!(f, "{}|{}", r1, r2),
    }
  }
}

impl fmt::Display for CharacterClass {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {

    fn inner_fmt(cc: &CharacterClass, f: &mut Formatter<'_>) -> fmt::Result {
      match cc {
        CharacterClass::Char(c) => write!(f, "{}", c),
        CharacterClass::Any => panic!("Any not allowed inside []"),
        CharacterClass::Range(a, b) => write!(f, "{}-{}", a, b),
        CharacterClass::Union(cc1, cc2) => {
          inner_fmt(cc1, f)?;
          inner_fmt(cc2, f)
        },
        CharacterClass::Word => write!(f, "\\w"),
        CharacterClass::Digit => write!(f, "\\d"),
        CharacterClass::Whitespace => write!(f, "\\s"),
        CharacterClass::Negation(cc) =>
          match cc.deref() {
            CharacterClass::Word => write!(f, "\\w"),
            CharacterClass::Digit => write!(f, "\\d"),
            CharacterClass::Whitespace => write!(f, "\\s"),
            _ => panic!("Negation must be at the top level of character class"),
          },
      }
    }

    match self {
      CharacterClass::Any => write!(f, "."),
      CharacterClass::Negation(cc) => { write!(f, "[^")?; inner_fmt(cc, f)?; write!(f, "]") },
      _ => { write!(f, "[")?; inner_fmt(self, f)?; write!(f, "]") },
    }
  }
}

impl fmt::Display for Boundary {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    match self {
      Boundary::Any => fmt::Result::Ok(()),
      Boundary::Start => write!(f, "^"),
      Boundary::End => write!(f, "$"),
      Boundary::Word => write!(f, "\\b"),
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
      _ => Err( "unreachable".into() )
    }
  }
}

/// turn char class into regex character class
impl From<CharacterClass> for Regex {
  fn from(char_class: CharacterClass) -> Self {
    Regex::Class(char_class)
  }
}

/// turn characters into regex character
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
  parse_alternative(input_str)
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
  while !(str_remaining.is_empty() || str_remaining.starts_with("|") || str_remaining.starts_with(")")) {
    let (regex2, str_remaining2) = parse_quantifiers(str_remaining)?;
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

// Char, Group, Class, Boundaries
fn parse_atom(input_str: &str) -> ParseResult {
  if input_str.starts_with("(") {
    let (regex, str_remaining) = parse_alternative(input_str.get(1..).unwrap())?;
    if str_remaining.starts_with(")") {
      Ok((Regex::Group(regex.into(), 0), str_remaining.get(1..).unwrap()))
    } else {
      Err("Unbalanced parenthesis in regex".into())
    }
  } else if input_str.starts_with("[") {
    let (regex, str_remaining) = parse_character_class(input_str.get(1..).unwrap())?;
    if str_remaining.starts_with("]") {
      Ok((regex, str_remaining.get(1..).unwrap()))
    } else {
      Err("Unclosed character class in regex".into())
    }
  } else if input_str.starts_with("^") {
    Ok((Regex::Boundary(Boundary::Start), input_str.get(1..).unwrap()))
  } else if input_str.starts_with("$") {
    Ok((Regex::Boundary(Boundary::End), input_str.get(1..).unwrap()))
  } else if input_str.starts_with(".") {
    Ok((Regex::Class(CharacterClass::Any), input_str.get(1..).unwrap()))
  } else if input_str.is_empty()
    || input_str.starts_with("|")
    || input_str.starts_with("?")
    || input_str.starts_with("*")
    || input_str.starts_with(")") {
    Ok((Regex::Boundary(Boundary::Any), input_str))
  } else {
    parse_single_char(input_str)
  }
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
  let mut char_class: CharacterClass = regex.try_into()?;

  while !str_remaining.starts_with("]") && !str_remaining.is_empty() {
    let (regex2, str_remaining2) = parse_character_class_atom(str_remaining)?;
    let next_char_class: CharacterClass = regex2.try_into()?;
    char_class = CharacterClass::Union(char_class.into(), next_char_class.into());
    str_remaining = str_remaining2;
  }

  if negation {
    char_class = CharacterClass::Negation(char_class.into());
  }

  Ok((Regex::Class(char_class), str_remaining))
}

fn parse_character_class_atom(input_str: &str) -> ParseResult {
  let (regex, mut str_remaining) = parse_single_char(input_str)?;
  let mut char_class = regex.try_into()?;

  if str_remaining.starts_with("-") {
    let start_char = match char_class {
      CharacterClass::Char(c) => c,
      _ => return Err( "Invalid `-` range expression".into() )
    };

    match parse_single_char(str_remaining.get(1..).unwrap())? {
      (Regex::Char(c), s) => {
        str_remaining = s;
        char_class = CharacterClass::Range(start_char, c);
      },
      _ => return Err( "Invalid `-` range expression".into() )
    }
  }

  Ok((Regex::Class(char_class), str_remaining))
}

fn parse_single_char(input_str: &str) -> ParseResult {
  let mut str_remaining = input_str;
  if str_remaining.starts_with("\\") {
    str_remaining = str_remaining.get(1..).unwrap();
    if str_remaining.is_empty() {
      Err(ParseError{msg: "lonely backslash wants to escape something".to_string()})
    } else {
      // We only care about ascii.
      let regex = escaped_char(str_remaining.chars().next().unwrap())?;
      Ok((regex, str_remaining.get(1..).unwrap()))
    }
  } else if str_remaining.is_empty() {
    Err("Expected char but got end of string".into())
  } else {
    let rest = str_remaining.get(1..).unwrap();
    let first_char = str_remaining.chars().next().unwrap();
    Ok((Regex::Char(first_char), rest))
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
    'S' => Ok(CharacterClass::Negation(CharacterClass::Whitespace.into()).into()),
    'd' => Ok(CharacterClass::Digit.into()),
    'D' => Ok(CharacterClass::Negation(CharacterClass::Digit.into()).into()),
    'b' => Ok(Regex::Boundary(Boundary::Word)),
    _ =>
      if SPECIAL_CHARS.contains(c) {
        Ok(c.into())
      } else {
        Err((&*format!("Illegal escape char: {}", c)).into())
      }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn test_parse_display_inverse(s: &str) {
    let regex: Regex = s.try_into().unwrap();
    println!("regex for {} is {:?}", s, regex);
    assert_eq!(format!("{}", regex), s)
  }

  #[test]
  fn concat() {
    test_parse_display_inverse("abc");
  }

  #[test]
  fn simple_group() {
    test_parse_display_inverse("(a)");
  }

  #[test]
  fn group() {
    test_parse_display_inverse("(a)bc");
  }

  #[test]
  fn char_class() {
    test_parse_display_inverse("[word][\\w]");
  }

  #[test]
  fn dots() {
    test_parse_display_inverse("a.*b.*c");
  }

  #[test]
  fn phone_number() {
    test_parse_display_inverse("^\\(?([0-9]{3})\\)?[-.]?([0-9]{3})[-.]?([0-9]{4})$");
  }
}
