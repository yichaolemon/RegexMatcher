use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::fmt::Formatter;
use std::ops::Deref;
use crate::character_class::CharacterClass;

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

type ParseResult<'a> = Result<(Regex, &'a str, GroupId), ParseError>;

/// zero-width matcher
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Boundary {
  Any, // matches any boundary
  Word, // \b
  Start, // ^
  End, // $
}

pub type GroupId = i32;

#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Regex {
  Char(char), // a, \n
  Group(Box<Regex>, GroupId), // (a) with a group index
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
    let (regex, str_remaining, _) = parse_regex(value)?;

    match str_remaining.is_empty() {
      true => Ok(regex),
      false => Err("Invalid regex expression".into())
    }
  }
}

impl Boundary {
  pub fn matches(&self, before: Option<char>, after: Option<char>) -> bool {
    let w = CharacterClass::Word;
    match self {
      Boundary::Any => true,
      Boundary::Word => match (before, after) {
        (None, None) => false,
        (Some(c1), Some(c2)) =>
          w.matches_char(c1) && !w.matches_char(c2)
            || !w.matches_char(c1) && w.matches_char(c2),
        (None, Some(c)) => w.matches_char(c),
        (Some(c), None) => w.matches_char(c),
      },
      Boundary::Start => before == None,
      Boundary::End => after == None,
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
  // GroupId starts at 1
  parse_alternative(input_str, 1)
}

fn parse_alternative(input_str: &str, group: GroupId) -> ParseResult {
  let (mut regex, mut str_remaining, mut group) = parse_concat(input_str, group)?;
  while str_remaining.starts_with("|") {
    let (regex2, str_remaining2, new_group) = parse_concat(
      str_remaining.get(1..).unwrap(), group)?;
    regex = Regex::Alternative(regex.into(), regex2.into());
    str_remaining = str_remaining2;
    group = new_group;
  }

  Ok((regex, str_remaining, group))
}

fn parse_concat(input_str: &str, group: GroupId) -> ParseResult {
  let (mut regex, mut str_remaining, mut group) = parse_quantifiers(input_str, group)?;
  while !(str_remaining.is_empty() || str_remaining.starts_with("|") || str_remaining.starts_with(")")) {
    let (regex2, str_remaining2, group2) = parse_quantifiers(str_remaining, group)?;
    regex = Regex::Concat(regex.into(), regex2.into());
    str_remaining = str_remaining2;
    group = group2;
  }

  Ok((regex, str_remaining, group))
}

fn parse_quantifiers(input_str: &str, group: GroupId) -> ParseResult {
  let (mut regex, mut str_remaining, group) = parse_atom(input_str, group)?;
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

  Ok((regex, str_remaining, group))
}

// Char, Group, Class, Boundaries
fn parse_atom(input_str: &str, group: GroupId) -> ParseResult {
  if input_str.starts_with("(") {
    let this_group = group;
    let (regex, str_remaining, group) =
      parse_alternative(input_str.get(1..).unwrap(), this_group+1)?;
    if str_remaining.starts_with(")") {
      // Here's where we increment group id.
      Ok((Regex::Group(regex.into(), this_group), str_remaining.get(1..).unwrap(), group))
    } else {
      Err("Unbalanced parenthesis in regex".into())
    }
  } else if input_str.starts_with("[") {
    let (regex, str_remaining, group) = parse_character_class(input_str.get(1..).unwrap(), group)?;
    if str_remaining.starts_with("]") {
      Ok((regex, str_remaining.get(1..).unwrap(), group))
    } else {
      Err("Unclosed character class in regex".into())
    }
  } else if input_str.starts_with("^") {
    Ok((Regex::Boundary(Boundary::Start), input_str.get(1..).unwrap(), group))
  } else if input_str.starts_with("$") {
    Ok((Regex::Boundary(Boundary::End), input_str.get(1..).unwrap(), group))
  } else if input_str.starts_with(".") {
    Ok((Regex::Class(CharacterClass::Any), input_str.get(1..).unwrap(), group))
  } else if input_str.is_empty()
    || input_str.starts_with("|")
    || input_str.starts_with("?")
    || input_str.starts_with("*")
    || input_str.starts_with(")") {
    Ok((Regex::Boundary(Boundary::Any), input_str, group))
  } else {
    parse_single_char(input_str, group)
  }
}

// parse [...]
fn parse_character_class(input_str: &str, group: GroupId) -> ParseResult {
  let mut negation = false;
  let mut str_remaining = input_str;

  // check if it's negation
  if str_remaining.starts_with("^") {
    negation = true;
    str_remaining = str_remaining.get(1..).unwrap();
  }

  // start out with a character class.
  let (regex, mut str_remaining, mut group) = parse_character_class_atom(str_remaining, group)?;
  let mut char_class: CharacterClass = regex.try_into()?;

  while !str_remaining.starts_with("]") && !str_remaining.is_empty() {
    let (regex2, str_remaining2, group2) = parse_character_class_atom(str_remaining, group)?;
    let next_char_class: CharacterClass = regex2.try_into()?;
    char_class = CharacterClass::Union(char_class.into(), next_char_class.into());
    str_remaining = str_remaining2;
    group = group2;
  }

  if negation {
    char_class = CharacterClass::Negation(char_class.into());
  }

  Ok((Regex::Class(char_class), str_remaining, group))
}

fn parse_character_class_atom(input_str: &str, group: GroupId) -> ParseResult {
  let (regex, mut str_remaining, mut group) = parse_single_char(input_str, group)?;
  let mut char_class = regex.try_into()?;

  if str_remaining.starts_with("-") {
    let start_char = match char_class {
      CharacterClass::Char(c) => c,
      _ => return Err( "Invalid `-` range expression".into() )
    };

    match parse_single_char(str_remaining.get(1..).unwrap(), group)? {
      (Regex::Char(c), s, g) => {
        str_remaining = s;
        group = g;
        char_class = CharacterClass::Range(start_char, c);
      },
      _ => return Err( "Invalid `-` range expression".into() )
    }
  }

  Ok((Regex::Class(char_class), str_remaining, group))
}

fn parse_single_char(input_str: &str, group: GroupId) -> ParseResult {
  let mut str_remaining = input_str;
  if str_remaining.starts_with("\\") {
    str_remaining = str_remaining.get(1..).unwrap();
    if str_remaining.is_empty() {
      Err(ParseError{msg: "lonely backslash wants to escape something".to_string()})
    } else {
      // We only care about ascii.
      let regex = escaped_char(str_remaining.chars().next().unwrap())?;
      Ok((regex, str_remaining.get(1..).unwrap(), group))
    }
  } else if str_remaining.is_empty() {
    Err("Expected char but got end of string".into())
  } else {
    let rest = str_remaining.get(1..).unwrap();
    let first_char = str_remaining.chars().next().unwrap();
    Ok((Regex::Char(first_char), rest, group))
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
