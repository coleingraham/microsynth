//! Tokenizer for the microsynth DSL.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Source position for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub line: usize,
    pub col: usize,
}

/// A token with its source position.
#[derive(Debug, Clone)]
pub struct Spanned {
    pub token: Token,
    pub span: Span,
}

/// DSL tokens.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    SynthDef,
    Let,
    In,

    // Literals
    Ident(String),
    Number(f32),

    // Operators
    Eq,    // =
    Plus,  // +
    Minus, // -
    Star,  // *
    Slash, // /

    // Delimiters
    LParen, // (
    RParen, // )

    // Structure
    Newline,
    Semicolon,

    // End
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::SynthDef => write!(f, "synthdef"),
            Token::Let => write!(f, "let"),
            Token::In => write!(f, "in"),
            Token::Ident(s) => write!(f, "{s}"),
            Token::Number(n) => write!(f, "{n}"),
            Token::Eq => write!(f, "="),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::Newline => write!(f, "newline"),
            Token::Semicolon => write!(f, ";"),
            Token::Eof => write!(f, "end of input"),
        }
    }
}

/// Tokenizes DSL source into a list of spanned tokens.
pub fn tokenize(source: &str) -> Result<Vec<Spanned>, LexError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut pos = 0;
    let mut line = 1;
    let mut col = 1;

    while pos < chars.len() {
        let ch = chars[pos];

        // Skip spaces and tabs (not newlines)
        if ch == ' ' || ch == '\t' {
            pos += 1;
            col += 1;
            continue;
        }

        // Comments: -- to end of line
        if ch == '-' && pos + 1 < chars.len() && chars[pos + 1] == '-' {
            while pos < chars.len() && chars[pos] != '\n' {
                pos += 1;
            }
            continue;
        }

        let span = Span { line, col };

        match ch {
            '\n' => {
                // Collapse multiple newlines
                if tokens.last().map_or(true, |t: &Spanned| t.token != Token::Newline) {
                    tokens.push(Spanned {
                        token: Token::Newline,
                        span,
                    });
                }
                pos += 1;
                line += 1;
                col = 1;
            }
            '\r' => {
                pos += 1;
                col += 1;
            }
            '=' => {
                tokens.push(Spanned {
                    token: Token::Eq,
                    span,
                });
                pos += 1;
                col += 1;
            }
            '+' => {
                tokens.push(Spanned {
                    token: Token::Plus,
                    span,
                });
                pos += 1;
                col += 1;
            }
            '-' => {
                tokens.push(Spanned {
                    token: Token::Minus,
                    span,
                });
                pos += 1;
                col += 1;
            }
            '*' => {
                tokens.push(Spanned {
                    token: Token::Star,
                    span,
                });
                pos += 1;
                col += 1;
            }
            '/' => {
                tokens.push(Spanned {
                    token: Token::Slash,
                    span,
                });
                pos += 1;
                col += 1;
            }
            '(' => {
                tokens.push(Spanned {
                    token: Token::LParen,
                    span,
                });
                pos += 1;
                col += 1;
            }
            ')' => {
                tokens.push(Spanned {
                    token: Token::RParen,
                    span,
                });
                pos += 1;
                col += 1;
            }
            ';' => {
                tokens.push(Spanned {
                    token: Token::Semicolon,
                    span,
                });
                pos += 1;
                col += 1;
            }
            c if c.is_ascii_digit() || (c == '.' && pos + 1 < chars.len() && chars[pos + 1].is_ascii_digit()) => {
                let start = pos;
                while pos < chars.len() && chars[pos].is_ascii_digit() {
                    pos += 1;
                }
                if pos < chars.len() && chars[pos] == '.' {
                    pos += 1;
                    while pos < chars.len() && chars[pos].is_ascii_digit() {
                        pos += 1;
                    }
                }
                let num_str: String = chars[start..pos].iter().collect();
                let value: f32 = num_str.parse().map_err(|_| LexError {
                    message: alloc::format!("invalid number: {num_str}"),
                    span,
                })?;
                col += pos - start;
                tokens.push(Spanned {
                    token: Token::Number(value),
                    span,
                });
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = pos;
                while pos < chars.len()
                    && (chars[pos].is_ascii_alphanumeric() || chars[pos] == '_')
                {
                    pos += 1;
                }
                let word: String = chars[start..pos].iter().collect();
                col += pos - start;
                let token = match word.as_str() {
                    "synthdef" => Token::SynthDef,
                    "let" => Token::Let,
                    "in" => Token::In,
                    _ => Token::Ident(word),
                };
                tokens.push(Spanned { token, span });
            }
            _ => {
                return Err(LexError {
                    message: alloc::format!("unexpected character: '{ch}'"),
                    span,
                });
            }
        }
    }

    tokens.push(Spanned {
        token: Token::Eof,
        span: Span { line, col },
    });
    Ok(tokens)
}

/// A lexer error.
#[derive(Debug, Clone)]
pub struct LexError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "lex error at {}:{}: {}",
            self.span.line, self.span.col, self.message
        )
    }
}
