//! Recursive descent parser for the microsynth DSL.
//!
//! Grammar (Haskell-inspired):
//!
//! ```text
//! program     = synthdef*
//! synthdef    = 'synthdef' IDENT param* '=' body
//! param       = IDENT '=' NUMBER
//! body        = statement* expr
//! statement   = 'let' IDENT '=' expr (NEWLINE | ';')
//! expr        = add_expr
//! add_expr    = mul_expr (('+' | '-') mul_expr)*
//! mul_expr    = unary_expr (('*' | '/') unary_expr)*
//! unary_expr  = '-' unary_expr | app_expr
//! app_expr    = atom atom*   -- first atom is function if followed by more atoms
//! atom        = NUMBER | IDENT | '(' expr ')'
//! ```
//!
//! Newlines separate `let` statements. Within expressions, newlines are ignored.
//! The last expression in a body (not preceded by `let`) is the output.

use crate::dsl::ast::*;
use crate::dsl::lexer::{Span, Spanned, Token};
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// Parser state.
pub struct Parser {
    tokens: Vec<Spanned>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Spanned>) -> Self {
        Parser { tokens, pos: 0 }
    }

    /// Parse a complete program (one or more synthdefs).
    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        self.skip_newlines();
        let mut defs = Vec::new();
        while !self.at_eof() {
            defs.push(self.parse_synthdef()?);
            self.skip_newlines();
        }
        if defs.is_empty() {
            return Err(self.error("expected at least one synthdef"));
        }
        Ok(Program { defs })
    }

    /// Parse a single synthdef declaration.
    fn parse_synthdef(&mut self) -> Result<SynthDefDecl, ParseError> {
        self.expect(Token::SynthDef)?;
        let name = self.expect_ident()?;

        // Parse params: IDENT '=' NUMBER, stopping when we hit a bare '='
        let mut params = Vec::new();
        loop {
            // Look ahead: is this IDENT '=' NUMBER (param) or just '=' (body start)?
            if self.check(&Token::Eq) {
                // Bare '=' means body starts
                break;
            }
            if !self.check_ident() {
                break;
            }
            // Peek: IDENT '=' NUMBER ?
            if self.pos + 2 < self.tokens.len() {
                if let Token::Ident(_) = &self.tokens[self.pos].token {
                    if self.tokens[self.pos + 1].token == Token::Eq {
                        if let Token::Number(_) = &self.tokens[self.pos + 2].token {
                            let pname = self.expect_ident()?;
                            self.expect(Token::Eq)?;
                            let default = self.expect_number()?;
                            params.push(Param {
                                name: pname,
                                default,
                            });
                            continue;
                        }
                        // IDENT '=' but not NUMBER — might be negative default
                        if self.tokens[self.pos + 2].token == Token::Minus {
                            if let Some(spanned) = self.tokens.get(self.pos + 3) {
                                if let Token::Number(_) = &spanned.token {
                                    let pname = self.expect_ident()?;
                                    self.expect(Token::Eq)?;
                                    self.expect(Token::Minus)?;
                                    let default = -self.expect_number()?;
                                    params.push(Param {
                                        name: pname,
                                        default,
                                    });
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
            break;
        }

        self.expect(Token::Eq)?;
        self.skip_newlines();

        let body = self.parse_body()?;

        Ok(SynthDefDecl { name, params, body })
    }

    /// Parse the body of a synthdef: delegates to parse_expr which handles
    /// both statement-level `let` bindings and inline `let...in`.
    fn parse_body(&mut self) -> Result<Expr, ParseError> {
        self.parse_let_or_expr()
    }

    /// Parse a sequence of `let` bindings (with or without `in`) followed
    /// by a final expression. Supports both styles:
    ///
    /// Statement-level (body of a synthdef):
    /// ```text
    /// let x = 1.0
    /// let y = 2.0
    /// x + y
    /// ```
    ///
    /// Inline with `in`:
    /// ```text
    /// let x = 1.0; y = 2.0 in x + y
    /// ```
    fn parse_let_or_expr(&mut self) -> Result<Expr, ParseError> {
        if !self.check(&Token::Let) {
            return self.parse_add();
        }

        self.advance(); // consume 'let'
        let mut bindings = Vec::new();

        loop {
            let name = self.expect_ident()?;
            self.expect(Token::Eq)?;
            let value = self.parse_add()?;
            bindings.push(Binding { name, value });

            // After a binding, look for what follows:
            self.skip_separators();

            // 'in' terminates bindings (inline let...in style)
            if self.check(&Token::In) {
                self.advance();
                let body = self.parse_let_or_expr()?;
                return Ok(Expr::Let(bindings, Box::new(body)));
            }

            // Another 'let' keyword starts a new binding
            if self.check(&Token::Let) {
                self.advance();
                continue;
            }

            // IDENT '=' pattern (without 'let') is another binding in same block
            if self.check_ident() && self.peek_next_is(&Token::Eq) {
                continue;
            }

            // Otherwise, what follows is the body expression (statement-level style)
            break;
        }

        let body = self.parse_let_or_expr()?;
        Ok(Expr::Let(bindings, Box::new(body)))
    }

    /// Parse an expression (no let handling — use parse_let_or_expr for that).
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_let_or_expr()
    }

    /// add_expr = mul_expr (('+' | '-') mul_expr)*
    fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_mul()?;
        loop {
            if self.check(&Token::Plus) {
                self.advance();
                self.skip_newlines();
                let right = self.parse_mul()?;
                left = Expr::BinOp(BinOp::Add, Box::new(left), Box::new(right));
            } else if self.check(&Token::Minus) {
                self.advance();
                self.skip_newlines();
                let right = self.parse_mul()?;
                left = Expr::BinOp(BinOp::Sub, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    /// mul_expr = unary_expr (('*' | '/') unary_expr)*
    fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            if self.check(&Token::Star) {
                self.advance();
                self.skip_newlines();
                let right = self.parse_unary()?;
                left = Expr::BinOp(BinOp::Mul, Box::new(left), Box::new(right));
            } else if self.check(&Token::Slash) {
                self.advance();
                self.skip_newlines();
                let right = self.parse_unary()?;
                left = Expr::BinOp(BinOp::Div, Box::new(left), Box::new(right));
            } else {
                break;
            }
        }
        Ok(left)
    }

    /// unary_expr = '-' unary_expr | app_expr
    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if self.check(&Token::Minus) {
            self.advance();
            let expr = self.parse_unary()?;
            // Optimize: negate literal directly
            if let Expr::Lit(v) = expr {
                return Ok(Expr::Lit(-v));
            }
            return Ok(Expr::Neg(Box::new(expr)));
        }
        self.parse_app()
    }

    /// app_expr = atom atom*
    ///
    /// If the first atom is an identifier and is followed by more atoms,
    /// this is a function application. Otherwise it's just the atom.
    fn parse_app(&mut self) -> Result<Expr, ParseError> {
        let first = self.parse_atom()?;

        // Only identifiers can be function names
        if let Expr::Var(ref name) = first {
            let func_name = name.clone();
            let mut args = Vec::new();
            // Consume atoms as arguments (stop at operators, newlines, etc.)
            while self.is_atom_start() {
                args.push(self.parse_atom()?);
            }
            if !args.is_empty() {
                return Ok(Expr::App(func_name, args));
            }
        }

        Ok(first)
    }

    /// atom = NUMBER | IDENT | '(' expr ')'
    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        if let Token::Number(v) = self.current().token {
            self.advance();
            return Ok(Expr::Lit(v));
        }
        if let Token::Ident(ref name) = self.current().token {
            let name = name.clone();
            self.advance();
            return Ok(Expr::Var(name));
        }
        if self.check(&Token::LParen) {
            self.advance();
            self.skip_newlines();
            let expr = self.parse_expr()?;
            self.skip_newlines();
            self.expect(Token::RParen)?;
            return Ok(expr);
        }
        Err(self.error(&alloc::format!(
            "expected number, identifier, or '(', got {}",
            self.current().token
        )))
    }

    // -- Helpers --

    fn current(&self) -> &Spanned {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn at_eof(&self) -> bool {
        self.current().token == Token::Eof
    }

    fn check(&self, token: &Token) -> bool {
        core::mem::discriminant(&self.current().token) == core::mem::discriminant(token)
    }

    fn check_ident(&self) -> bool {
        matches!(self.current().token, Token::Ident(_))
    }

    fn peek_next_is(&self, token: &Token) -> bool {
        if self.pos + 1 < self.tokens.len() {
            core::mem::discriminant(&self.tokens[self.pos + 1].token)
                == core::mem::discriminant(token)
        } else {
            false
        }
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        if self.check(&expected) {
            self.advance();
            Ok(())
        } else {
            Err(self.error(&alloc::format!(
                "expected {expected}, got {}",
                self.current().token
            )))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        if let Token::Ident(name) = &self.current().token {
            let name = name.clone();
            self.advance();
            Ok(name)
        } else {
            Err(self.error(&alloc::format!(
                "expected identifier, got {}",
                self.current().token
            )))
        }
    }

    fn expect_number(&mut self) -> Result<f32, ParseError> {
        if let Token::Number(v) = self.current().token {
            self.advance();
            Ok(v)
        } else {
            Err(self.error(&alloc::format!(
                "expected number, got {}",
                self.current().token
            )))
        }
    }

    fn skip_newlines(&mut self) {
        while self.check(&Token::Newline) {
            self.advance();
        }
    }

    fn skip_separators(&mut self) {
        while self.check(&Token::Newline) || self.check(&Token::Semicolon) {
            self.advance();
        }
    }

    /// Check if current token can start an atom (for function application).
    fn is_atom_start(&self) -> bool {
        matches!(
            self.current().token,
            Token::Number(_) | Token::Ident(_) | Token::LParen
        )
    }

    fn error(&self, message: &str) -> ParseError {
        let span = self.current().span;
        ParseError {
            message: String::from(message),
            span,
        }
    }
}

/// A parse error with location info.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parse error at {}:{}: {}",
            self.span.line, self.span.col, self.message
        )
    }
}
