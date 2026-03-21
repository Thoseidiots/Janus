// engine-scripting/src/lib.rs

pub mod ast;
pub mod parser;
pub mod compiler;
pub mod vm;

// --- Tokenizer / Lexer ---

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords
    Fn, Let, If, Else, Return, True, False, While,

    // Identifiers and literals
    Ident(String),
    Number(f64),
    String(String),

    // Operators
    Plus, Minus, Star, Slash,
    Equal, EqualEqual, Bang, BangEqual,
    Less, LessEqual, Greater, GreaterEqual,

    // Delimiters
    LParen, RParen, LBrace, RBrace, Comma, Semicolon,

    // End of file
    Eof,
}

pub struct Lexer {
    source: Vec<char>,
    start: usize,
    current: usize,
    line: usize,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Lexer {
            source: source.chars().collect(),
            start: 0,
            current: 0,
            line: 1,
        }
    }

    pub fn scan_token(&mut self) -> Token {
        self.skip_whitespace();
        self.start = self.current;

        if self.is_at_end() {
            return Token::Eof;
        }

        let c = self.advance();

        if c.is_alphabetic() || c == '_' {
            return self.identifier();
        }

        if c.is_digit(10) {
            return self.number();
        }

        match c {
            '(' => Token::LParen,
            ')' => Token::RParen,
            '{' => Token::LBrace,
            '}' => Token::RBrace,
            ',' => Token::Comma,
            ';' => Token::Semicolon,
            '+' => Token::Plus,
            '-' => Token::Minus,
            '*' => Token::Star,
            '/' => Token::Slash,
            '=' => {
                if self.match_char('=') {
                    Token::EqualEqual
                } else {
                    Token::Equal
                }
            }
            '!' => {
                if self.match_char('=') {
                    Token::BangEqual
                } else {
                    Token::Bang
                }
            }
            '<' => {
                if self.match_char('=') {
                    Token::LessEqual
                } else {
                    Token::Less
                }
            }
            '>' => {
                if self.match_char('=') {
                    Token::GreaterEqual
                } else {
                    Token::Greater
                }
            }
            '"' => self.string(),
            _ => panic!("Unexpected character on line {}", self.line),
        }
    }

    fn skip_whitespace(&mut self) {
        while !self.is_at_end() {
            match self.peek() {
                ' ' | '\r' | '\t' => {
                    self.advance();
                }
                '\n' => {
                    self.line += 1;
                    self.advance();
                }
                '/' => {
                    if self.peek_next() == '/' {
                        while self.peek() != '\n' && !self.is_at_end() {
                            self.advance();
                        }
                    } else {
                        return;
                    }
                }
                _ => return,
            }
        }
    }

    fn identifier(&mut self) -> Token {
        while self.peek().is_alphanumeric() || self.peek() == '_' {
            self.advance();
        }

        let text: String = self.source[self.start..self.current].iter().collect();
        match text.as_str() {
            "fn" => Token::Fn,
            "let" => Token::Let,
            "if" => Token::If,
            "else" => Token::Else,
            "return" => Token::Return,
            "true" => Token::True,
            "false" => Token::False,
            "while" => Token::While,
            _ => Token::Ident(text),
        }
    }

    fn number(&mut self) -> Token {
        while self.peek().is_digit(10) {
            self.advance();
        }

        if self.peek() == '.' && self.peek_next().is_digit(10) {
            self.advance(); // Consume the '.'
            while self.peek().is_digit(10) {
                self.advance();
            }
        }

        let text: String = self.source[self.start..self.current].iter().collect();
        Token::Number(text.parse().unwrap())
    }

    fn string(&mut self) -> Token {
        while self.peek() != '"' && !self.is_at_end() {
            if self.peek() == '\n' {
                self.line += 1;
            }
            self.advance();
        }

        if self.is_at_end() {
            panic!("Unterminated string on line {}", self.line);
        }

        self.advance(); // The closing ".

        let value: String = self.source[self.start + 1..self.current - 1].iter().collect();
        Token::String(value)
    }

    fn advance(&mut self) -> char {
        self.current += 1;
        self.source[self.current - 1]
    }

    fn match_char(&mut self, expected: char) -> bool {
        if self.is_at_end() || self.source[self.current] != expected {
            false
        } else {
            self.current += 1;
            true
        }
    }

    fn peek(&self) -> char {
        if self.is_at_end() {
            '\0'
        } else {
            self.source[self.current]
        }
    }

    fn peek_next(&self) -> char {
        if self.current + 1 >= self.source.len() {
            '\0'
        } else {
            self.source[self.current + 1]
        }
    }

    fn is_at_end(&self) -> bool {
        self.current >= self.source.len()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokens() {
        let source = "( ) { } , ; + - * / = == ! != < <= > >=";
        let mut lexer = Lexer::new(source);
        let tokens: Vec<Token> = std::iter::from_fn(|| {
            let token = lexer.scan_token();
            if token == Token::Eof { None } else { Some(token) }
        }).collect();

        assert_eq!(tokens, vec![
            Token::LParen, Token::RParen, Token::LBrace, Token::RBrace, Token::Comma, Token::Semicolon,
            Token::Plus, Token::Minus, Token::Star, Token::Slash, Token::Equal, Token::EqualEqual,
            Token::Bang, Token::BangEqual, Token::Less, Token::LessEqual, Token::Greater, Token::GreaterEqual
        ]);
    }

    #[test]
    fn test_keywords_and_identifiers() {
        let source = "fn my_function let x = 10; return true;";
        let mut lexer = Lexer::new(source);
        let tokens: Vec<Token> = std::iter::from_fn(|| {
            let token = lexer.scan_token();
            if token == Token::Eof { None } else { Some(token) }
        }).collect();

        assert_eq!(tokens, vec![
            Token::Fn, Token::Ident("my_function".to_string()), Token::Let, Token::Ident("x".to_string()),
            Token::Equal, Token::Number(10.0), Token::Semicolon, Token::Return, Token::True, Token::Semicolon
        ]);
    }
    
    #[test]
    fn test_string_literal() {
        let source = "let message = \"Hello, world!\";";
        let mut lexer = Lexer::new(source);
        let tokens: Vec<Token> = std::iter::from_fn(|| {
            let token = lexer.scan_token();
            if token == Token::Eof { None } else { Some(token) }
        }).collect();

        assert_eq!(tokens, vec![
            Token::Let, Token::Ident("message".to_string()), Token::Equal, 
            Token::String("Hello, world!".to_string()), Token::Semicolon
        ]);
    }
    
    #[test]
    fn test_comments() {
        let source = "// this is a comment\nlet x = 10; // another comment";
        let mut lexer = Lexer::new(source);
        let tokens: Vec<Token> = std::iter::from_fn(|| {
            let token = lexer.scan_token();
            if token == Token::Eof { None } else { Some(token) }
        }).collect();

        assert_eq!(tokens, vec![
            Token::Let, Token::Ident("x".to_string()), Token::Equal, Token::Number(10.0), Token::Semicolon
        ]);
    }
    
    // Property 22: Loom Script AST Round-Trip
    // Validates: Requirements 6.7, 6.9
    #[test]
    fn property_loom_ast_round_trip() {
        let source_code = "\
fn factorial(n) {
  if (n <= 1) {
    return 1;
  } else {
    return (n * factorial((n - 1)));
  }
}
";
        let mut lexer = Lexer::new(source_code);
        let mut tokens = Vec::new();
        loop {
            let t = lexer.scan_token();
            tokens.push(t.clone());
            if let Token::Eof = t { break; }
        }
        
        let mut parser = crate::parser::Parser::new(tokens);
        let ast = parser.parse().expect("Failed to parse AST");
        
        // Pretty print AST back to source string
        let pretty = crate::ast::PrettyPrinter::print(&ast);
        
        // Lex and parse the pretty-printed string back
        let mut lexer2 = Lexer::new(&pretty);
        let mut tokens2 = Vec::new();
        loop {
            let t = lexer2.scan_token();
            tokens2.push(t.clone());
            if let Token::Eof = t { break; }
        }
        
        let mut parser2 = crate::parser::Parser::new(tokens2);
        let ast2 = parser2.parse().expect("Failed to parse AST from pretty-printed source");
        
        
        // Check structural equality of round-tripped AST
        assert_eq!(ast, ast2);
    }

    // Property 19: Script on_start Called Exactly Once
    // Validates: Requirements 6.2
    #[test]
    fn property_script_on_start_called_exactly_once() {
        let mut runtime = crate::vm::ScriptingRuntime::new();
        runtime.on_start();
        assert_eq!(runtime.start_calls, 1);
    }

    // Property 20: Script on_update Called on All Active Scripts
    // Validates: Requirements 6.3
    #[test]
    fn property_script_on_update_called_on_all_active_scripts() {
        let mut runtime = crate::vm::ScriptingRuntime::new();
        runtime.active_scripts.push(std::path::PathBuf::from("script1.loom"));
        runtime.active_scripts.push(std::path::PathBuf::from("script2.loom"));
        runtime.on_update(0.16);
        assert_eq!(runtime.update_calls, 2);
    }

    // Property 21: Bytecode Cache Invalidation
    // Validates: Requirements 6.6
    #[test]
    fn property_bytecode_cache_invalidation() {
        let mut runtime = crate::vm::ScriptingRuntime::new();
        let path = std::path::PathBuf::from("test.loom");
        
        // Simulate caching
        runtime.cache.insert(path.clone(), (std::time::SystemTime::now(), vec![]));
        assert!(runtime.is_cached(&path));
        
        // Invalidate
        runtime.uncache(&path);
        assert!(!runtime.is_cached(&path));
    }
}
