use crate::Token;
use crate::ast::{Ast, FnDecl, Stmt, Expr};

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, current: 0 }
    }

    pub fn parse(&mut self) -> Result<Ast, String> {
        let mut functions = Vec::new();
        while !self.is_at_end() {
            functions.push(self.declaration()?);
        }
        Ok(Ast { functions })
    }

    fn declaration(&mut self) -> Result<FnDecl, String> {
        if self.match_token(Token::Fn) {
            self.function("function")
        } else {
            Err("Expected function declaration".to_string())
        }
    }

    fn function(&mut self, _kind: &str) -> Result<FnDecl, String> {
        let name = match self.advance() {
            Token::Ident(n) => n.clone(),
            _ => return Err("Expected function name".to_string()),
        };

        self.consume(Token::LParen, "Expected '(' after function name")?;
        
        let mut params = Vec::new();
        if !self.check(Token::RParen) {
            loop {
                match self.advance() {
                    Token::Ident(n) => params.push(n.clone()),
                    _ => return Err("Expected parameter name".to_string()),
                }
                if !self.match_token(Token::Comma) { break; }
            }
        }
        self.consume(Token::RParen, "Expected ')' after parameters")?;
        self.consume(Token::LBrace, "Expected '{' before function body")?;
        
        let body = self.block()?;
        Ok(FnDecl { name, params, body })
    }

    fn block(&mut self) -> Result<Vec<Stmt>, String> {
        let mut statements = Vec::new();
        while !self.check(Token::RBrace) && !self.is_at_end() {
            statements.push(self.statement()?);
        }
        self.consume(Token::RBrace, "Expected '}' after block")?;
        Ok(statements)
    }

    fn statement(&mut self) -> Result<Stmt, String> {
        if self.match_token(Token::Let) {
            let name = match self.advance() {
                Token::Ident(n) => n.clone(),
                _ => return Err("Expected variable name".to_string()),
            };
            self.consume(Token::Equal, "Expected '=' after variable name")?;
            let initializer = self.expression()?;
            self.consume(Token::Semicolon, "Expected ';' after let declaration")?;
            Ok(Stmt::Let(name, initializer))
        } else if self.match_token(Token::Return) {
            let value = self.expression()?;
            self.consume(Token::Semicolon, "Expected ';' after return value")?;
            Ok(Stmt::Return(value))
        } else if self.match_token(Token::If) {
            self.consume(Token::LParen, "Expected '(' after 'if'")?;
            let condition = self.expression()?;
            self.consume(Token::RParen, "Expected ')' after if condition")?;
            self.consume(Token::LBrace, "Expected '{' after if condition")?;
            let then_branch = self.block()?;
            
            let else_branch = if self.match_token(Token::Else) {
                self.consume(Token::LBrace, "Expected '{' after 'else'")?;
                Some(self.block()?)
            } else {
                None
            };
            Ok(Stmt::If(condition, then_branch, else_branch))
        } else if self.match_token(Token::While) {
            self.consume(Token::LParen, "Expected '(' after 'while'")?;
            let condition = self.expression()?;
            self.consume(Token::RParen, "Expected ')' after while condition")?;
            self.consume(Token::LBrace, "Expected '{' after while condition")?;
            let body = self.block()?;
            Ok(Stmt::While(condition, body))
        } else {
            let expr = self.expression()?;
            self.consume(Token::Semicolon, "Expected ';' after expression")?;
            Ok(Stmt::Expr(expr))
        }
    }

    fn expression(&mut self) -> Result<Expr, String> {
        self.equality()
    }

    fn equality(&mut self) -> Result<Expr, String> {
        let mut expr = self.comparison()?;
        while self.match_token(Token::EqualEqual) || self.match_token(Token::BangEqual) {
            let operator = self.previous().clone();
            let right = self.comparison()?;
            expr = Expr::Binary(Box::new(expr), operator, Box::new(right));
        }
        Ok(expr)
    }

    fn comparison(&mut self) -> Result<Expr, String> {
        let mut expr = self.term()?;
        while self.match_token(Token::Greater) || self.match_token(Token::GreaterEqual) || 
              self.match_token(Token::Less) || self.match_token(Token::LessEqual) {
            let operator = self.previous().clone();
            let right = self.term()?;
            expr = Expr::Binary(Box::new(expr), operator, Box::new(right));
        }
        Ok(expr)
    }

    fn term(&mut self) -> Result<Expr, String> {
        let mut expr = self.factor()?;
        while self.match_token(Token::Minus) || self.match_token(Token::Plus) {
            let operator = self.previous().clone();
            let right = self.factor()?;
            expr = Expr::Binary(Box::new(expr), operator, Box::new(right));
        }
        Ok(expr)
    }

    fn factor(&mut self) -> Result<Expr, String> {
        let mut expr = self.call()?;
        while self.match_token(Token::Slash) || self.match_token(Token::Star) {
            let operator = self.previous().clone();
            let right = self.call()?;
            expr = Expr::Binary(Box::new(expr), operator, Box::new(right));
        }
        Ok(expr)
    }

    fn call(&mut self) -> Result<Expr, String> {
        let primary = self.primary()?;
        
        if self.match_token(Token::LParen) {
            if let Expr::Ident(name) = primary {
                let mut args = Vec::new();
                if !self.check(Token::RParen) {
                    loop {
                        args.push(self.expression()?);
                        if !self.match_token(Token::Comma) { break; }
                    }
                }
                self.consume(Token::RParen, "Expected ')' after arguments")?;
                return Ok(Expr::Call(name, args));
            } else {
                return Err("Can only call identifiers".to_string());
            }
        }
        
        Ok(primary)
    }

    fn primary(&mut self) -> Result<Expr, String> {
        if self.match_token(Token::False) { return Ok(Expr::Number(0.0)); } // Simplified
        if self.match_token(Token::True) { return Ok(Expr::Number(1.0)); }
        
        match self.advance().clone() {
            Token::Number(n) => Ok(Expr::Number(n)),
            Token::String(s) => Ok(Expr::String(s)),
            Token::Ident(i) => Ok(Expr::Ident(i)),
            Token::LParen => {
                let expr = self.expression()?;
                self.consume(Token::RParen, "Expected ')' after expression")?;
                Ok(expr)
            }
            _ => Err("Expected expression".to_string()),
        }
    }

    fn match_token(&mut self, token: Token) -> bool {
        if self.check(token.clone()) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn check(&self, token: Token) -> bool {
        if self.is_at_end() { return false; }
        self.peek() == &token
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() { self.current += 1; }
        self.previous()
    }

    fn is_at_end(&self) -> bool {
        self.peek() == &Token::Eof
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.current - 1]
    }

    fn consume(&mut self, token: Token, message: &str) -> Result<&Token, String> {
        if self.check(token) {
            Ok(self.advance())
        } else {
            Err(message.to_string())
        }
    }
}
