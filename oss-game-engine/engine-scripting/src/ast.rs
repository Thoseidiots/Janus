use crate::Token;

#[derive(Debug, PartialEq, Clone)]
pub enum Expr {
    Number(f64),
    String(String),
    Ident(String),
    Binary(Box<Expr>, Token, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Stmt {
    Expr(Expr),
    Let(String, Expr),
    Return(Expr),
    If(Expr, Vec<Stmt>, Option<Vec<Stmt>>),
    While(Expr, Vec<Stmt>),
}

#[derive(Debug, PartialEq, Clone)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<String>,
    pub body: Vec<Stmt>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct Ast {
    pub functions: Vec<FnDecl>,
}

pub struct PrettyPrinter;

impl PrettyPrinter {
    pub fn print(ast: &Ast) -> String {
        let mut out = String::new();
        for f in &ast.functions {
            out.push_str(&Self::print_fn(f));
            out.push_str("\n");
        }
        out
    }

    fn print_fn(f: &FnDecl) -> String {
        let params = f.params.join(", ");
        let mut out = format!("fn {}({}) {{\n", f.name, params);
        for stmt in &f.body {
            out.push_str(&format!("  {}\n", Self::print_stmt(stmt)));
        }
        out.push_str("}\n");
        out
    }

    fn print_stmt(stmt: &Stmt) -> String {
        match stmt {
            Stmt::Expr(e) => format!("{};", Self::print_expr(e)),
            Stmt::Let(n, e) => format!("let {} = {};", n, Self::print_expr(e)),
            Stmt::Return(e) => format!("return {};", Self::print_expr(e)),
            Stmt::If(c, t, e) => {
                let true_branch: Vec<String> = t.iter().map(Self::print_stmt).collect();
                let mut out = format!("if {} {{\n  {}\n}}", Self::print_expr(c), true_branch.join("\n  "));
                if let Some(elb) = e {
                    let false_branch: Vec<String> = elb.iter().map(Self::print_stmt).collect();
                    out.push_str(&format!(" else {{\n  {}\n}}", false_branch.join("\n  ")));
                }
                out
            }
            Stmt::While(c, b) => {
                let body: Vec<String> = b.iter().map(Self::print_stmt).collect();
                format!("while {} {{\n  {}\n}}", Self::print_expr(c), body.join("\n  "))
            }
        }
    }

    fn print_expr(expr: &Expr) -> String {
        match expr {
            Expr::Number(n) => n.to_string(),
            Expr::String(s) => format!("\"{}\"", s),
            Expr::Ident(i) => i.clone(),
            Expr::Binary(l, op, r) => {
                let op_str = match op {
                    Token::Plus => "+", Token::Minus => "-",
                    Token::Star => "*", Token::Slash => "/",
                    Token::EqualEqual => "==", Token::BangEqual => "!=",
                    Token::Less => "<", Token::LessEqual => "<=",
                    Token::Greater => ">", Token::GreaterEqual => ">=",
                    _ => "?",
                };
                format!("({} {} {})", Self::print_expr(l), op_str, Self::print_expr(r))
            }
            Expr::Call(n, args) => {
                let arg_strs: Vec<String> = args.iter().map(Self::print_expr).collect();
                format!("{}({})", n, arg_strs.join(", "))
            }
        }
    }
}
