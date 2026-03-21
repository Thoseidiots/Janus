/// KiroScene text format serializer and deserializer.

// ─── Data types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct SceneId(pub u64);

#[derive(Debug, Clone, PartialEq)]
pub struct SceneMetadata {
    pub name: String,
    pub version: u32,
}

/// A component value stored as key-value pairs (generic, not typed to ECS).
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentData {
    pub type_name: String,
    pub fields: Vec<(String, FieldValue)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Int(i64),
    Float(f32),
    Str(String),
    Vec3(f32, f32, f32),
    Quat(f32, f32, f32, f32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneEntityData {
    pub local_id: u64,
    pub components: Vec<ComponentData>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneData {
    pub metadata: SceneMetadata,
    pub entities: Vec<SceneEntityData>,
}

// ─── Error type ──────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum SceneError {
    ParseError { path: String, field: String, reason: String },
    IoError { path: String, reason: String },
}

// ─── Serializer ──────────────────────────────────────────────────────────────

pub struct SceneSerializer;

impl SceneSerializer {
    pub fn serialize(scene: &SceneData) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "scene \"{}\" version {}\n",
            escape_str(&scene.metadata.name),
            scene.metadata.version
        ));
        for entity in &scene.entities {
            out.push_str(&format!("  entity {}\n", entity.local_id));
            for comp in &entity.components {
                out.push_str("    ");
                out.push_str(&comp.type_name);
                for (key, val) in &comp.fields {
                    out.push(' ');
                    out.push_str(key);
                    out.push(' ');
                    out.push_str(&serialize_field_value(val));
                }
                out.push('\n');
            }
        }
        out
    }

    pub fn deserialize(text: &str, path: &str) -> Result<SceneData, SceneError> {
        let mut parser = Parser::new(text, path);
        parser.parse_scene()
    }
}

fn escape_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn serialize_field_value(v: &FieldValue) -> String {
    match v {
        FieldValue::Int(i) => i.to_string(),
        FieldValue::Float(f) => {
            let s = format!("{}", f);
            if s.contains('.') || s.contains('e') || s.contains('E') { s } else { format!("{}.0", s) }
        }
        FieldValue::Str(s) => format!("\"{}\"", escape_str(s)),
        FieldValue::Vec3(x, y, z) => format!("{} {} {}", fmt_f32(*x), fmt_f32(*y), fmt_f32(*z)),
        FieldValue::Quat(x, y, z, w) => format!("{} {} {} {}", fmt_f32(*x), fmt_f32(*y), fmt_f32(*z), fmt_f32(*w)),
    }
}

fn fmt_f32(v: f32) -> String {
    let s = format!("{}", v);
    if s.contains('.') || s.contains('e') || s.contains('E') { s } else { format!("{}.0", s) }
}

// ─── Tokenizer ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    StringLit(String),
    IntLit(i64),
    FloatLit(f32),
    Newline,
    Eof,
}

struct Tokenizer<'a> {
    src: &'a str,
    pos: usize,
    line: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(src: &'a str) -> Self { Tokenizer { src, pos: 0, line: 1 } }

    fn peek_char(&self) -> Option<char> { self.src[self.pos..].chars().next() }

    fn advance_char(&mut self) -> Option<char> {
        let ch = self.src[self.pos..].chars().next()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn skip_spaces(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch == ' ' || ch == '\t' || ch == '\r' { self.advance_char(); } else { break; }
        }
    }

    fn skip_comment(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch == '\n' { break; }
            self.advance_char();
        }
    }

    fn next_token(&mut self) -> Result<(Token, usize), SceneError> {
        loop {
            self.skip_spaces();
            match self.peek_char() {
                None => return Ok((Token::Eof, self.line)),
                Some('#') => { self.skip_comment(); continue; }
                Some('\n') => {
                    self.advance_char();
                    let ln = self.line;
                    self.line += 1;
                    return Ok((Token::Newline, ln));
                }
                Some('"') => {
                    let ln = self.line;
                    return Ok((Token::StringLit(self.read_string_lit()?), ln));
                }
                Some(ch) if ch == '-' || ch.is_ascii_digit() => {
                    let ln = self.line;
                    return Ok((self.read_number()?, ln));
                }
                Some(ch) if ch.is_alphabetic() || ch == '_' => {
                    let ln = self.line;
                    return Ok((Token::Ident(self.read_ident()), ln));
                }
                Some(ch) => {
                    let ln = self.line;
                    return Err(SceneError::ParseError {
                        path: String::new(),
                        field: "token".to_string(),
                        reason: format!("unexpected character '{}'", ch),
                    });
                }
            }
        }
    }

    fn read_string_lit(&mut self) -> Result<String, SceneError> {
        self.advance_char();
        let mut s = String::new();
        loop {
            match self.advance_char() {
                None => return Err(SceneError::ParseError { path: String::new(), field: "string".to_string(), reason: "unterminated string literal".into() }),
                Some('"') => break,
                Some('\\') => match self.advance_char() {
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some(c) => s.push(c),
                    None => return Err(SceneError::ParseError { path: String::new(), field: "string".to_string(), reason: "unterminated escape".into() }),
                },
                Some(c) => s.push(c),
            }
        }
        Ok(s)
    }

    fn read_number(&mut self) -> Result<Token, SceneError> {
        let start = self.pos;
        if self.peek_char() == Some('-') { self.advance_char(); }
        while self.peek_char().map(|c| c.is_ascii_digit()).unwrap_or(false) { self.advance_char(); }
        let is_float = matches!(self.peek_char(), Some('.') | Some('e') | Some('E'));
        if is_float {
            if self.peek_char() == Some('.') {
                self.advance_char();
                while self.peek_char().map(|c| c.is_ascii_digit()).unwrap_or(false) { self.advance_char(); }
            }
            if matches!(self.peek_char(), Some('e') | Some('E')) {
                self.advance_char();
                if matches!(self.peek_char(), Some('+') | Some('-')) { self.advance_char(); }
                while self.peek_char().map(|c| c.is_ascii_digit()).unwrap_or(false) { self.advance_char(); }
            }
            let s = &self.src[start..self.pos];
            s.parse::<f32>().map(Token::FloatLit).map_err(|_| SceneError::ParseError { path: String::new(), field: "number".to_string(), reason: format!("invalid float '{}'", s) })
        } else {
            let s = &self.src[start..self.pos];
            s.parse::<i64>().map(Token::IntLit).map_err(|_| SceneError::ParseError { path: String::new(), field: "number".to_string(), reason: format!("invalid integer '{}'", s) })
        }
    }

    fn read_ident(&mut self) -> String {
        let start = self.pos;
        while self.peek_char().map(|c| c.is_alphanumeric() || c == '_').unwrap_or(false) { self.advance_char(); }
        self.src[start..self.pos].to_string()
    }
}

// ─── Parser ──────────────────────────────────────────────────────────────────

struct Parser<'a> {
    tokenizer: Tokenizer<'a>,
    lookahead: Option<(Token, usize)>,
    path: String,
}

impl<'a> Parser<'a> {
    fn new(src: &'a str, path: &str) -> Self {
        Parser { tokenizer: Tokenizer::new(src), lookahead: None, path: path.to_string() }
    }

    fn peek(&mut self) -> Result<&Token, SceneError> {
        if self.lookahead.is_none() {
            let tok = self.tokenizer.next_token().map_err(|e| self.attach_path(e))?;
            self.lookahead = Some(tok);
        }
        Ok(&self.lookahead.as_ref().unwrap().0)
    }

    fn peek_line(&mut self) -> Result<usize, SceneError> {
        if self.lookahead.is_none() {
            let tok = self.tokenizer.next_token().map_err(|e| self.attach_path(e))?;
            self.lookahead = Some(tok);
        }
        Ok(self.lookahead.as_ref().unwrap().1)
    }

    fn consume(&mut self) -> Result<(Token, usize), SceneError> {
        if let Some(tok) = self.lookahead.take() { return Ok(tok); }
        self.tokenizer.next_token().map_err(|e| self.attach_path(e))
    }

    fn attach_path(&self, e: SceneError) -> SceneError {
        match e {
            SceneError::ParseError { path: _, field, reason } => SceneError::ParseError { path: self.path.clone(), field, reason },
            other => other,
        }
    }

    fn parse_error(&self, field: &str, reason: &str) -> SceneError {
        SceneError::ParseError { path: self.path.clone(), field: field.to_string(), reason: reason.to_string() }
    }

    fn skip_newlines(&mut self) -> Result<(), SceneError> {
        loop {
            match self.peek()? {
                Token::Newline => { self.consume()?; }
                _ => break,
            }
        }
        Ok(())
    }

    fn expect_newline_or_eof(&mut self) -> Result<(), SceneError> {
        match self.peek()? {
            Token::Newline | Token::Eof => { self.consume()?; Ok(()) }
            _ => {
                let ln = self.peek_line()?;
                Err(self.parse_error("newline", &format!("expected newline at line {}", ln)))
            }
        }
    }

    fn expect_ident(&mut self, expected: &str) -> Result<(), SceneError> {
        let ln = self.peek_line()?;
        match self.consume()? {
            (Token::Ident(s), _) if s == expected => Ok(()),
            (tok, _) => Err(self.parse_error(expected, &format!("expected '{}', got {:?} at line {}", expected, tok, ln))),
        }
    }

    fn expect_string(&mut self, field: &str) -> Result<String, SceneError> {
        let ln = self.peek_line()?;
        match self.consume()? {
            (Token::StringLit(s), _) => Ok(s),
            (tok, _) => Err(self.parse_error(field, &format!("expected string literal, got {:?} at line {}", tok, ln))),
        }
    }

    fn expect_u64(&mut self, field: &str) -> Result<u64, SceneError> {
        let ln = self.peek_line()?;
        match self.consume()? {
            (Token::IntLit(i), _) if i >= 0 => Ok(i as u64),
            (tok, _) => Err(self.parse_error(field, &format!("expected non-negative integer, got {:?} at line {}", tok, ln))),
        }
    }

    fn expect_u32(&mut self, field: &str) -> Result<u32, SceneError> {
        let ln = self.peek_line()?;
        match self.consume()? {
            (Token::IntLit(i), _) if i >= 0 && i <= u32::MAX as i64 => Ok(i as u32),
            (tok, _) => Err(self.parse_error(field, &format!("expected u32, got {:?} at line {}", tok, ln))),
        }
    }

    fn parse_scene(&mut self) -> Result<SceneData, SceneError> {
        self.skip_newlines()?;
        self.expect_ident("scene")?;
        let name = self.expect_string("scene.name")?;
        self.expect_ident("version")?;
        let version = self.expect_u32("scene.version")?;
        self.expect_newline_or_eof()?;

        let mut entities = Vec::new();
        loop {
            self.skip_newlines()?;
            match self.peek()? {
                Token::Eof => break,
                Token::Ident(s) if s == "entity" => { entities.push(self.parse_entity()?); }
                _ => {
                    let ln = self.peek_line()?;
                    return Err(self.parse_error("entity", &format!("expected 'entity' or end of file at line {}", ln)));
                }
            }
        }

        Ok(SceneData { metadata: SceneMetadata { name, version }, entities })
    }

    fn parse_entity(&mut self) -> Result<SceneEntityData, SceneError> {
        self.expect_ident("entity")?;
        let local_id = self.expect_u64("entity.id")?;
        self.expect_newline_or_eof()?;

        let mut components = Vec::new();
        loop {
            self.skip_newlines()?;
            match self.peek()? {
                Token::Eof => break,
                Token::Ident(s) if s == "entity" => break,
                Token::Ident(_) => { components.push(self.parse_component()?); }
                _ => break,
            }
        }

        Ok(SceneEntityData { local_id, components })
    }

    fn parse_component(&mut self) -> Result<ComponentData, SceneError> {
        let ln = self.peek_line()?;
        let type_name = match self.consume()? {
            (Token::Ident(s), _) => s,
            (tok, _) => return Err(self.parse_error("component.type_name", &format!("expected component name, got {:?} at line {}", tok, ln))),
        };

        let mut fields = Vec::new();
        loop {
            match self.peek()? {
                Token::Newline | Token::Eof => { self.consume()?; break; }
                Token::Ident(_) => { fields.push(self.parse_field(&type_name)?); }
                _ => {
                    let ln2 = self.peek_line()?;
                    return Err(self.parse_error(&type_name, &format!("expected field name or newline at line {}", ln2)));
                }
            }
        }

        Ok(ComponentData { type_name, fields })
    }

    fn parse_field(&mut self, comp_name: &str) -> Result<(String, FieldValue), SceneError> {
        let ln = self.peek_line()?;
        let key = match self.consume()? {
            (Token::Ident(s), _) => s,
            (tok, _) => return Err(self.parse_error(comp_name, &format!("expected field name, got {:?} at line {}", tok, ln))),
        };
        let value = self.parse_field_value(&key, comp_name)?;
        Ok((key, value))
    }

    fn parse_field_value(&mut self, key: &str, comp_name: &str) -> Result<FieldValue, SceneError> {
        let ln = self.peek_line()?;
        let field_id = format!("{}.{}", comp_name, key);
        let is_vec3_key = matches!(key, "position" | "scale");
        let is_quat_key = key == "rotation";

        match self.peek()? {
            Token::StringLit(_) => {
                let s = self.expect_string(&field_id)?;
                Ok(FieldValue::Str(s))
            }
            Token::IntLit(_) | Token::FloatLit(_) => {
                let first_is_int = matches!(self.peek()?, Token::IntLit(_));
                if is_vec3_key { return self.parse_vec3(&field_id); }
                if is_quat_key { return self.parse_quat(&field_id); }
                let first = self.consume_number(&field_id)?;
                match self.peek()? {
                    Token::IntLit(_) | Token::FloatLit(_) => {
                        let second = self.consume_number(&field_id)?;
                        let third = self.consume_number(&field_id)?;
                        match self.peek()? {
                            Token::IntLit(_) | Token::FloatLit(_) => {
                                let fourth = self.consume_number(&field_id)?;
                                Ok(FieldValue::Quat(first, second, third, fourth))
                            }
                            _ => Ok(FieldValue::Vec3(first, second, third)),
                        }
                    }
                    _ => {
                        if first_is_int { Ok(FieldValue::Int(first as i64)) } else { Ok(FieldValue::Float(first)) }
                    }
                }
            }
            _ => Err(self.parse_error(&field_id, &format!("expected field value at line {}", ln))),
        }
    }

    fn parse_vec3(&mut self, field: &str) -> Result<FieldValue, SceneError> {
        let x = self.consume_number(field)?;
        let y = self.consume_number(field)?;
        let z = self.consume_number(field)?;
        Ok(FieldValue::Vec3(x, y, z))
    }

    fn parse_quat(&mut self, field: &str) -> Result<FieldValue, SceneError> {
        let x = self.consume_number(field)?;
        let y = self.consume_number(field)?;
        let z = self.consume_number(field)?;
        let w = self.consume_number(field)?;
        Ok(FieldValue::Quat(x, y, z, w))
    }

    fn consume_number(&mut self, field: &str) -> Result<f32, SceneError> {
        let ln = self.peek_line()?;
        match self.consume()? {
            (Token::IntLit(i), _) => Ok(i as f32),
            (Token::FloatLit(f), _) => Ok(f),
            (tok, _) => Err(self.parse_error(field, &format!("expected number, got {:?} at line {}", tok, ln))),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Quat};

    fn sample_scene() -> SceneData {
        SceneData {
            metadata: SceneMetadata { name: "level_01".into(), version: 1 },
            entities: vec![
                SceneEntityData {
                    local_id: 1,
                    components: vec![
                        ComponentData {
                            type_name: "transform".into(),
                            fields: vec![
                                ("position".into(), FieldValue::Vec3(0.0, 1.0, 0.0)),
                                ("rotation".into(), FieldValue::Quat(0.0, 0.0, 0.0, 1.0)),
                                ("scale".into(), FieldValue::Vec3(1.0, 1.0, 1.0)),
                            ],
                        },
                        ComponentData {
                            type_name: "mesh_renderer".into(),
                            fields: vec![
                                ("mesh".into(), FieldValue::Str("assets/cube.glb".into())),
                                ("material".into(), FieldValue::Str("assets/stone.mat".into())),
                            ],
                        },
                    ],
                },
                SceneEntityData {
                    local_id: 2,
                    components: vec![
                        ComponentData {
                            type_name: "rigid_body".into(),
                            fields: vec![
                                ("mass".into(), FieldValue::Float(1.0)),
                                ("linear_damping".into(), FieldValue::Float(0.1)),
                            ],
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn serialize_produces_valid_text() {
        let scene = sample_scene();
        let text = SceneSerializer::serialize(&scene);
        assert!(text.starts_with("scene \"level_01\" version 1\n"));
        assert!(text.contains("  entity 1\n"));
        assert!(text.contains("  entity 2\n"));
        assert!(text.contains("transform"));
        assert!(text.contains("mesh_renderer"));
    }

    #[test]
    fn round_trip_serialize_deserialize() {
        let original = sample_scene();
        let text = SceneSerializer::serialize(&original);
        let restored = SceneSerializer::deserialize(&text, "test.ks").unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn round_trip_with_strings() {
        let original = SceneData {
            metadata: SceneMetadata { name: "assets_scene".into(), version: 1 },
            entities: vec![SceneEntityData {
                local_id: 1,
                components: vec![ComponentData {
                    type_name: "mesh_renderer".into(),
                    fields: vec![
                        ("mesh".into(), FieldValue::Str("assets/cube.glb".into())),
                        ("material".into(), FieldValue::Str("assets/stone.mat".into())),
                    ],
                }],
            }],
        };
        let text = SceneSerializer::serialize(&original);
        let restored = SceneSerializer::deserialize(&text, "test.ks").unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn round_trip_empty_scene() {
        let original = SceneData {
            metadata: SceneMetadata { name: "empty".into(), version: 0 },
            entities: vec![],
        };
        let text = SceneSerializer::serialize(&original);
        let restored = SceneSerializer::deserialize(&text, "empty.ks").unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn round_trip_integer_field() {
        let original = SceneData {
            metadata: SceneMetadata { name: "int_test".into(), version: 1 },
            entities: vec![SceneEntityData {
                local_id: 1,
                components: vec![ComponentData {
                    type_name: "counter".into(),
                    fields: vec![("count".into(), FieldValue::Int(42))],
                }],
            }],
        };
        let text = SceneSerializer::serialize(&original);
        let restored = SceneSerializer::deserialize(&text, "int.ks").unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn malformed_header_missing_scene_keyword() {
        let text = "\"level_01\" version 1\n  entity 1\n";
        let result = SceneSerializer::deserialize(text, "bad.ks");
        assert!(matches!(result, Err(SceneError::ParseError { .. })));
    }

    #[test]
    fn malformed_header_missing_version_keyword() {
        let text = "scene \"level_01\"\n  entity 1\n";
        let result = SceneSerializer::deserialize(text, "bad.ks");
        assert!(matches!(result, Err(SceneError::ParseError { .. })));
    }

    #[test]
    fn malformed_header_float_version() {
        let text = "scene \"level_01\" version 1.5\n";
        let result = SceneSerializer::deserialize(text, "bad.ks");
        assert!(matches!(result, Err(SceneError::ParseError { .. })));
    }

    #[test]
    fn malformed_entity_missing_id() {
        let text = "scene \"s\" version 1\nentity\n";
        let result = SceneSerializer::deserialize(text, "bad.ks");
        assert!(matches!(result, Err(SceneError::ParseError { .. })));
    }

    #[test]
    fn parse_error_on_empty_input() {
        let result = SceneSerializer::deserialize("", "empty.ks");
        assert!(matches!(result, Err(SceneError::ParseError { .. })));
    }

    // Property 5: Scene Serialization Round-Trip
    // Validates: Requirements 2.1, 8.6
    #[test]
    fn property_scene_serialization_round_trip() {
        use std::collections::HashSet;
        
        // Test with various scene configurations
        let test_scenes = vec![
            // Empty scene
            SceneData {
                metadata: SceneMetadata { name: "empty".into(), version: 1 },
                entities: vec![],
            },
            // Single entity with one component
            SceneData {
                metadata: SceneMetadata { name: "simple".into(), version: 1 },
                entities: vec![SceneEntityData {
                    local_id: 1,
                    components: vec![ComponentData {
                        type_name: "transform".into(),
                        fields: vec![("position".into(), FieldValue::Vec3(1.0, 2.0, 3.0))],
                    }],
                }],
            },
            // Multiple entities with multiple components
            SceneData {
                metadata: SceneMetadata { name: "complex".into(), version: 2 },
                entities: vec![
                    SceneEntityData {
                        local_id: 1,
                        components: vec![
                            ComponentData {
                                type_name: "transform".into(),
                                fields: vec![
                                    ("position".into(), FieldValue::Vec3(0.0, 0.0, 0.0)),
                                    ("rotation".into(), FieldValue::Quat(0.0, 0.0, 0.0, 1.0)),
                                ],
                            },
                            ComponentData {
                                type_name: "mesh_renderer".into(),
                                fields: vec![
                                    ("mesh".into(), FieldValue::Str("cube.glb".into())),
                                    ("material".into(), FieldValue::Str("stone.mat".into())),
                                ],
                            },
                        ],
                    },
                    SceneEntityData {
                        local_id: 2,
                        components: vec![ComponentData {
                            type_name: "light".into(),
                            fields: vec![
                                ("type".into(), FieldValue::Str("directional".into())),
                                ("color".into(), FieldValue::Vec3(1.0, 1.0, 0.9)),
                                ("intensity".into(), FieldValue::Float(2.5)),
                            ],
                        }],
                    },
                ],
            },
        ];
        
        for original in test_scenes {
            // Serialize to text
            let text = SceneSerializer::serialize(&original);
            
            // Verify text is not empty and contains expected structure
            assert!(!text.is_empty());
            assert!(text.contains("scene"));
            assert!(text.contains("version"));
            
            // Deserialize back
            let restored = SceneSerializer::deserialize(&text, "test.ks").unwrap();
            
            // Verify exact round-trip equality
            assert_eq!(original, restored, "Scene round-trip failed for: {}", original.metadata.name);
            
            // Verify metadata preservation
            assert_eq!(original.metadata.name, restored.metadata.name);
            assert_eq!(original.metadata.version, restored.metadata.version);
            
            // Verify entity count
            assert_eq!(original.entities.len(), restored.entities.len());
            
            // Verify all entities are preserved
            let original_ids: HashSet<_> = original.entities.iter().map(|e| e.local_id).collect();
            let restored_ids: HashSet<_> = restored.entities.iter().map(|e| e.local_id).collect();
            assert_eq!(original_ids, restored_ids);
            
            // Verify component data integrity
            for (orig_entity, rest_entity) in original.entities.iter().zip(restored.entities.iter()) {
                assert_eq!(orig_entity.local_id, rest_entity.local_id);
                assert_eq!(orig_entity.components.len(), rest_entity.components.len());
                
                for (orig_comp, rest_comp) in orig_entity.components.iter().zip(rest_entity.components.iter()) {
                    assert_eq!(orig_comp.type_name, rest_comp.type_name);
                    assert_eq!(orig_comp.fields.len(), rest_comp.fields.len());
                    
                    for (orig_field, rest_field) in orig_comp.fields.iter().zip(rest_comp.fields.iter()) {
                        assert_eq!(orig_field.0, rest_field.0); // field name
                        assert_eq!(orig_field.1, rest_field.1); // field value
                    }
                }
            }
        }
    }
}
