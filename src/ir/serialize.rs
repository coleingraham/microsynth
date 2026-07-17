//! Serialization for [`IrSynthDef`]: a canonical hand-rolled binary form (the
//! production interchange), a hand-rolled JSON text form (dev/authoring), and a
//! content hash over the canonical binary layout.
//!
//! Hand-rolled to preserve the crate's zero-dependency policy (precedent: the
//! hand-rolled WAV writer in the CLI). Both forms round-trip; the binary form
//! is the canonical one the content hash is computed over.

use super::{IrEdge, IrNode, IrParam, IrSynthDef, SynthDefClass};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Magic prefix for the binary format.
const MAGIC: &[u8] = b"MICROSYNTH-IR";

/// Node discriminant tags, shared by the binary encoder/decoder and the content
/// hash so the wire values live in one place.
const TAG_UGEN: u8 = 0;
const TAG_CONST: u8 = 1;
const TAG_PARAM: u8 = 2;

/// Errors from decoding a serialized IR.
#[derive(Debug, Clone, PartialEq)]
pub enum IrCodecError {
    /// The binary magic prefix did not match.
    BadMagic,
    /// The format version is newer than this build understands.
    UnsupportedVersion(u16),
    /// The input ended before a full record could be read.
    UnexpectedEof,
    /// A tag/enum discriminant was not a known value.
    BadTag(&'static str, u8),
    /// A string field was not valid UTF-8.
    BadUtf8,
    /// The JSON text was malformed (with a short reason).
    BadJson(String),
}

impl fmt::Display for IrCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrCodecError::BadMagic => write!(f, "bad magic (not a MICROSYNTH-IR stream)"),
            IrCodecError::UnsupportedVersion(v) => write!(f, "unsupported IR format version {v}"),
            IrCodecError::UnexpectedEof => write!(f, "unexpected end of input"),
            IrCodecError::BadTag(what, v) => write!(f, "invalid {what} tag {v}"),
            IrCodecError::BadUtf8 => write!(f, "invalid UTF-8 in string field"),
            IrCodecError::BadJson(msg) => write!(f, "malformed JSON: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Binary encoding
// ---------------------------------------------------------------------------

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}
fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

/// A cursor reading little-endian records with bounds checks.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], IrCodecError> {
        let end = self.pos.checked_add(n).ok_or(IrCodecError::UnexpectedEof)?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or(IrCodecError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }
    fn u16(&mut self) -> Result<u16, IrCodecError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32, IrCodecError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn usize32(&mut self) -> Result<usize, IrCodecError> {
        Ok(self.u32()? as usize)
    }
    fn f32(&mut self) -> Result<f32, IrCodecError> {
        Ok(f32::from_bits(u32::from_le_bytes(
            self.take(4)?.try_into().unwrap(),
        )))
    }
    fn u8(&mut self) -> Result<u8, IrCodecError> {
        Ok(self.take(1)?[0])
    }
    fn string(&mut self) -> Result<String, IrCodecError> {
        let len = self.usize32()?;
        let bytes = self.take(len)?;
        core::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|_| IrCodecError::BadUtf8)
    }
}

fn class_tag(c: SynthDefClass) -> u8 {
    match c {
        SynthDefClass::Source => 0,
        SynthDefClass::Effect => 1,
    }
}
fn class_from_tag(t: u8) -> Result<SynthDefClass, IrCodecError> {
    match t {
        0 => Ok(SynthDefClass::Source),
        1 => Ok(SynthDefClass::Effect),
        other => Err(IrCodecError::BadTag("class", other)),
    }
}

fn class_str(c: SynthDefClass) -> &'static str {
    match c {
        SynthDefClass::Source => "Source",
        SynthDefClass::Effect => "Effect",
    }
}
fn class_from_str(s: &str) -> Option<SynthDefClass> {
    match s {
        "Source" => Some(SynthDefClass::Source),
        "Effect" => Some(SynthDefClass::Effect),
        _ => None,
    }
}

impl IrSynthDef {
    /// Encode to the canonical binary form (magic, version, length-prefixed
    /// sections).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        put_u16(&mut out, self.format_version);
        out.push(class_tag(self.class));
        put_u16(&mut out, self.output_channels);
        put_str(&mut out, &self.name);

        put_u32(&mut out, self.nodes.len() as u32);
        for node in &self.nodes {
            match node {
                IrNode::UGen { kind, consts } => {
                    out.push(TAG_UGEN);
                    put_str(&mut out, kind);
                    put_u32(&mut out, consts.len() as u32);
                    for &(input, value) in consts {
                        put_u32(&mut out, input);
                        put_f32(&mut out, value);
                    }
                }
                IrNode::Const(v) => {
                    out.push(TAG_CONST);
                    put_f32(&mut out, *v);
                }
                IrNode::Param { name, default } => {
                    out.push(TAG_PARAM);
                    put_str(&mut out, name);
                    put_f32(&mut out, *default);
                }
            }
        }

        put_u32(&mut out, self.edges.len() as u32);
        for e in &self.edges {
            put_u32(&mut out, e.from as u32);
            put_u32(&mut out, e.to as u32);
            put_u32(&mut out, e.to_input as u32);
        }

        put_u32(&mut out, self.params.len() as u32);
        for p in &self.params {
            put_str(&mut out, &p.name);
            put_u32(&mut out, p.node as u32);
            put_u32(&mut out, p.input as u32);
            put_f32(&mut out, p.default);
        }

        put_u32(&mut out, self.audio_inputs.len() as u32);
        for (name, node) in &self.audio_inputs {
            put_str(&mut out, name);
            put_u32(&mut out, *node as u32);
        }

        put_u32(&mut out, self.output_node as u32);
        out
    }

    /// Decode from the canonical binary form.
    pub fn from_bytes(bytes: &[u8]) -> Result<IrSynthDef, IrCodecError> {
        let mut r = Reader::new(bytes);
        if r.take(MAGIC.len())? != MAGIC {
            return Err(IrCodecError::BadMagic);
        }
        let format_version = r.u16()?;
        if format_version > super::FORMAT_VERSION {
            return Err(IrCodecError::UnsupportedVersion(format_version));
        }
        let class = class_from_tag(r.u8()?)?;
        let output_channels = r.u16()?;
        let name = r.string()?;

        let node_count = r.usize32()?;
        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            let tag = r.u8()?;
            let node = match tag {
                TAG_UGEN => {
                    let kind = r.string()?;
                    let nc = r.usize32()?;
                    let mut consts = Vec::with_capacity(nc);
                    for _ in 0..nc {
                        consts.push((r.u32()?, r.f32()?));
                    }
                    IrNode::UGen { kind, consts }
                }
                TAG_CONST => IrNode::Const(r.f32()?),
                TAG_PARAM => {
                    let name = r.string()?;
                    let default = r.f32()?;
                    IrNode::Param { name, default }
                }
                other => return Err(IrCodecError::BadTag("node", other)),
            };
            nodes.push(node);
        }

        let edge_count = r.usize32()?;
        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            edges.push(IrEdge {
                from: r.usize32()?,
                to: r.usize32()?,
                to_input: r.usize32()?,
            });
        }

        let param_count = r.usize32()?;
        let mut params = Vec::with_capacity(param_count);
        for _ in 0..param_count {
            params.push(IrParam {
                name: r.string()?,
                node: r.usize32()?,
                input: r.usize32()?,
                default: r.f32()?,
            });
        }

        let ai_count = r.usize32()?;
        let mut audio_inputs = Vec::with_capacity(ai_count);
        for _ in 0..ai_count {
            audio_inputs.push((r.string()?, r.usize32()?));
        }

        let output_node = r.usize32()?;

        Ok(IrSynthDef {
            format_version,
            name,
            class,
            output_channels,
            nodes,
            edges,
            params,
            audio_inputs,
            output_node,
        })
    }
}

// ---------------------------------------------------------------------------
// Content hash (FNV-1a-128 over a canonical byte stream)
// ---------------------------------------------------------------------------

const FNV_OFFSET_128: u128 = 0x6c62272e07bb0142_62b821756295c58d;
const FNV_PRIME_128: u128 = 0x0000000001000000_000000000000013B;

struct Fnv128(u128);
impl Fnv128 {
    fn new() -> Self {
        Fnv128(FNV_OFFSET_128)
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u128;
            self.0 = self.0.wrapping_mul(FNV_PRIME_128);
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.write(&v.to_le_bytes());
    }
    fn write_f32(&mut self, v: f32) {
        // Hash the bit pattern; normalize the two NaN-free zeros so +0/-0 hash
        // equal (they render identically).
        let bits = if v == 0.0 { 0 } else { v.to_bits() };
        self.write(&bits.to_le_bytes());
    }
    fn finish(self) -> u128 {
        self.0
    }
}

impl IrSynthDef {
    /// A stable 128-bit content hash over the canonical structure.
    ///
    /// With `include_values = false` the hash is **topology-only** — kinds and
    /// wiring, excluding const/param/inline values — so two graphs that differ
    /// only in their parameter values hash alike. With `include_values = true`
    /// it also folds in every constant and default, the variant used for exact
    /// dedup.
    ///
    /// Independent of node `name`s and of the format version, so cosmetic
    /// renames and version bumps that preserve structure do not perturb it.
    pub fn content_hash(&self, include_values: bool) -> u128 {
        let mut h = Fnv128::new();
        h.write_u32(class_tag(self.class) as u32);
        h.write_u32(self.output_channels as u32);

        h.write_u32(self.nodes.len() as u32);
        for node in &self.nodes {
            match node {
                IrNode::UGen { kind, consts } => {
                    h.write(&[TAG_UGEN]);
                    h.write(kind.as_bytes());
                    h.write_u32(consts.len() as u32);
                    for &(input, value) in consts {
                        h.write_u32(input);
                        if include_values {
                            h.write_f32(value);
                        }
                    }
                }
                IrNode::Const(v) => {
                    h.write(&[TAG_CONST]);
                    if include_values {
                        h.write_f32(*v);
                    }
                }
                IrNode::Param { default, .. } => {
                    // Param names are metadata (not topology); values fold in
                    // only for the full hash.
                    h.write(&[TAG_PARAM]);
                    if include_values {
                        h.write_f32(*default);
                    }
                }
            }
        }

        h.write_u32(self.edges.len() as u32);
        for e in &self.edges {
            h.write_u32(e.from as u32);
            h.write_u32(e.to as u32);
            h.write_u32(e.to_input as u32);
        }

        h.write_u32(self.output_node as u32);
        h.finish()
    }
}

// ---------------------------------------------------------------------------
// JSON encoding
// ---------------------------------------------------------------------------

fn json_escape(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str("\\u");
                for shift in [12, 8, 4, 0] {
                    let nib = ((c as u32) >> shift) & 0xf;
                    out.push(char::from_digit(nib, 16).unwrap());
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Emit a JSON element separator before all but the first item.
fn json_sep(out: &mut String, i: usize) {
    if i > 0 {
        out.push(',');
    }
}

fn json_num_f32(out: &mut String, v: f32) {
    // Rust's default float formatting is shortest round-trippable. Guard the
    // non-finite cases JSON cannot express by emitting 0 (IR consts from the
    // DSL are always finite; authored IR should avoid non-finite values).
    if v.is_finite() {
        use core::fmt::Write;
        let _ = write!(out, "{v:?}");
    } else {
        out.push('0');
    }
}

impl IrSynthDef {
    /// Encode to a hand-rolled JSON text form (dev / DSL-adjacent authoring).
    pub fn to_json(&self) -> String {
        use core::fmt::Write;
        let mut s = String::new();
        s.push('{');
        let _ = write!(s, "\"format_version\":{},", self.format_version);
        s.push_str("\"name\":");
        json_escape(&mut s, &self.name);
        s.push_str(",\"class\":");
        json_escape(&mut s, class_str(self.class));
        let _ = write!(s, ",\"output_channels\":{}", self.output_channels);

        s.push_str(",\"nodes\":[");
        for (i, node) in self.nodes.iter().enumerate() {
            json_sep(&mut s, i);
            match node {
                IrNode::UGen { kind, consts } => {
                    s.push_str("{\"UGen\":{\"kind\":");
                    json_escape(&mut s, kind);
                    s.push_str(",\"consts\":[");
                    for (j, (input, value)) in consts.iter().enumerate() {
                        json_sep(&mut s, j);
                        let _ = write!(s, "[{input},");
                        json_num_f32(&mut s, *value);
                        s.push(']');
                    }
                    s.push_str("]}}");
                }
                IrNode::Const(v) => {
                    s.push_str("{\"Const\":");
                    json_num_f32(&mut s, *v);
                    s.push('}');
                }
                IrNode::Param { name, default } => {
                    s.push_str("{\"Param\":{\"name\":");
                    json_escape(&mut s, name);
                    s.push_str(",\"default\":");
                    json_num_f32(&mut s, *default);
                    s.push_str("}}");
                }
            }
        }
        s.push(']');

        s.push_str(",\"edges\":[");
        for (i, e) in self.edges.iter().enumerate() {
            json_sep(&mut s, i);
            let _ = write!(
                s,
                "{{\"from\":{},\"to\":{},\"to_input\":{}}}",
                e.from, e.to, e.to_input
            );
        }
        s.push(']');

        s.push_str(",\"params\":[");
        for (i, p) in self.params.iter().enumerate() {
            json_sep(&mut s, i);
            s.push_str("{\"name\":");
            json_escape(&mut s, &p.name);
            let _ = write!(s, ",\"node\":{},\"input\":{},\"default\":", p.node, p.input);
            json_num_f32(&mut s, p.default);
            s.push('}');
        }
        s.push(']');

        s.push_str(",\"audio_inputs\":[");
        for (i, (name, node)) in self.audio_inputs.iter().enumerate() {
            json_sep(&mut s, i);
            s.push('[');
            json_escape(&mut s, name);
            let _ = write!(s, ",{node}]");
        }
        s.push(']');

        let _ = write!(s, ",\"output_node\":{}}}", self.output_node);
        s
    }
}

// ---------------------------------------------------------------------------
// JSON decoding (a small general parser, then schema mapping)
// ---------------------------------------------------------------------------

/// A parsed JSON value — enough of the data model for the IR schema.
#[derive(Debug, Clone, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

struct JsonParser<'a> {
    chars: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self {
        JsonParser {
            chars: s.as_bytes(),
            pos: 0,
        }
    }

    fn err(msg: &str) -> IrCodecError {
        IrCodecError::BadJson(msg.to_string())
    }

    fn skip_ws(&mut self) {
        while let Some(&c) = self.chars.get(self.pos) {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.chars.get(self.pos).copied()
    }

    fn value(&mut self) -> Result<Json, IrCodecError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') | Some(b'f') => self.boolean(),
            Some(b'n') => self.null(),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(Self::err("unexpected token")),
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), IrCodecError> {
        self.skip_ws();
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(Self::err("expected delimiter"))
        }
    }

    fn object(&mut self) -> Result<Json, IrCodecError> {
        self.expect(b'{')?;
        let mut fields = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.expect(b':')?;
            let val = self.value()?;
            fields.push((key, val));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(Self::err("expected ',' or '}'")),
            }
        }
        Ok(Json::Obj(fields))
    }

    fn array(&mut self) -> Result<Json, IrCodecError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(Self::err("expected ',' or ']'")),
            }
        }
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, IrCodecError> {
        self.skip_ws();
        if self.peek() != Some(b'"') {
            return Err(Self::err("expected string"));
        }
        self.pos += 1;
        let mut out = String::new();
        while let Some(c) = self.peek() {
            self.pos += 1;
            match c {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self.peek().ok_or_else(|| Self::err("bad escape"))?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{08}'),
                        b'f' => out.push('\u{0c}'),
                        b'u' => {
                            let hex = self
                                .chars
                                .get(self.pos..self.pos + 4)
                                .ok_or_else(|| Self::err("bad \\u escape"))?;
                            let code = core::str::from_utf8(hex)
                                .ok()
                                .and_then(|h| u32::from_str_radix(h, 16).ok())
                                .ok_or_else(|| Self::err("bad \\u hex"))?;
                            out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                            self.pos += 4;
                        }
                        _ => return Err(Self::err("unknown escape")),
                    }
                }
                _ => {
                    // Push this UTF-8 byte and any continuation bytes verbatim.
                    let start = self.pos - 1;
                    while let Some(&nc) = self.chars.get(self.pos) {
                        if nc == b'"' || nc == b'\\' {
                            break;
                        }
                        self.pos += 1;
                    }
                    let slice = &self.chars[start..self.pos];
                    out.push_str(core::str::from_utf8(slice).map_err(|_| IrCodecError::BadUtf8)?);
                }
            }
        }
        Err(Self::err("unterminated string"))
    }

    fn number(&mut self) -> Result<Json, IrCodecError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == b'-' || c == b'+' || c == b'.' || c == b'e' || c == b'E' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = core::str::from_utf8(&self.chars[start..self.pos])
            .map_err(|_| IrCodecError::BadUtf8)?;
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| Self::err("bad number"))
    }

    fn boolean(&mut self) -> Result<Json, IrCodecError> {
        if self.chars[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(Json::Bool(true))
        } else if self.chars[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(Json::Bool(false))
        } else {
            Err(Self::err("bad boolean"))
        }
    }

    fn null(&mut self) -> Result<Json, IrCodecError> {
        if self.chars[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(Json::Null)
        } else {
            Err(Self::err("bad null"))
        }
    }
}

// --- schema mapping ---

impl Json {
    fn get<'a>(&'a self, key: &str) -> Option<&'a Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_str(&self) -> Result<&str, IrCodecError> {
        match self {
            Json::Str(s) => Ok(s),
            _ => Err(IrCodecError::BadJson("expected string".to_string())),
        }
    }
    fn as_f64(&self) -> Result<f64, IrCodecError> {
        match self {
            Json::Num(n) => Ok(*n),
            _ => Err(IrCodecError::BadJson("expected number".to_string())),
        }
    }
    fn as_arr(&self) -> Result<&[Json], IrCodecError> {
        match self {
            Json::Arr(a) => Ok(a),
            _ => Err(IrCodecError::BadJson("expected array".to_string())),
        }
    }
    fn field<'a>(&'a self, key: &str) -> Result<&'a Json, IrCodecError> {
        self.get(key)
            .ok_or_else(|| IrCodecError::BadJson(alloc::format!("missing field {key:?}")))
    }
    fn usize_field(&self, key: &str) -> Result<usize, IrCodecError> {
        Ok(self.field(key)?.as_f64()? as usize)
    }
    fn f32_field(&self, key: &str) -> Result<f32, IrCodecError> {
        Ok(self.field(key)?.as_f64()? as f32)
    }
}

impl IrSynthDef {
    /// Decode from the JSON text form produced by [`to_json`](Self::to_json).
    pub fn from_json(text: &str) -> Result<IrSynthDef, IrCodecError> {
        let root = {
            let mut p = JsonParser::new(text);
            let v = p.value()?;
            p.skip_ws();
            v
        };

        let format_version = root.field("format_version")?.as_f64()? as u16;
        if format_version > super::FORMAT_VERSION {
            return Err(IrCodecError::UnsupportedVersion(format_version));
        }
        let name = root.field("name")?.as_str()?.to_string();
        let class_name = root.field("class")?.as_str()?;
        let class = class_from_str(class_name)
            .ok_or_else(|| IrCodecError::BadJson(alloc::format!("bad class {class_name:?}")))?;
        let output_channels = root.field("output_channels")?.as_f64()? as u16;

        let mut nodes = Vec::new();
        for node in root.field("nodes")?.as_arr()? {
            if let Some(u) = node.get("UGen") {
                let kind = u.field("kind")?.as_str()?.to_string();
                let mut consts = Vec::new();
                for pair in u.field("consts")?.as_arr()? {
                    let arr = pair.as_arr()?;
                    if arr.len() != 2 {
                        return Err(IrCodecError::BadJson(
                            "const must be [input, value]".to_string(),
                        ));
                    }
                    consts.push((arr[0].as_f64()? as u32, arr[1].as_f64()? as f32));
                }
                nodes.push(IrNode::UGen { kind, consts });
            } else if let Some(c) = node.get("Const") {
                nodes.push(IrNode::Const(c.as_f64()? as f32));
            } else if let Some(p) = node.get("Param") {
                nodes.push(IrNode::Param {
                    name: p.field("name")?.as_str()?.to_string(),
                    default: p.f32_field("default")?,
                });
            } else {
                return Err(IrCodecError::BadJson("unknown node variant".to_string()));
            }
        }

        let mut edges = Vec::new();
        for e in root.field("edges")?.as_arr()? {
            edges.push(IrEdge {
                from: e.usize_field("from")?,
                to: e.usize_field("to")?,
                to_input: e.usize_field("to_input")?,
            });
        }

        let mut params = Vec::new();
        for p in root.field("params")?.as_arr()? {
            params.push(IrParam {
                name: p.field("name")?.as_str()?.to_string(),
                node: p.usize_field("node")?,
                input: p.usize_field("input")?,
                default: p.f32_field("default")?,
            });
        }

        let mut audio_inputs = Vec::new();
        for ai in root.field("audio_inputs")?.as_arr()? {
            let arr = ai.as_arr()?;
            if arr.len() != 2 {
                return Err(IrCodecError::BadJson(
                    "audio_input must be [name, node]".to_string(),
                ));
            }
            audio_inputs.push((arr[0].as_str()?.to_string(), arr[1].as_f64()? as usize));
        }

        let output_node = root.usize_field("output_node")?;

        Ok(IrSynthDef {
            format_version,
            name,
            class,
            output_channels,
            nodes,
            edges,
            params,
            audio_inputs,
            output_node,
        })
    }
}
