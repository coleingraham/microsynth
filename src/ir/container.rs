//! The IR routing container: buses, routes, and effect SynthDefs as one
//! serializable, versioned unit — the wire-expressible counterpart of
//! [`RoutingGraph`](crate::routing::RoutingGraph).
//!
//! A single [`IrSynthDef`] (this module's sibling `super`) already crosses
//! the wire; a *topology* connecting several of them through buses and
//! insert effects could not, until this module. [`IrRoutingContainer`] is
//! the missing piece: it names a set of buses, a set of effect SynthDefs,
//! and the routes between them, and it loads into a live
//! [`RoutingGraph`] + effect [`SynthDef`](crate::synthdef::SynthDef) list via
//! [`to_routing_graph`](IrRoutingContainer::to_routing_graph) — from there,
//! [`Engine::build_routing`](crate::engine::Engine::build_routing) wires it
//! into a live audio graph exactly as if the topology had been constructed
//! directly through the Rust API (see `tests/ir_routing.rs` for the
//! equivalence proof).
//!
//! **Scope.** A container carries *routing infrastructure only* — buses and
//! insert effects. It does not carry instrument/voice SynthDefs: those
//! still travel as plain [`IrSynthDef`]s (unchanged) and get spawned onto a
//! named bus at the source-side entry points that already exist
//! (`Engine::spawn_voice_on_routing_bus`, and — over the wasm ABI — the
//! sibling exports this ticket adds in `web.rs`). Folding voices into the
//! container as well was considered and rejected: voices come and go far
//! more often than the mix topology they feed, and coupling their lifetime
//! to the topology's would force a full container re-send for every new
//! voice.
//!
//! # Bus identity (`busId`) — the wire spelling, stated once
//!
//! **A bus's identity is its `name: String` — the exact UTF-8 bytes in
//! [`IrBus::name`], nothing else.** This is not a design detail; it is the
//! one fact every consumer of this container (in this repo or across the
//! wire) must agree on, because nothing besides this doc comment tells them
//! so.
//!
//! Concretely:
//!
//! - A route's [`IrRoute::source_bus`] / [`IrRoute::target_bus`] are that
//!   same name string, not an index into `buses` and not a numeric id.
//! - The literal name `"main"` is reserved for the graph's always-present
//!   stereo output bus (`RoutingGraph::new()`'s default bus, exactly as the
//!   DSL's `bus`/`route` grammar already treats it — see `dsl/mod.rs`'s
//!   "Bus Declarations" docs: *"A `main` bus (stereo) always exists by
//!   default — you do not need to declare it"*). A container's own `buses`
//!   list must **not** redeclare `"main"`; routes may still name it as a
//!   source or target.
//! - Every other name in `buses` must be unique within one container
//!   (checked by [`to_routing_graph`](IrRoutingContainer::to_routing_graph),
//!   see [`IrRoutingError::DuplicateBus`]).
//!
//! **What `busId` is *not*.** The runtime [`BusId`](crate::routing::BusId)
//! — `struct BusId(pub(crate) usize)` — is a crate-private array index that
//! [`RoutingGraph::add_bus`](crate::routing::RoutingGraph::add_bus) hands
//! out fresh, in construction order, every time a `RoutingGraph` is built.
//! It is not exported as a wire type, is not stable across two builds of the
//! "same" topology (a container loaded twice, or loaded after a bus was
//! reordered upstream, can hand out different `BusId` values each time), and
//! must never be serialized or used as a cross-process/cross-repo key.
//!
//! Any producer that needs to key something off "which bus" — including a
//! deterministic seed derivation keyed on a path like
//! `(busId, stageIndex, paramName)` — must key off the `name` string
//! defined here, verbatim, not off array position or a synthesized numeric
//! id. This is the identity this crate commits to on the wire; a consumer
//! should read it from here rather than re-deriving its own spelling.
//!
//! # Format version
//!
//! Introducing this container is the semantic change behind the
//! [`super::FORMAT_VERSION`] bump from 1 to 2 (see that constant's doc
//! comment: *"bump on any semantic change to the byte layout or node
//! model"*) — the IR crate's on-wire vocabulary as a whole now includes a
//! second artifact type, even though [`IrSynthDef`]'s own byte layout is
//! completely unchanged. Back-compat is unaffected: [`IrSynthDef::from_bytes`]
//! rejects only a stored `format_version` *greater than* the reader's
//! `FORMAT_VERSION`, so pre-existing version-1 single-def files decode
//! exactly as before. A version-1 file is never a valid
//! [`IrRoutingContainer`] and vice versa — the two share no magic prefix
//! (see [`CONTAINER_MAGIC`] vs. `serialize::MAGIC`), so a decoder for one
//! format cleanly rejects bytes from the other rather than misparsing them.

use super::serialize::{Json, JsonParser, Reader, json_escape, json_sep, put_str, put_u16, put_u32};
use super::{FORMAT_VERSION, IrCodecError, IrError, IrSynthDef, SynthDefClass};
use crate::dsl::compiler::UGenRegistry;
use crate::routing::RoutingGraph;
use crate::synthdef::SynthDef;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

/// Magic prefix for the routing-container binary format. Deliberately shares
/// no byte prefix with `IrSynthDef`'s own magic (`"MICROSYNTH-IR"`, defined
/// in `serialize.rs`) so a decoder for either format rejects bytes meant for
/// the other with [`IrCodecError::BadMagic`] instead of silently misreading
/// them as a truncated/corrupt instance of itself. See this module's back-
/// compat test (`tests/ir_routing.rs`), which feeds a legacy single-def
/// stream to [`IrRoutingContainer::from_bytes`] and asserts the rejection.
const CONTAINER_MAGIC: &[u8] = b"MSIR-ROUTE";

/// The reserved bus name for the graph's always-present stereo output bus
/// (`RoutingGraph::new()`'s default bus). See the module docs' "Bus identity"
/// section.
pub const MAIN_BUS: &str = "main";

/// A named bus declaration. See the module docs' "Bus identity" section for
/// what `name` means on the wire.
#[derive(Debug, Clone, PartialEq)]
pub struct IrBus {
    /// The bus's identity — see the module docs. Must not be `"main"`
    /// (reserved) and must be unique within a container.
    pub name: String,
    /// Channel count (e.g. 2 for stereo).
    pub channels: u16,
}

/// A route: `source_bus`'s output feeds `effect_def`, whose output feeds
/// `target_bus`. All three are names — `source_bus`/`target_bus` are bus
/// identities (see the module docs), `effect_def` is the `name` of an
/// [`IrSynthDef`] in the same container's [`IrRoutingContainer::effects`].
#[derive(Debug, Clone, PartialEq)]
pub struct IrRoute {
    pub source_bus: String,
    pub effect_def: String,
    pub target_bus: String,
}

/// Buses, routes, and effect SynthDefs as one serializable unit. See the
/// module docs for the full contract, especially the "Bus identity" section.
#[derive(Debug, Clone, PartialEq)]
pub struct IrRoutingContainer {
    pub format_version: u16,
    /// Buses beyond the implicit `"main"` bus (mirrors the DSL's `bus`
    /// declarations, which likewise never redeclare `main`).
    pub buses: Vec<IrBus>,
    pub routes: Vec<IrRoute>,
    /// The effect SynthDefs `routes` may reference by name. Each must be
    /// `class == SynthDefClass::Effect` (checked by
    /// [`to_routing_graph`](Self::to_routing_graph), which is also what
    /// activates `SynthDefClass::Effect`'s structural validation — see
    /// `IrSynthDef::validate`/`check_shell` in `super`).
    pub effects: Vec<IrSynthDef>,
}

/// Errors building a [`RoutingGraph`] from an [`IrRoutingContainer`].
#[derive(Debug, Clone, PartialEq)]
pub enum IrRoutingError {
    /// Two buses (or a bus and the reserved `"main"`) share a name.
    DuplicateBus(String),
    /// Two effect defs share a name.
    DuplicateEffectDef(String),
    /// A route names a bus absent from `buses` and not `"main"`.
    UnknownBus { at: &'static str, name: String },
    /// A route's `effect_def` names no entry in `effects`.
    UnknownEffectDef(String),
    /// An entry in `effects` is not `class == SynthDefClass::Effect` (e.g. a
    /// `Source` was mistakenly included).
    NotAnEffect(String),
    /// Structural validation or compilation of one of `effects` failed.
    Effect { name: String, error: IrError },
}

impl fmt::Display for IrRoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrRoutingError::DuplicateBus(name) => write!(f, "duplicate bus name: {name:?}"),
            IrRoutingError::DuplicateEffectDef(name) => {
                write!(f, "duplicate effect def name: {name:?}")
            }
            IrRoutingError::UnknownBus { at, name } => {
                write!(f, "{at} references unknown bus {name:?}")
            }
            IrRoutingError::UnknownEffectDef(name) => {
                write!(f, "route references unknown effect def {name:?}")
            }
            IrRoutingError::NotAnEffect(name) => {
                write!(f, "effect def {name:?} is not class Effect")
            }
            IrRoutingError::Effect { name, error } => {
                write!(f, "effect def {name:?}: {error}")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Loading: container -> RoutingGraph
// ---------------------------------------------------------------------------

impl IrRoutingContainer {
    /// Build a live [`RoutingGraph`] and its effect [`SynthDef`]s from this
    /// container: buses are added in declaration order (skipping `"main"`,
    /// already present), each effect def is validated (activating
    /// `SynthDefClass::Effect`'s shell check) and compiled, and each route is
    /// resolved by bus/effect *name* — the identity this module's docs
    /// define — and wired via `RoutingGraph::add_effect`.
    ///
    /// Pass the result to
    /// [`Engine::build_routing`](crate::engine::Engine::build_routing) to
    /// instantiate it into a live audio graph.
    pub fn to_routing_graph(
        &self,
        reg: &UGenRegistry,
    ) -> Result<(RoutingGraph, Vec<SynthDef>), IrRoutingError> {
        let mut routing = RoutingGraph::new();

        for bus in &self.buses {
            if bus.name == MAIN_BUS {
                return Err(IrRoutingError::DuplicateBus(bus.name.clone()));
            }
            if routing.bus_by_name(&bus.name).is_some() {
                return Err(IrRoutingError::DuplicateBus(bus.name.clone()));
            }
            routing.add_bus(bus.name.clone(), bus.channels as usize);
        }

        let mut effect_defs: Vec<SynthDef> = Vec::with_capacity(self.effects.len());
        for eff in &self.effects {
            if effect_defs.iter().any(|d| d.name() == eff.name) {
                return Err(IrRoutingError::DuplicateEffectDef(eff.name.clone()));
            }
            if eff.class != SynthDefClass::Effect {
                return Err(IrRoutingError::NotAnEffect(eff.name.clone()));
            }
            eff.validate(reg).map_err(|error| IrRoutingError::Effect {
                name: eff.name.clone(),
                error,
            })?;
            let def = eff.compile(reg).map_err(|error| IrRoutingError::Effect {
                name: eff.name.clone(),
                error,
            })?;
            effect_defs.push(def);
        }

        for route in &self.routes {
            let source_bus = routing
                .bus_by_name(&route.source_bus)
                .ok_or_else(|| IrRoutingError::UnknownBus {
                    at: "route.source_bus",
                    name: route.source_bus.clone(),
                })?;
            let target_bus = routing
                .bus_by_name(&route.target_bus)
                .ok_or_else(|| IrRoutingError::UnknownBus {
                    at: "route.target_bus",
                    name: route.target_bus.clone(),
                })?;
            let def = effect_defs
                .iter()
                .find(|d| d.name() == route.effect_def)
                .ok_or_else(|| IrRoutingError::UnknownEffectDef(route.effect_def.clone()))?;
            routing.add_effect(source_bus, def, target_bus);
        }

        Ok((routing, effect_defs))
    }
}

// ---------------------------------------------------------------------------
// Binary encoding — same length-prefixed-record style as IrSynthDef, reusing
// its primitives (`put_*`/`Reader`) from `serialize.rs`.
// ---------------------------------------------------------------------------

impl IrRoutingContainer {
    /// Encode to the canonical binary form.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(CONTAINER_MAGIC);
        put_u16(&mut out, self.format_version);

        put_u32(&mut out, self.buses.len() as u32);
        for bus in &self.buses {
            put_str(&mut out, &bus.name);
            put_u16(&mut out, bus.channels);
        }

        put_u32(&mut out, self.routes.len() as u32);
        for route in &self.routes {
            put_str(&mut out, &route.source_bus);
            put_str(&mut out, &route.effect_def);
            put_str(&mut out, &route.target_bus);
        }

        put_u32(&mut out, self.effects.len() as u32);
        for eff in &self.effects {
            // Each effect is embedded as a complete, self-describing
            // IrSynthDef stream (own magic + version), length-prefixed. This
            // reuses IrSynthDef's own codec verbatim instead of restating
            // node/edge encoding here.
            let bytes = eff.to_bytes();
            put_u32(&mut out, bytes.len() as u32);
            out.extend_from_slice(&bytes);
        }

        out
    }

    /// Decode from the canonical binary form.
    pub fn from_bytes(bytes: &[u8]) -> Result<IrRoutingContainer, IrCodecError> {
        let mut r = Reader::new(bytes);
        if r.take(CONTAINER_MAGIC.len())? != CONTAINER_MAGIC {
            return Err(IrCodecError::BadMagic);
        }
        let format_version = r.u16()?;
        if format_version > FORMAT_VERSION {
            return Err(IrCodecError::UnsupportedVersion(format_version));
        }

        let bus_count = r.usize32()?;
        let mut buses = Vec::with_capacity(bus_count);
        for _ in 0..bus_count {
            buses.push(IrBus {
                name: r.string()?,
                channels: r.u16()?,
            });
        }

        let route_count = r.usize32()?;
        let mut routes = Vec::with_capacity(route_count);
        for _ in 0..route_count {
            routes.push(IrRoute {
                source_bus: r.string()?,
                effect_def: r.string()?,
                target_bus: r.string()?,
            });
        }

        let effect_count = r.usize32()?;
        let mut effects = Vec::with_capacity(effect_count);
        for _ in 0..effect_count {
            let len = r.usize32()?;
            let bytes = r.take(len)?;
            effects.push(IrSynthDef::from_bytes(bytes)?);
        }

        Ok(IrRoutingContainer {
            format_version,
            buses,
            routes,
            effects,
        })
    }
}

// ---------------------------------------------------------------------------
// JSON encoding (dev/authoring form, mirroring IrSynthDef::to_json/from_json)
// ---------------------------------------------------------------------------

impl IrRoutingContainer {
    /// Encode to a hand-rolled JSON text form.
    pub fn to_json(&self) -> String {
        use core::fmt::Write;
        let mut s = String::new();
        s.push('{');
        let _ = write!(s, "\"format_version\":{},", self.format_version);

        s.push_str("\"buses\":[");
        for (i, bus) in self.buses.iter().enumerate() {
            json_sep(&mut s, i);
            s.push_str("{\"name\":");
            json_escape(&mut s, &bus.name);
            let _ = write!(s, ",\"channels\":{}}}", bus.channels);
        }
        s.push(']');

        s.push_str(",\"routes\":[");
        for (i, route) in self.routes.iter().enumerate() {
            json_sep(&mut s, i);
            s.push_str("{\"source_bus\":");
            json_escape(&mut s, &route.source_bus);
            s.push_str(",\"effect_def\":");
            json_escape(&mut s, &route.effect_def);
            s.push_str(",\"target_bus\":");
            json_escape(&mut s, &route.target_bus);
            s.push('}');
        }
        s.push(']');

        s.push_str(",\"effects\":[");
        for (i, eff) in self.effects.iter().enumerate() {
            json_sep(&mut s, i);
            // Embed each effect's own JSON form verbatim (same reuse
            // principle as the binary encoder).
            s.push_str(&eff.to_json());
        }
        s.push_str("]}");
        s
    }

    /// Decode from the JSON text form produced by [`to_json`](Self::to_json).
    pub fn from_json(text: &str) -> Result<IrRoutingContainer, IrCodecError> {
        let root = {
            let mut p = JsonParser::new(text);
            let v = p.value()?;
            p.skip_ws();
            v
        };

        let format_version = root.field("format_version")?.as_f64()? as u16;
        if format_version > FORMAT_VERSION {
            return Err(IrCodecError::UnsupportedVersion(format_version));
        }

        let mut buses = Vec::new();
        for b in root.field("buses")?.as_arr()? {
            buses.push(IrBus {
                name: b.field("name")?.as_str()?.to_string(),
                channels: b.usize_field("channels")? as u16,
            });
        }

        let mut routes = Vec::new();
        for rt in root.field("routes")?.as_arr()? {
            routes.push(IrRoute {
                source_bus: rt.field("source_bus")?.as_str()?.to_string(),
                effect_def: rt.field("effect_def")?.as_str()?.to_string(),
                target_bus: rt.field("target_bus")?.as_str()?.to_string(),
            });
        }

        let mut effects = Vec::new();
        for e in root.field("effects")?.as_arr()? {
            // Each element is a full IrSynthDef object; re-serialize it to
            // text and hand it to IrSynthDef::from_json rather than
            // duplicating the node/edge schema mapping here.
            effects.push(IrSynthDef::from_json(&json_reencode(e))?);
        }

        Ok(IrRoutingContainer {
            format_version,
            buses,
            routes,
            effects,
        })
    }
}

/// Re-render a parsed [`Json`] value back to text, so a nested object parsed
/// as part of the container's document can be handed to
/// [`IrSynthDef::from_json`] unchanged. Only the value shapes `IrSynthDef`'s
/// own encoder ever produces (object/array/string/number) are exercised here
/// in practice, but null/bool are handled for completeness of the `Json`
/// enum.
fn json_reencode(v: &Json) -> String {
    let mut out = String::new();
    json_reencode_into(v, &mut out);
    out
}

fn json_reencode_into(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            use core::fmt::Write;
            let _ = write!(out, "{n}");
        }
        Json::Str(s) => json_escape(out, s),
        Json::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                json_sep(out, i);
                json_reencode_into(item, out);
            }
            out.push(']');
        }
        Json::Obj(fields) => {
            out.push('{');
            for (i, (k, val)) in fields.iter().enumerate() {
                json_sep(out, i);
                json_escape(out, k);
                out.push(':');
                json_reencode_into(val, out);
            }
            out.push('}');
        }
    }
}
