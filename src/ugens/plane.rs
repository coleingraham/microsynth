//! Linear interpolation over a triangulated plane.
//!
//! - [`Plane2d`]: a quantity given at the vertices of a triangle mesh,
//!   interpolated barycentrically at a position `(x, y)`, with ghost points
//!   that clamp positions outside the vertices' hull to its edge.

use crate::buffer::{AudioBuffer, read_input};
use crate::context::ProcessContext;
use crate::node::UGen;
use alloc::vec::Vec;

/// The most valued vertices one [`Plane2d`] carries.
pub const PLANE_MAX_VALUES: usize = 16;
/// The most points (valued vertices and ghosts) one [`Plane2d`] carries.
pub const PLANE_MAX_POINTS: usize = 32;
/// The most triangles one [`Plane2d`] carries.
pub const PLANE_MAX_TRIANGLES: usize = 64;

/// Port of `x`.
pub const PLANE_X: usize = 0;
/// Port of `y`.
pub const PLANE_Y: usize = 1;
/// Port of `n`, the number of valued vertices.
pub const PLANE_N: usize = 2;
/// Port of `blend`, the triangle blend steepness (0 = none).
pub const PLANE_BLEND: usize = 3;
/// Port of `axis` (0 = none, 1 = an exponential along `x`, 2 = along `y`).
pub const PLANE_AXIS: usize = 4;
/// Port of `lo`, the axis exponential at coordinate 0.
pub const PLANE_LO: usize = 5;
/// Port of `hi`, the axis exponential at coordinate 1.
pub const PLANE_HI: usize = 6;
/// Port of `step`, the axis exponential's widest linear segment (0 = exact).
pub const PLANE_STEP: usize = 7;

/// The port of valued vertex `i`'s value `vI`.
pub const fn plane_value_port(i: usize) -> usize {
    8 + i
}

/// The port of point `i`'s coordinate `c` (0 = `pxI`, 1 = `pyI`).
pub const fn plane_point_port(i: usize, c: usize) -> usize {
    8 + PLANE_MAX_VALUES + 2 * i + c
}

/// The port of triangle `t` (`triT`).
pub const fn plane_triangle_port(t: usize) -> usize {
    8 + PLANE_MAX_VALUES + 2 * PLANE_MAX_POINTS + t
}

/// A triangle's three point indices packed into one port value:
/// `a + 32·b + 1024·c`.
pub const fn plane_pack_triangle(a: usize, b: usize, c: usize) -> f32 {
    (a + PLANE_MAX_POINTS * b + PLANE_MAX_POINTS * PLANE_MAX_POINTS * c) as f32
}

/// The total number of ports.
pub const PLANE_PORTS: usize = plane_triangle_port(PLANE_MAX_TRIANGLES);

/// Mesh ports (everything but `x`, `y` and the values), compared every
/// block to decide whether the mesh must be rebuilt.
const N_MESH: usize = 6 + 2 * PLANE_MAX_POINTS + PLANE_MAX_TRIANGLES;

/// The axis exponential's segments are no narrower than this.
const MIN_STEP: f64 = 1.0e-3;

/// A gap of the axis knots is divided into `⌈gap / step − PLANE_KNOT_SLACK⌉` parts.
pub const PLANE_KNOT_SLACK: f64 = 1.0e-6;

/// Linear interpolation of a quantity over a triangle mesh in the plane: the
/// vowel trapezium, a timbre space, any two controls that address a
/// scattered set of measured points.
///
/// The quantity is given at `n` **valued vertices** (points `0 … n−1`, values
/// `v0 … v{n−1}`); the remaining points referenced by the triangles are
/// **ghosts**, each valued at the nearest point of the valued vertices'
/// convex hull (a linear combination of one or two hull vertices). Ghosts
/// placed just outside the region of interest make a position outside the
/// hull read (approximately) the value at the hull's edge, without the
/// extrapolation a bare triangulation would give.
///
/// At `(x, y)` the output is the barycentric interpolation of the point
/// values in the triangle containing the position: with
/// `λ_j(x, y)` the triangle's barycentric coordinates (affine in the
/// position, summing to 1),
///
/// ```text
/// out = A(x or y) · Σ_j λ_j · value(t_j)
/// ```
///
/// exact at every valued vertex, linear within a triangle, continuous
/// across shared edges.
///
/// - **Containing triangle.** With `blend` 0 (the default), the triangle
///   whose smallest `λ` is largest (the containing one; outside the mesh
///   but inside its bounding box the nearest, extrapolated). With `blend` `K > 0`, every triangle gets the
///   weight `clip(1 + K · min λ, 0, 1)` (1 inside it, 0 beyond `1/K` outside)
///   and the triangles' interpolations are averaged by those weights. On a
///   shared edge the triangles agree, so this changes the output only within
///   `1/K` of an edge, by a fraction of the change of slope there; it exists
///   so that a graph of arithmetic nodes that blends this way can be
///   replaced sample for sample.
/// - **Axis exponential `A`.** `axis` 0 (the default): `A = 1`. `axis` 1
///   (2): `A` is the exponential `lo · (hi / lo)^c` of `c` = `x` (`y`),
///   clamped to `[0, 1]`: a quantity whose logarithm is linear along that
///   axis times an interpolated ratio to it (a formant frequency on the
///   trapezium's ln F1 axis). With `step > 0` the exponential is replaced by
///   its piecewise-linear interpolation through knots at 0, 1 and every
///   valued vertex's coordinate on the axis, each gap divided evenly into
///   `⌈gap / step − 10⁻⁶⌉` parts (no segment wider than `step`, and a gap of
///   a whole number of steps not split once more by rounding): exact at the
///   knots, within `step² · ln²(hi/lo) / 8` relative between them.
///
/// # Ports
///
/// - `x`, `y` (required): the position. Read every sample; non-finite reads
///   as 0. The position is clamped to the bounding box of the mesh's points
///   (those its triangles use) before interpolation, so a control that
///   overshoots reads the value at the box's edge instead of an unbounded
///   extrapolation.
/// - `n`: the number of valued vertices, 0 … 16 (default 0: output 0).
/// - `blend`: see above (default 0).
/// - `axis`, `lo`, `hi`, `step`: see above (defaults 0, 1, 1, 0).
/// - `v0 … v15`: the values at the valued vertices.
/// - `px0 py0 … px31 py31`: the points' coordinates (valued vertices first).
/// - `tri0 … tri63`: the triangles, each its three point indices packed as
///   `a + 32·b + 1024·c` ([`plane_pack_triangle`]); a negative or non-finite
///   value (the default −1) is an unused slot. Triangles naming a point
///   outside `0 … 31` or with no area are ignored.
///
/// The mesh, the values and `blend`/`axis`/`lo`/`hi`/`step` are per-instrument
/// constants: they travel as inline consts and are read once per block (a
/// change rebuilds the mesh at the next block). The output is a pure function
/// of the position, recomputed only when `x` or `y` changes, so a position
/// held by its controls costs one comparison per sample.
///
/// # Done
///
/// The output is a value, not a signal its inputs' silence silences (the
/// interpolation at `(0, 0)` is not 0), so the node passes no silence to the
/// done rule and is never done itself.
pub struct Plane2d {
    /// The mesh ports as last read (bit patterns), to detect a change.
    mesh_key: [u32; N_MESH],
    /// The values as last read (bit patterns).
    value_key: [u32; PLANE_MAX_VALUES],
    built: bool,
    mesh: Mesh,
    /// Each point's value: its own, or its fold onto the hull.
    point_value: [f64; PLANE_MAX_POINTS],
    /// The last position (bit patterns) and its output.
    last: Option<(u32, u32, f32)>,
}

/// A triangle: its point indices and the affine coefficients `(a, b, c)` of
/// its barycentric coordinates `λ_j = a·x + b·y + c`.
type Triangle = ([usize; 3], [(f64, f64, f64); 3]);

/// The mesh in evaluation form.
#[derive(Clone, Default)]
struct Mesh {
    blend: f64,
    axis: u8,
    triangles: Vec<Triangle>,
    /// Per point: its value as a combination of valued vertices.
    fold: Vec<[(usize, f64); 2]>,
    /// The axis curve's knots `(c, A(c))`, ascending; empty = `A = 1`.
    knots: Vec<(f64, f64)>,
    /// `(ln lo, ln hi)` for the exact exponential (`step` 0).
    ln_span: Option<(f64, f64)>,
    /// The triangles' points' bounding box `(x min, x max, y min, y max)`;
    /// positions are clamped to it.
    bbox: (f64, f64, f64, f64),
}

impl Default for Plane2d {
    fn default() -> Self {
        Self::new()
    }
}

impl Plane2d {
    pub fn new() -> Self {
        Plane2d {
            mesh_key: [0; N_MESH],
            value_key: [0; PLANE_MAX_VALUES],
            built: false,
            mesh: Mesh::default(),
            point_value: [0.0; PLANE_MAX_POINTS],
            last: None,
        }
    }
}

/// `v` if finite, else `default`.
#[inline]
fn finite_or(v: f32, default: f32) -> f32 {
    if v.is_finite() { v } else { default }
}

fn cross(o: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Indices of the convex hull of `p`, counter-clockwise (Andrew's monotone
/// chain; collinear points on an edge are left out).
fn convex_hull(p: &[(f64, f64)]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    if p.len() < 2 {
        // a single point is its own hull (the chain below would drop it)
        return idx;
    }
    idx.sort_by(|&i, &j| {
        p[i].partial_cmp(&p[j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut lower: Vec<usize> = Vec::new();
    for &i in &idx {
        while lower.len() >= 2
            && cross(p[lower[lower.len() - 2]], p[lower[lower.len() - 1]], p[i]) <= 0.0
        {
            lower.pop();
        }
        lower.push(i);
    }
    let mut upper: Vec<usize> = Vec::new();
    for &i in idx.iter().rev() {
        while upper.len() >= 2
            && cross(p[upper[upper.len() - 2]], p[upper[upper.len() - 1]], p[i]) <= 0.0
        {
            upper.pop();
        }
        upper.push(i);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The nearest point of the hull (indices into `p`) to `g`, as a combination
/// of one or two hull vertices.
fn fold_onto_hull(p: &[(f64, f64)], hull: &[usize], g: (f64, f64)) -> [(usize, f64); 2] {
    if hull.len() == 1 {
        return [(hull[0], 1.0), (hull[0], 0.0)];
    }
    let mut best = (f64::MAX, [(hull[0], 1.0), (hull[0], 0.0)]);
    for e in 0..hull.len() {
        let (i, j) = (hull[e], hull[(e + 1) % hull.len()]);
        let (a, b) = (p[i], p[j]);
        let d = (b.0 - a.0, b.1 - a.1);
        let len2 = d.0 * d.0 + d.1 * d.1;
        let t = if len2 > 0.0 {
            (((g.0 - a.0) * d.0 + (g.1 - a.1) * d.1) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q = (a.0 + t * d.0, a.1 + t * d.1);
        let dist = (g.0 - q.0).hypot(g.1 - q.1);
        if dist < best.0 - 1e-12 {
            best = (dist, [(i, 1.0 - t), (j, t)]);
        }
    }
    best.1
}

/// The affine coefficients of a triangle's barycentric coordinates, or
/// `None` for a triangle with no area.
fn barycentric_affine(p: &[(f64, f64)], t: [usize; 3]) -> Option<[(f64, f64, f64); 3]> {
    let (a, b, c) = (p[t[0]], p[t[1]], p[t[2]]);
    let det = (b.1 - c.1) * (a.0 - c.0) + (c.0 - b.0) * (a.1 - c.1);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let l0 = (
        (b.1 - c.1) / det,
        (c.0 - b.0) / det,
        (-(b.1 - c.1) * c.0 - (c.0 - b.0) * c.1) / det,
    );
    let l1 = (
        (c.1 - a.1) / det,
        (a.0 - c.0) / det,
        (-(c.1 - a.1) * c.0 - (a.0 - c.0) * c.1) / det,
    );
    Some([l0, l1, (-l0.0 - l1.0, -l0.1 - l1.1, 1.0 - l0.2 - l1.2)])
}

impl Mesh {
    /// Build from the mesh ports (`port(k)` reads port `k` at block start).
    fn build(port: &dyn Fn(usize, f32) -> f32) -> Mesh {
        let n = (port(PLANE_N, 0.0).max(0.0) as usize).min(PLANE_MAX_VALUES);
        let blend = port(PLANE_BLEND, 0.0).max(0.0) as f64;
        let axis = match port(PLANE_AXIS, 0.0) {
            1.0 => 1,
            2.0 => 2,
            _ => 0,
        };
        let points: Vec<(f64, f64)> = (0..PLANE_MAX_POINTS)
            .map(|i| {
                (
                    port(plane_point_port(i, 0), 0.0) as f64,
                    port(plane_point_port(i, 1), 0.0) as f64,
                )
            })
            .collect();
        let mut triangles = Vec::new();
        if n > 0 {
            for t in 0..PLANE_MAX_TRIANGLES {
                let packed = port(plane_triangle_port(t), -1.0);
                if packed < 0.0 {
                    continue;
                }
                let k = packed as usize;
                let m = PLANE_MAX_POINTS;
                if k >= m * m * m {
                    continue;
                }
                let idx = [k % m, (k / m) % m, k / (m * m)];
                if let Some(co) = barycentric_affine(&points, idx) {
                    triangles.push((idx, co));
                }
            }
        }
        // valued vertices are themselves; ghosts fold onto the valued hull
        let hull = if n > 0 {
            convex_hull(&points[..n])
        } else {
            Vec::new()
        };
        let fold = (0..PLANE_MAX_POINTS)
            .map(|i| {
                if i < n {
                    [(i, 1.0), (i, 0.0)]
                } else if hull.is_empty() {
                    [(0, 0.0), (0, 0.0)]
                } else {
                    fold_onto_hull(&points[..n], &hull, points[i])
                }
            })
            .collect();
        let (mut knots, mut ln_span) = (Vec::new(), None);
        if axis != 0 && n > 0 {
            let lo = port(PLANE_LO, 1.0) as f64;
            let hi = port(PLANE_HI, 1.0) as f64;
            if lo > 0.0 && hi > 0.0 {
                let span = (lo.ln(), hi.ln());
                let a = |c: f64| (span.0 + c.clamp(0.0, 1.0) * (span.1 - span.0)).exp();
                let step = port(PLANE_STEP, 0.0) as f64;
                if step > 0.0 {
                    let step = step.max(MIN_STEP);
                    let mut c: Vec<f64> = points[..n]
                        .iter()
                        // the curve is flat outside [0, 1]: a coordinate
                        // beyond it adds no shape, only knots
                        .map(|p| (if axis == 1 { p.0 } else { p.1 }).clamp(0.0, 1.0))
                        .chain([0.0, 1.0])
                        .collect();
                    c.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                    c.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
                    knots.push((c[0], a(c[0])));
                    for w in c.windows(2) {
                        // a gap that is a whole number of steps (up to the
                        // f32 rounding of its ends) is not split once more
                        let m = ((w[1] - w[0]) / step - PLANE_KNOT_SLACK).ceil().max(1.0) as usize;
                        for j in 1..=m {
                            let k = w[0] + (w[1] - w[0]) * j as f64 / m as f64;
                            knots.push((k, a(k)));
                        }
                    }
                } else {
                    ln_span = Some(span);
                }
            }
        }
        let mut bbox = (
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        );
        for &(idx, _) in &triangles {
            for &(px, py) in idx.iter().map(|&i| &points[i]) {
                bbox = (
                    bbox.0.min(px),
                    bbox.1.max(px),
                    bbox.2.min(py),
                    bbox.3.max(py),
                );
            }
        }
        Mesh {
            bbox,
            blend,
            axis,
            triangles,
            fold,
            knots,
            ln_span,
        }
    }

    /// The axis factor at `c`.
    fn axis_factor(&self, c: f64) -> f64 {
        if let Some((lo, hi)) = self.ln_span {
            return (lo + c.clamp(0.0, 1.0) * (hi - lo)).exp();
        }
        let k = &self.knots;
        if k.is_empty() {
            return 1.0;
        }
        let c = c.clamp(k[0].0, k[k.len() - 1].0);
        // the segment containing c (the knots are few: a scan is enough)
        let mut s = 0;
        while s + 2 < k.len() && c > k[s + 1].0 {
            s += 1;
        }
        if k.len() == 1 {
            return k[0].1;
        }
        let (c0, a0) = k[s];
        let (c1, a1) = k[s + 1];
        a0 + (a1 - a0) / (c1 - c0) * (c - c0)
    }

    fn eval(&self, pv: &[f64; PLANE_MAX_POINTS], x: f64, y: f64) -> f64 {
        if self.triangles.is_empty() {
            return 0.0;
        }
        // no extrapolation beyond the mesh's bounding box
        let x = x.clamp(self.bbox.0, self.bbox.1);
        let y = y.clamp(self.bbox.2, self.bbox.3);
        let lambdas = |co: &[(f64, f64, f64); 3]| {
            [
                co[0].0 * x + co[0].1 * y + co[0].2,
                co[1].0 * x + co[1].1 * y + co[1].2,
                co[2].0 * x + co[2].1 * y + co[2].2,
            ]
        };
        let interp =
            |t: &[usize; 3], l: &[f64; 3]| l[0] * pv[t[0]] + l[1] * pv[t[1]] + l[2] * pv[t[2]];
        let mut v = None;
        if self.blend > 0.0 {
            let (mut acc, mut tot) = (0.0, 0.0);
            for (t, co) in &self.triangles {
                let l = lambdas(co);
                let w = (1.0 + self.blend * l[0].min(l[1]).min(l[2])).clamp(0.0, 1.0);
                if w > 0.0 {
                    tot += w;
                    acc += w * interp(t, &l);
                }
            }
            if tot > 0.0 {
                v = Some(acc / tot);
            }
        }
        let v = v.unwrap_or_else(|| {
            let mut best: Option<(&[usize; 3], [f64; 3], f64)> = None;
            for (t, co) in &self.triangles {
                let l = lambdas(co);
                let m = l[0].min(l[1]).min(l[2]);
                if best.as_ref().is_none_or(|b| m > b.2) {
                    best = Some((t, l, m));
                }
            }
            let (t, l, _) = best.expect("a triangle");
            interp(t, &l)
        });
        match self.axis {
            1 => v * self.axis_factor(x),
            2 => v * self.axis_factor(y),
            _ => v,
        }
    }
}

impl UGen for Plane2d {
    ugen_spec!(
        "Plane2d",
        category = Utility,
        inputs = ["x", "y"],
        optional_inputs = [
            "n", "blend", "axis", "lo", "hi", "step", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
            "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "px0", "py0", "px1", "py1",
            "px2", "py2", "px3", "py3", "px4", "py4", "px5", "py5", "px6", "py6", "px7", "py7",
            "px8", "py8", "px9", "py9", "px10", "py10", "px11", "py11", "px12", "py12", "px13",
            "py13", "px14", "py14", "px15", "py15", "px16", "py16", "px17", "py17", "px18", "py18",
            "px19", "py19", "px20", "py20", "px21", "py21", "px22", "py22", "px23", "py23", "px24",
            "py24", "px25", "py25", "px26", "py26", "px27", "py27", "px28", "py28", "px29", "py29",
            "px30", "py30", "px31", "py31", "tri0", "tri1", "tri2", "tri3", "tri4", "tri5", "tri6",
            "tri7", "tri8", "tri9", "tri10", "tri11", "tri12", "tri13", "tri14", "tri15", "tri16",
            "tri17", "tri18", "tri19", "tri20", "tri21", "tri22", "tri23", "tri24", "tri25",
            "tri26", "tri27", "tri28", "tri29", "tri30", "tri31", "tri32", "tri33", "tri34",
            "tri35", "tri36", "tri37", "tri38", "tri39", "tri40", "tri41", "tri42", "tri43",
            "tri44", "tri45", "tri46", "tri47", "tri48", "tri49", "tri50", "tri51", "tri52",
            "tri53", "tri54", "tri55", "tri56", "tri57", "tri58", "tri59", "tri60", "tri61",
            "tri62", "tri63",
        ],
        outputs = ["out"]
    );

    fn init(&mut self, _context: &ProcessContext) {
        self.reset();
    }

    fn reset(&mut self) {
        self.built = false;
        self.last = None;
    }

    fn process(
        &mut self,
        _context: &ProcessContext,
        inputs: &[Option<&AudioBuffer>],
        output: &mut AudioBuffer,
    ) {
        let port = |k: usize| inputs.get(k).copied().flatten();
        let x_buf = port(PLANE_X);
        let y_buf = port(PLANE_Y);

        // The constants, read at the block's first sample of channel 0.
        let at = |k: usize, d: f32| -> f32 { read_input(port(k), 0, 0, d) };
        let mesh_port = |j: usize| -> usize {
            if j < 6 {
                PLANE_N + j
            } else {
                plane_point_port(0, 0) + (j - 6)
            }
        };
        let defaults = |k: usize| -> f32 {
            match k {
                PLANE_LO | PLANE_HI => 1.0,
                k if k >= plane_triangle_port(0) => -1.0,
                _ => 0.0,
            }
        };
        let mut changed = !self.built;
        for j in 0..N_MESH {
            let k = mesh_port(j);
            let bits = at(k, defaults(k)).to_bits();
            if bits != self.mesh_key[j] {
                self.mesh_key[j] = bits;
                changed = true;
            }
        }
        if changed {
            self.mesh = Mesh::build(&|k, d| at(k, d));
            self.built = true;
        }
        let mut values_changed = changed;
        for i in 0..PLANE_MAX_VALUES {
            let bits = at(plane_value_port(i), 0.0).to_bits();
            if bits != self.value_key[i] {
                self.value_key[i] = bits;
                values_changed = true;
            }
        }
        if values_changed {
            let v: Vec<f64> = (0..PLANE_MAX_VALUES)
                .map(|i| finite_or(f32::from_bits(self.value_key[i]), 0.0) as f64)
                .collect();
            for (pv, f) in self.point_value.iter_mut().zip(&self.mesh.fold) {
                *pv = f[0].1 * v[f[0].0] + f[1].1 * v[f[1].0];
            }
            self.last = None;
        }

        for ch in 0..output.num_channels() {
            let out = output.channel_mut(ch).samples_mut();
            for (i, o) in out.iter_mut().enumerate() {
                let x = finite_or(read_input(x_buf, ch, i, 0.0), 0.0);
                let y = finite_or(read_input(y_buf, ch, i, 0.0), 0.0);
                let (xb, yb) = (x.to_bits(), y.to_bits());
                if let Some((lx, ly, v)) = self.last
                    && lx == xb
                    && ly == yb
                {
                    *o = v;
                    continue;
                }
                let v = self.mesh.eval(&self.point_value, x as f64, y as f64) as f32;
                let v = if v.is_finite() { v } else { 0.0 };
                self.last = Some((xb, yb, v));
                *o = v;
            }
        }
    }
}
