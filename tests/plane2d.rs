//! `plane2d`: linear interpolation of a quantity over a triangle mesh.
//!
//! The node must be a partition of unity, exact at every valued vertex,
//! linear inside a triangle, clamp outside the valued hull through its
//! ghost points, equal an f64 reference evaluation on a grid, and reproduce,
//! sample for sample, the graph of arithmetic nodes it replaces (one
//! weight per triangle, `clip(1 + K · min λ, 0, 1)`, normalised).

use microsynth::ir::{IrEdge, IrNode, IrParam, IrSynthDef, SynthDefClass};
use microsynth::node::UGen;
use microsynth::ugens::{
    PLANE_AXIS, PLANE_BLEND, PLANE_HI, PLANE_LO, PLANE_MAX_POINTS, PLANE_MAX_VALUES, PLANE_N,
    PLANE_PORTS, PLANE_STEP, Plane2d, plane_pack_triangle, plane_point_port, plane_triangle_port,
    plane_value_port,
};
use microsynth::*;

mod common;
use common::builtin_registry;

const SR: f32 = 44100.0;
const BLOCK: usize = 64;

// ------------------------------------------------------------------ the reference mesh

type P = (f64, f64);

/// A vowel trapezium: eleven vowels (a baritone's placement, rounded to three
/// decimals) in the unit square.
const VOWELS: [P; 11] = [
    (1.000, 0.388), // ah
    (0.205, 0.000), // oo
    (0.000, 1.000), // ee
    (0.791, 0.878), // eh
    (0.553, 0.112), // oh
    (0.344, 0.886), // ih
    (0.500, 0.909), // ey
    (0.929, 0.830), // ae
    (0.727, 0.087), // ao
    (0.448, 0.241), // uu
    (0.836, 0.404), // uh
];

fn cross(o: P, a: P, b: P) -> f64 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

fn convex_hull(p: &[P]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&i, &j| p[i].partial_cmp(&p[j]).unwrap());
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

/// Bowyer–Watson, counter-clockwise triangles.
fn delaunay(p: &[P]) -> Vec<[usize; 3]> {
    let n = p.len();
    let mut pts = p.to_vec();
    pts.extend([(-100.0, -100.0), (100.0, -100.0), (0.0, 100.0)]);
    let in_circle = |t: &[usize; 3], q: P| {
        let (a, b, c) = (pts[t[0]], pts[t[1]], pts[t[2]]);
        let (ax, ay) = (a.0 - q.0, a.1 - q.1);
        let (bx, by) = (b.0 - q.0, b.1 - q.1);
        let (cx, cy) = (c.0 - q.0, c.1 - q.1);
        (ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay)
            + (cx * cx + cy * cy) * (ax * by - bx * ay)
            > 0.0
    };
    let mut tris: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];
    for (i, &q) in p.iter().enumerate() {
        let (bad, good): (Vec<[usize; 3]>, Vec<[usize; 3]>) =
            tris.into_iter().partition(|t| in_circle(t, q));
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for t in &bad {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                if let Some(k) = edges.iter().position(|&(x, y)| x == b && y == a) {
                    edges.remove(k);
                } else {
                    edges.push((a, b));
                }
            }
        }
        tris = good;
        for (a, b) in edges {
            tris.push([a, b, i]);
        }
    }
    tris.retain(|t| t.iter().all(|&v| v < n));
    for t in tris.iter_mut() {
        if cross(p[t[0]], p[t[1]], p[t[2]]) < 0.0 {
            t.swap(1, 2);
        }
    }
    tris
}

/// The mesh in f64: points (valued first), triangles, and each point's value
/// as a combination of valued vertices.
struct Mesh {
    n: usize,
    points: Vec<P>,
    triangles: Vec<[usize; 3]>,
    fold: Vec<Vec<(usize, f64)>>,
}

/// Valued vertices plus sixteen ghosts just outside the unit square (corners
/// pushed out along the diagonal, three bulged points per side), each valued
/// at the nearest point of the valued hull.
fn mesh_with_ghosts(valued: &[P]) -> Mesh {
    let mut points: Vec<P> = valued.to_vec();
    let n = points.len();
    let hull = convex_hull(&points);
    let corners: [P; 4] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    let mut ghosts = Vec::new();
    for s in 0..4 {
        let (a, b) = (corners[s], corners[(s + 1) % 4]);
        ghosts.push((
            a.0 + 0.03 * (a.0 - 0.5).signum(),
            a.1 + 0.03 * (a.1 - 0.5).signum(),
        ));
        let nrm = (b.1 - a.1, a.0 - b.0);
        for t in [0.25, 0.5, 0.75] {
            let bulge = 0.03 + 0.03 * (std::f64::consts::PI * t).sin();
            ghosts.push((
                a.0 + t * (b.0 - a.0) + nrm.0 * bulge,
                a.1 + t * (b.1 - a.1) + nrm.1 * bulge,
            ));
        }
    }
    let mut fold: Vec<Vec<(usize, f64)>> = (0..n).map(|i| vec![(i, 1.0)]).collect();
    for &g in &ghosts {
        let mut best = (f64::MAX, Vec::new());
        for e in 0..hull.len() {
            let (i, j) = (hull[e], hull[(e + 1) % hull.len()]);
            let (a, b) = (points[i], points[j]);
            let d = (b.0 - a.0, b.1 - a.1);
            let t =
                (((g.0 - a.0) * d.0 + (g.1 - a.1) * d.1) / (d.0 * d.0 + d.1 * d.1)).clamp(0.0, 1.0);
            let q = (a.0 + t * d.0, a.1 + t * d.1);
            let dist = (g.0 - q.0).hypot(g.1 - q.1);
            if dist < best.0 - 1e-12 {
                let mut c = Vec::new();
                if t < 1.0 {
                    c.push((i, 1.0 - t));
                }
                if t > 0.0 {
                    c.push((j, t));
                }
                best = (dist, c);
            }
        }
        points.push(g);
        fold.push(best.1);
    }
    let triangles = delaunay(&points);
    Mesh {
        n,
        points,
        triangles,
        fold,
    }
}

fn barycentric(p: &[P], t: [usize; 3], x: f64, y: f64) -> [f64; 3] {
    let (a, b, c) = (p[t[0]], p[t[1]], p[t[2]]);
    let det = (b.1 - c.1) * (a.0 - c.0) + (c.0 - b.0) * (a.1 - c.1);
    let l0 = ((b.1 - c.1) * (x - c.0) + (c.0 - b.0) * (y - c.1)) / det;
    let l1 = ((c.1 - a.1) * (x - c.0) + (a.0 - c.0) * (y - c.1)) / det;
    [l0, l1, 1.0 - l0 - l1]
}

impl Mesh {
    fn point_values(&self, v: &[f64]) -> Vec<f64> {
        self.fold
            .iter()
            .map(|c| c.iter().map(|&(i, w)| w * v[i]).sum())
            .collect()
    }

    /// The reference: the containing triangle (largest smallest λ).
    fn eval(&self, v: &[f64], x: f64, y: f64) -> f64 {
        let pv = self.point_values(v);
        let mut best: Option<([usize; 3], [f64; 3], f64)> = None;
        for t in &self.triangles {
            let l = barycentric(&self.points, *t, x, y);
            let m = l[0].min(l[1]).min(l[2]);
            if best.as_ref().is_none_or(|b| m > b.2) {
                best = Some((*t, l, m));
            }
        }
        let (t, l, _) = best.unwrap();
        (0..3).map(|j| l[j] * pv[t[j]]).sum()
    }

    /// The reference of the blended form.
    fn eval_blend(&self, v: &[f64], x: f64, y: f64, k: f64) -> f64 {
        let pv = self.point_values(v);
        let (mut acc, mut tot) = (0.0, 0.0);
        for t in &self.triangles {
            let l = barycentric(&self.points, *t, x, y);
            let w = (1.0 + k * l[0].min(l[1]).min(l[2])).clamp(0.0, 1.0);
            tot += w;
            acc += w * (0..3).map(|j| l[j] * pv[t[j]]).sum::<f64>();
        }
        acc / tot
    }

    /// The node's constant ports for this mesh and `values`.
    fn ports(&self, values: &[f64], blend: f64) -> Vec<(usize, f32)> {
        assert!(self.points.len() <= PLANE_MAX_POINTS && self.n <= PLANE_MAX_VALUES);
        let mut p = vec![(PLANE_N, self.n as f32), (PLANE_BLEND, blend as f32)];
        for (i, &v) in values.iter().enumerate() {
            p.push((plane_value_port(i), v as f32));
        }
        for (i, &(x, y)) in self.points.iter().enumerate() {
            p.push((plane_point_port(i, 0), x as f32));
            p.push((plane_point_port(i, 1), y as f32));
        }
        for (k, t) in self.triangles.iter().enumerate() {
            p.push((
                plane_triangle_port(k),
                plane_pack_triangle(t[0], t[1], t[2]),
            ));
        }
        p
    }
}

/// Smooth, unrelated values per vertex (a formant-like spread).
fn values(seed: u64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let s = ((seed * 7919 + i as u64 * 104_729) % 10_007) as f64 / 10_007.0;
            300.0 + 2700.0 * s
        })
        .collect()
}

// ------------------------------------------------------------------ harness

/// Run one `Plane2d` over whole `x`/`y` signals with constant ports `consts`.
fn run_node(x: &[f32], y: &[f32], consts: &[(usize, f32)]) -> Vec<f32> {
    let mut node = Plane2d::new();
    let ctx = ProcessContext::new(SR, BLOCK);
    node.init(&ctx);
    let n = x.len();
    let mut out = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let len = BLOCK.min(n - start);
        let mut bufs: Vec<Option<AudioBuffer>> = (0..PLANE_PORTS).map(|_| None).collect();
        let sig = |v: &[f32]| {
            let mut b = AudioBuffer::new(1, len);
            b.channel_mut(0)
                .samples_mut()
                .copy_from_slice(&v[start..start + len]);
            b
        };
        bufs[0] = Some(sig(x));
        bufs[1] = Some(sig(y));
        for &(k, v) in consts {
            let mut b = AudioBuffer::new(1, len);
            b.channel_mut(0).fill(v);
            bufs[k] = Some(b);
        }
        let refs: Vec<Option<&AudioBuffer>> = bufs.iter().map(|b| b.as_ref()).collect();
        let mut o = AudioBuffer::new(1, len);
        node.process(&ctx, &refs, &mut o);
        out.extend_from_slice(o.channel(0).samples());
        start += len;
    }
    out
}

/// Evaluate the node at a list of positions (one per sample).
fn eval_at(pos: &[(f64, f64)], consts: &[(usize, f32)]) -> Vec<f64> {
    let x: Vec<f32> = pos.iter().map(|p| p.0 as f32).collect();
    let y: Vec<f32> = pos.iter().map(|p| p.1 as f32).collect();
    run_node(&x, &y, consts)
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

fn grid(n: usize, lo: f64, hi: f64) -> Vec<(f64, f64)> {
    let mut g = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let s = |k: usize| lo + (hi - lo) * k as f64 / (n - 1) as f64;
            // f32-exact positions, so the reference sees what the node sees
            g.push((s(i) as f32 as f64, s(j) as f32 as f64));
        }
    }
    g
}

fn rel(a: f64, b: f64, scale: f64) -> f64 {
    (a - b).abs() / scale
}

// ------------------------------------------------------------------ tests

#[test]
fn the_mesh_fits_the_ports() {
    let m = mesh_with_ghosts(&VOWELS);
    assert_eq!(m.points.len(), 27);
    assert!(m.triangles.len() <= 64, "{}", m.triangles.len());
    // the ghosts cover the unit square: every grid point has a containing triangle
    for (x, y) in grid(21, 0.0, 1.0) {
        let inside = m.triangles.iter().any(|t| {
            let l = barycentric(&m.points, *t, x, y);
            l.iter().all(|&v| v >= -1e-12)
        });
        assert!(inside, "({x}, {y}) uncovered");
    }
}

#[test]
fn exact_at_every_valued_vertex() {
    let m = mesh_with_ghosts(&VOWELS);
    for seed in 0..4 {
        let v = values(seed, m.n);
        let out = eval_at(
            &VOWELS.map(|p| (p.0 as f32 as f64, p.1 as f32 as f64)),
            &m.ports(&v, 0.0),
        );
        for (i, o) in out.iter().enumerate() {
            let want = v[i] as f32 as f64;
            assert!(rel(*o, want, want) < 2e-7, "vertex {i}: {o} vs {want}");
        }
    }
}

#[test]
fn partition_of_unity_and_non_negative_weights() {
    let m = mesh_with_ghosts(&VOWELS);
    let pos = grid(41, -0.02, 1.02);
    // a constant is reproduced everywhere, inside the hull or not
    let c = vec![437.0; m.n];
    for o in eval_at(&pos, &m.ports(&c, 0.0)) {
        assert!((o - 437.0).abs() < 437.0 * 2e-7, "{o}");
    }
    // the weights of the eleven vertices (indicator values) are ≥ 0 and sum to 1
    let mut sum = vec![0.0; pos.len()];
    for i in 0..m.n {
        let mut e = vec![0.0; m.n];
        e[i] = 1.0;
        for (s, w) in sum.iter_mut().zip(eval_at(&pos, &m.ports(&e, 0.0))) {
            assert!(w >= -1e-6, "negative weight {w}");
            *s += w;
        }
    }
    for s in sum {
        assert!((s - 1.0).abs() < 1e-6, "weights sum to {s}");
    }
}

#[test]
fn linear_inside_a_triangle() {
    // values of a linear function at the vertices are reproduced exactly in
    // every triangle of valued vertices (linear precision); triangles with a
    // ghost corner (some cross the valued hull's edge) carry the clamp instead
    let m = mesh_with_ghosts(&VOWELS);
    let f = |p: P| 800.0 + 900.0 * p.0 - 300.0 * p.1;
    let v: Vec<f64> = VOWELS.iter().map(|&p| f(p)).collect();
    let consts = m.ports(&v, 0.0);
    let in_valued_triangle = |p: P| {
        m.triangles.iter().any(|t| {
            t.iter().all(|&i| i < m.n)
                && barycentric(&m.points, *t, p.0, p.1)
                    .iter()
                    .all(|&l| l >= 1e-9)
        })
    };
    let pos: Vec<P> = grid(41, 0.0, 1.0)
        .into_iter()
        .filter(|&p| in_valued_triangle(p))
        .collect();
    assert!(pos.len() > 500, "{}", pos.len());
    for (p, o) in pos.iter().zip(eval_at(&pos, &consts)) {
        assert!(rel(o, f(*p), 1000.0) < 1e-6, "{p:?}: {o} vs {}", f(*p));
    }
    // a segment inside one triangle of valued vertices: second differences vanish
    let t = m
        .triangles
        .iter()
        .find(|t| t.iter().all(|&i| i < m.n))
        .expect("a vowel triangle");
    let (a, b, c) = (m.points[t[0]], m.points[t[1]], m.points[t[2]]);
    let centre = ((a.0 + b.0 + c.0) / 3.0, (a.1 + b.1 + c.1) / 3.0);
    let line: Vec<P> = (0..=20)
        .map(|k| {
            let s = k as f64 / 20.0;
            let q = (0.1 * a.0 + 0.9 * centre.0, 0.1 * a.1 + 0.9 * centre.1);
            let r = (0.1 * b.0 + 0.9 * centre.0, 0.1 * b.1 + 0.9 * centre.1);
            (
                (q.0 + s * (r.0 - q.0)) as f32 as f64,
                (q.1 + s * (r.1 - q.1)) as f32 as f64,
            )
        })
        .collect();
    let w = values(3, m.n);
    let o = eval_at(&line, &m.ports(&w, 0.0));
    let r: Vec<f64> = line.iter().map(|p| m.eval(&w, p.0, p.1)).collect();
    for k in 0..o.len() {
        assert!(rel(o[k], r[k], 3000.0) < 1e-6);
    }
    for k in 1..o.len() - 1 {
        let d2 = o[k - 1] - 2.0 * o[k] + o[k + 1];
        assert!(d2.abs() < 3000.0 * 1e-6, "second difference {d2}");
    }
}

#[test]
fn clamps_outside_the_hull() {
    let m = mesh_with_ghosts(&VOWELS);
    let v = values(5, m.n);
    let consts = m.ports(&v, 0.0);
    // at each ghost the value is the nearest hull point's
    let ghosts: Vec<P> = m.points[m.n..]
        .iter()
        .map(|p| (p.0 as f32 as f64, p.1 as f32 as f64))
        .collect();
    let pv = m.point_values(&v);
    for (k, o) in eval_at(&ghosts, &consts).iter().enumerate() {
        let want = pv[m.n + k];
        assert!(rel(*o, want, 3000.0) < 1e-6, "ghost {k}: {o} vs {want}");
    }
    // everywhere in and around the square the output stays within the values' range
    let (lo, hi) = v
        .iter()
        .fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    for o in eval_at(&grid(61, -0.03, 1.03), &consts) {
        assert!(o >= lo - 1e-3 && o <= hi + 1e-3, "{o} outside [{lo}, {hi}]");
    }
}

#[test]
fn equals_the_reference_on_a_grid() {
    let m = mesh_with_ghosts(&VOWELS);
    let pos = grid(21, 0.0, 1.0);
    for seed in 0..5 {
        let v = values(seed, m.n);
        let scale = v.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let out = eval_at(&pos, &m.ports(&v, 0.0));
        for (p, o) in pos.iter().zip(&out) {
            let r = m.eval(&v, p.0, p.1);
            assert!(rel(*o, r, scale) < 5e-7, "{p:?}: {o} vs {r}");
        }
        let out = eval_at(&pos, &m.ports(&v, 4000.0));
        for (p, o) in pos.iter().zip(&out) {
            let r = m.eval_blend(&v, p.0, p.1, 4000.0);
            assert!(rel(*o, r, scale) < 5e-7, "blend {p:?}: {o} vs {r}");
        }
    }
}

#[test]
fn the_blend_changes_the_output_only_near_edges() {
    // on a coarse grid (no point within 1/K of an edge but the vertices and
    // shared edges, where the triangles agree) blend 4000 equals blend 0
    let m = mesh_with_ghosts(&VOWELS);
    let v = values(9, m.n);
    let pos = grid(21, 0.0, 1.0);
    let a = eval_at(&pos, &m.ports(&v, 0.0));
    let b = eval_at(&pos, &m.ports(&v, 4000.0));
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| rel(*x, *y, 3000.0))
        .fold(0.0, f64::max);
    assert!(worst < 1e-4, "{worst}");
}

#[test]
fn axis_exponential_exact_and_piecewise_linear() {
    let m = mesh_with_ghosts(&VOWELS);
    let ratio = values(2, m.n)
        .iter()
        .map(|v| v / 1500.0)
        .collect::<Vec<_>>();
    let (lo, hi) = (278.0f64, 676.0f64);
    let exp_at = |c: f64| lo * (hi / lo).powf(c.clamp(0.0, 1.0));
    let pos = grid(21, 0.0, 1.0);
    for axis in [1usize, 2] {
        let mut consts = m.ports(&ratio, 0.0);
        consts.extend([
            (PLANE_AXIS, axis as f32),
            (PLANE_LO, lo as f32),
            (PLANE_HI, hi as f32),
        ]);
        // step 0: the exponential itself
        let out = eval_at(&pos, &consts);
        for (p, o) in pos.iter().zip(&out) {
            let c = if axis == 1 { p.0 } else { p.1 };
            let r = exp_at(c) * m.eval(&ratio, p.0, p.1);
            assert!(rel(*o, r, r) < 1e-6, "axis {axis} {p:?}: {o} vs {r}");
        }
        // step 0.05: the chord through knots at 0, 1, every vertex coordinate
        // and even subdivisions; exact at the knots, under the curve between
        consts.push((PLANE_STEP, 0.05));
        let mut knots: Vec<f64> = VOWELS
            .iter()
            .map(|p| (if axis == 1 { p.0 } else { p.1 }) as f32 as f64)
            .chain([0.0, 1.0])
            .collect();
        knots.sort_by(|a, b| a.partial_cmp(b).unwrap());
        knots.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        let mut fine = vec![knots[0]];
        for w in knots.windows(2) {
            let n = ((w[1] - w[0]) / 0.05 - 1e-6).ceil().max(1.0) as usize;
            for j in 1..=n {
                fine.push(w[0] + (w[1] - w[0]) * j as f64 / n as f64);
            }
        }
        let lo_f = lo as f32 as f64;
        let hi_f = hi as f32 as f64;
        let curve = |c: f64| {
            let e = |k: f64| lo_f * (hi_f / lo_f).powf(k);
            let s = fine
                .windows(2)
                .position(|w| c <= w[1])
                .unwrap_or(fine.len() - 2);
            let (c0, c1) = (fine[s], fine[s + 1]);
            e(c0) + (e(c1) - e(c0)) / (c1 - c0) * (c - c0)
        };
        let out = eval_at(&pos, &consts);
        for (p, o) in pos.iter().zip(&out) {
            let c = if axis == 1 { p.0 } else { p.1 };
            let r = curve(c) * m.eval(&ratio, p.0, p.1);
            assert!(rel(*o, r, r) < 1e-6, "pwl axis {axis} {p:?}: {o} vs {r}");
            let exact = exp_at(c) * m.eval(&ratio, p.0, p.1);
            assert!(
                *o >= exact * (1.0 - 1e-6) && *o <= exact * 1.0004,
                "chord error {o} vs {exact}"
            );
        }
    }
}

#[test]
fn an_audio_rate_path_and_a_held_position() {
    let m = mesh_with_ghosts(&VOWELS);
    let v = values(7, m.n);
    let consts = m.ports(&v, 0.0);
    // a glide ah -> ee over 3000 samples, then held: every sample equals the reference
    let n = 6000;
    let x: Vec<f32> = (0..n)
        .map(|i| (1.0 - (i.min(3000) as f64 / 3000.0)) as f32)
        .collect();
    let y: Vec<f32> = (0..n)
        .map(|i| (0.388 + 0.612 * (i.min(3000) as f64 / 3000.0)) as f32)
        .collect();
    let out = run_node(&x, &y, &consts);
    for i in (0..n).step_by(7) {
        let r = m.eval(&v, x[i] as f64, y[i] as f64);
        assert!(
            rel(out[i] as f64, r, 3000.0) < 5e-7,
            "sample {i}: {} vs {r}",
            out[i]
        );
    }
    assert!(
        out[3000..].iter().all(|&o| o == out[3000]),
        "a held position holds its value"
    );
}

#[test]
fn a_change_of_values_or_mesh_takes_effect_at_the_next_block() {
    let m = mesh_with_ghosts(&VOWELS);
    let (v1, v2) = (values(1, m.n), values(2, m.n));
    let mut node = Plane2d::new();
    let ctx = ProcessContext::new(SR, BLOCK);
    node.init(&ctx);
    let run = |node: &mut Plane2d, consts: &[(usize, f32)]| {
        let mut bufs: Vec<Option<AudioBuffer>> = (0..PLANE_PORTS).map(|_| None).collect();
        let mut fill = |k: usize, v: f32| {
            let mut b = AudioBuffer::new(1, BLOCK);
            b.channel_mut(0).fill(v);
            bufs[k] = Some(b);
        };
        fill(0, 0.6);
        fill(1, 0.5);
        for &(k, v) in consts {
            fill(k, v);
        }
        let refs: Vec<Option<&AudioBuffer>> = bufs.iter().map(|b| b.as_ref()).collect();
        let mut o = AudioBuffer::new(1, BLOCK);
        node.process(&ctx, &refs, &mut o);
        o.channel(0).samples()[0] as f64
    };
    let a = run(&mut node, &m.ports(&v1, 0.0));
    let b = run(&mut node, &m.ports(&v2, 0.0));
    assert!(rel(a, m.eval(&v1, 0.6, 0.5), 3000.0) < 5e-7);
    assert!(rel(b, m.eval(&v2, 0.6, 0.5), 3000.0) < 5e-7);
    // a mesh of the same points with one triangle fewer
    let mut c = m.ports(&v2, 0.0);
    c.retain(|&(k, _)| k != plane_triangle_port(0));
    let d = run(&mut node, &c);
    assert!(d.is_finite());
}

#[test]
fn degenerate_and_non_finite_inputs() {
    let m = mesh_with_ghosts(&VOWELS);
    let v = values(4, m.n);
    // n = 0: output 0
    let mut c = m.ports(&v, 0.0);
    c.retain(|&(k, _)| k != PLANE_N);
    assert!(eval_at(&[(0.5, 0.5)], &c).iter().all(|&o| o == 0.0));
    // a triangle naming a point out of range, and one with no area, are ignored
    let mut c = m.ports(&v, 0.0);
    c.push((
        plane_triangle_port(m.triangles.len()),
        plane_pack_triangle(0, 1, 31) + 40_000.0,
    ));
    c.push((
        plane_triangle_port(m.triangles.len() + 1),
        plane_pack_triangle(0, 0, 1),
    ));
    let pos = grid(11, 0.0, 1.0);
    let base = eval_at(&pos, &m.ports(&v, 0.0));
    assert_eq!(eval_at(&pos, &c), base);
    // non-finite x / y read as 0; far outside, the output stays finite
    let x = [f32::NAN, f32::INFINITY, 1.0e30, -1.0e30];
    let y = [0.0, 0.0, 1.0e30, 0.0];
    let out = run_node(&x, &y, &m.ports(&v, 4000.0));
    assert!(out.iter().all(|o| o.is_finite()), "{out:?}");
    assert_eq!(
        out[0] as f64,
        eval_at(&[(0.0, 0.0)], &m.ports(&v, 4000.0))[0]
    );
}

// ------------------------------------------------------------------ against the graph it replaces

/// A tiny IR builder.
struct B {
    nodes: Vec<IrNode>,
    edges: Vec<IrEdge>,
}

impl B {
    fn ugen(&mut self, kind: &str, consts: Vec<(u32, f32)>) -> usize {
        self.nodes.push(IrNode::UGen {
            kind: kind.into(),
            consts,
        });
        self.nodes.len() - 1
    }
    fn edge(&mut self, from: usize, to: usize, to_input: usize) {
        self.edges.push(IrEdge { from, to, to_input });
    }
    fn scaled(&mut self, a: usize, k: f64) -> usize {
        let m = self.ugen("Mul", vec![(1, k as f32)]);
        self.edge(a, m, 0);
        m
    }
    fn offset(&mut self, a: usize, k: f64) -> usize {
        let m = self.ugen("Add", vec![(1, k as f32)]);
        self.edge(a, m, 0);
        m
    }
    fn bin(&mut self, kind: &str, a: usize, b: usize) -> usize {
        let m = self.ugen(kind, Vec::new());
        self.edge(a, m, 0);
        self.edge(b, m, 1);
        m
    }
    fn clip(&mut self, a: usize, lo: f64, hi: f64) -> usize {
        let m = self.ugen("clip", vec![(1, lo as f32), (2, hi as f32)]);
        self.edge(a, m, 0);
        m
    }
    fn sum(&mut self, xs: &[usize]) -> usize {
        let mut acc = xs[0];
        for &x in &xs[1..] {
            acc = self.bin("Add", acc, x);
        }
        acc
    }
    fn min(&mut self, a: usize, b: usize) -> usize {
        let d = self.bin("Sub", a, b);
        let c = self.clip(d, 0.0, 1.0e9);
        self.bin("Sub", a, c)
    }
}

/// Render a def with `x`/`y` params for `secs`, at `(x, y)`.
fn render_ir(ir: &IrSynthDef, x: f32, y: f32, blocks: usize) -> Vec<f32> {
    let registry = builtin_registry();
    ir.validate(&registry).expect("validate");
    let def = ir.compile(&registry).expect("compile");
    let mut engine = Engine::new(EngineConfig {
        sample_rate: SR,
        block_size: BLOCK,
    });
    let synth = engine.instantiate_synthdef(&def);
    engine.graph_mut().set_sink(synth.output_node());
    engine.set_param(&synth, "x", x);
    engine.set_param(&synth, "y", y);
    engine.prepare();
    let mut out = Vec::new();
    for _ in 0..blocks {
        out.extend_from_slice(engine.render().unwrap().channel(0).samples());
    }
    out
}

fn def(b: B, out: usize) -> IrSynthDef {
    IrSynthDef {
        format_version: 3,
        name: "plane".into(),
        class: SynthDefClass::Source,
        output_channels: 1,
        nodes: b.nodes,
        edges: b.edges,
        params: vec![
            IrParam {
                name: "x".into(),
                node: 0,
                input: 0,
                default: 0.5,
            },
            IrParam {
                name: "y".into(),
                node: 1,
                input: 0,
                default: 0.5,
            },
        ],
        audio_inputs: Vec::new(),
        table_bindings: Vec::new(),
        output_node: out,
    }
}

fn params() -> B {
    let nodes = vec![
        IrNode::Param {
            name: "x".into(),
            default: 0.5,
        },
        IrNode::Param {
            name: "y".into(),
            default: 0.5,
        },
    ];
    B {
        nodes,
        edges: Vec::new(),
    }
}

/// The arithmetic graph: per triangle λ affine in the position, weight
/// `clip(1 + K · min λ, 0, 1)`, normalised; folded into one weight per valued
/// vertex; the value `v_0 + Σ (v_i − v_0) · w_i`.
fn graph_def(m: &Mesh, v: &[f64], k: f64) -> IrSynthDef {
    let mut b = params();
    let (x, y) = (0, 1);
    let mut tri_w = Vec::new();
    let mut tri_l = Vec::new();
    for t in &m.triangles {
        let (pa, pb, pc) = (m.points[t[0]], m.points[t[1]], m.points[t[2]]);
        let det = (pb.1 - pc.1) * (pa.0 - pc.0) + (pc.0 - pb.0) * (pa.1 - pc.1);
        let co = [
            (
                (pb.1 - pc.1) / det,
                (pc.0 - pb.0) / det,
                (-(pb.1 - pc.1) * pc.0 - (pc.0 - pb.0) * pc.1) / det,
            ),
            (
                (pc.1 - pa.1) / det,
                (pa.0 - pc.0) / det,
                (-(pc.1 - pa.1) * pc.0 - (pa.0 - pc.0) * pc.1) / det,
            ),
        ];
        let mut l = Vec::new();
        for c in co {
            let ax = b.scaled(x, c.0);
            let by = b.scaled(y, c.1);
            let s = b.bin("Add", ax, by);
            l.push(b.offset(s, c.2));
        }
        let s01 = b.bin("Add", l[0], l[1]);
        let n = b.scaled(s01, -1.0);
        let l2 = b.offset(n, 1.0);
        let mn = b.min(l[0], l[1]);
        let mn = b.min(mn, l2);
        let kk = b.scaled(mn, k);
        let kk = b.offset(kk, 1.0);
        tri_w.push(b.clip(kk, 0.0, 1.0));
        tri_l.push([l[0], l[1], l2]);
    }
    let total = b.sum(&tri_w);
    let inv = b.ugen("Div", vec![(0, 1.0)]);
    b.edge(total, inv, 1);
    let mut contrib: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m.n];
    for (ti, t) in m.triangles.iter().enumerate() {
        let wn = b.bin("Mul", tri_w[ti], inv);
        for j in 0..3 {
            let c = b.bin("Mul", wn, tri_l[ti][j]);
            for &(i, f) in &m.fold[t[j]] {
                contrib[i].push((c, f));
            }
        }
    }
    let mut terms = Vec::new();
    for (i, cs) in contrib.iter().enumerate().skip(1) {
        let ws: Vec<usize> = cs
            .iter()
            .map(|&(c, f)| if f == 1.0 { c } else { b.scaled(c, f) })
            .collect();
        let w = b.sum(&ws);
        terms.push(b.scaled(w, v[i] - v[0]));
    }
    let s = b.sum(&terms);
    let out = b.offset(s, v[0]);
    def(b, out)
}

/// The same quantity through one `plane2d`, its mesh on shared `Const` nodes.
fn node_def(m: &Mesh, v: &[f64], k: f64) -> IrSynthDef {
    let mut b = params();
    let mut shared = Vec::new();
    for (port, value) in m.ports(v, k) {
        if (plane_value_port(0)..plane_value_port(PLANE_MAX_VALUES)).contains(&port) {
            continue;
        }
        b.nodes.push(IrNode::Const(value));
        shared.push((b.nodes.len() - 1, port));
    }
    let consts = v
        .iter()
        .enumerate()
        .map(|(i, &x)| (plane_value_port(i) as u32, x as f32))
        .collect();
    let p = b.ugen("plane2d", consts);
    b.edge(0, p, 0);
    b.edge(1, p, 1);
    for (c, port) in shared {
        b.edge(c, p, port);
    }
    def(b, p)
}

#[test]
fn matches_the_arithmetic_graph_it_replaces() {
    let m = mesh_with_ghosts(&VOWELS);
    let mut worst = 0.0f64;
    for seed in 0..3 {
        let v = values(seed, m.n);
        let (g, n) = (graph_def(&m, &v, 4000.0), node_def(&m, &v, 4000.0));
        assert!(
            g.nodes.len() > 1000 && n.nodes.len() < 120,
            "{} / {}",
            g.nodes.len(),
            n.nodes.len()
        );
        for (x, y) in grid(7, 0.0, 1.0)
            .into_iter()
            .chain([(1.0, 0.388), (0.0, 1.0), (0.205, 0.0)])
        {
            let a = render_ir(&g, x as f32, y as f32, 2);
            let o = render_ir(&n, x as f32, y as f32, 2);
            let (a, o) = (a[BLOCK + 3] as f64, o[BLOCK + 3] as f64);
            worst = worst.max(rel(a, o, a.abs()));
        }
    }
    // the graph computes in f32 (a few ulp per node); the node in f64
    assert!(worst < 2e-6, "graph vs node: {worst}");
}

#[test]
fn ir_form_with_inline_consts_matches_the_node() {
    let m = mesh_with_ghosts(&VOWELS);
    let v = values(6, m.n);
    let mut b = params();
    let consts = m
        .ports(&v, 0.0)
        .into_iter()
        .map(|(k, c)| (k as u32, c))
        .collect();
    let p = b.ugen("plane2d", consts);
    b.edge(0, p, 0);
    b.edge(1, p, 1);
    let ir = def(b, p);
    let json = ir.to_json();
    let ir = IrSynthDef::from_json(&json).expect("round trip");
    let y = render_ir(&ir, 0.3, 0.7, 3);
    let want = eval_at(&[(0.3f32 as f64, 0.7f32 as f64)], &m.ports(&v, 0.0))[0];
    assert!(y.iter().all(|&s| s as f64 == want), "{} vs {want}", y[0]);
}
