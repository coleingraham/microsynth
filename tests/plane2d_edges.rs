//! `plane2d` edge cases: one valued vertex, a vertex far off the axis, a
//! position far outside the mesh.
use microsynth::node::UGen;
use microsynth::ugens::{
    PLANE_AXIS, PLANE_BLEND, PLANE_HI, PLANE_LO, PLANE_N, PLANE_PORTS, PLANE_STEP, Plane2d,
    plane_pack_triangle, plane_point_port, plane_triangle_port, plane_value_port,
};
use microsynth::*;

const SR: f32 = 44100.0;
const BLOCK: usize = 64;

fn run(x: f32, y: f32, consts: &[(usize, f32)]) -> (f32, std::time::Duration) {
    let mut node = Plane2d::new();
    let ctx = ProcessContext::new(SR, BLOCK);
    node.init(&ctx);
    let mut bufs: Vec<Option<AudioBuffer>> = (0..PLANE_PORTS).map(|_| None).collect();
    let mut set = |k: usize, v: f32| {
        let mut b = AudioBuffer::new(1, BLOCK);
        b.channel_mut(0).fill(v);
        bufs[k] = Some(b);
    };
    set(0, x);
    set(1, y);
    for &(k, v) in consts {
        set(k, v);
    }
    let refs: Vec<Option<&AudioBuffer>> = bufs.iter().map(|b| b.as_ref()).collect();
    let mut o = AudioBuffer::new(1, BLOCK);
    let t = std::time::Instant::now();
    node.process(&ctx, &refs, &mut o);
    (o.channel(0).samples()[0], t.elapsed())
}

#[test]
fn one_valued_vertex_ghosts_read_its_value() {
    // point 0 valued (3.0) at the centre; ghosts 1..3 around it
    let mut c = vec![
        (PLANE_N, 1.0),
        (plane_value_port(0), 3.0),
        (plane_point_port(0, 0), 0.5),
        (plane_point_port(0, 1), 0.5),
        (plane_point_port(1, 0), -1.0),
        (plane_point_port(1, 1), -1.0),
        (plane_point_port(2, 0), 2.0),
        (plane_point_port(2, 1), -1.0),
        (plane_point_port(3, 0), 0.5),
        (plane_point_port(3, 1), 2.0),
    ];
    for (t, tri) in [(0, 1, 2), (0, 2, 3), (0, 3, 1)].iter().enumerate() {
        c.push((
            plane_triangle_port(t),
            plane_pack_triangle(tri.0, tri.1, tri.2),
        ));
    }
    let (at_vertex, _) = run(0.5, 0.5, &c);
    let (near_ghost, _) = run(0.5, 1.9, &c);
    assert_eq!(at_vertex, 3.0);
    assert!(
        (near_ghost - 3.0).abs() < 1e-5,
        "near a ghost: {near_ghost} (want 3)"
    );
}

#[test]
fn a_far_vertex_coordinate_does_not_explode_the_axis_knots() {
    // three valued vertices, one far outside [0, 1] on the axis
    let far = 10_000.0f32;
    let mut c = vec![
        (PLANE_N, 3.0),
        (PLANE_AXIS, 1.0),
        (PLANE_LO, 1.0),
        (PLANE_HI, 2.0),
        (PLANE_STEP, 0.001),
        (plane_value_port(0), 1.0),
        (plane_value_port(1), 1.0),
        (plane_value_port(2), 1.0),
        (plane_point_port(0, 0), 0.0),
        (plane_point_port(0, 1), 0.0),
        (plane_point_port(1, 0), far),
        (plane_point_port(1, 1), 0.0),
        (plane_point_port(2, 0), 0.0),
        (plane_point_port(2, 1), 1.0),
    ];
    c.push((plane_triangle_port(0), plane_pack_triangle(0, 1, 2)));
    let (v, dt) = run(0.5, 0.1, &c);
    let want = 2f32.powf(0.5);
    assert!((v - want).abs() < 1e-3 * want, "{v} vs {want}");
    assert!(dt.as_millis() < 50, "first block took {dt:?}");
}

#[test]
fn a_position_far_outside_the_mesh_reads_the_bounding_box_edge() {
    // one triangle (0,0) (1,0) (0,1) carrying v = 2x + 1: linear, so an
    // unclamped position would extrapolate without bound
    let mut c = vec![
        (PLANE_N, 3.0),
        (plane_value_port(0), 1.0),
        (plane_value_port(1), 3.0),
        (plane_value_port(2), 1.0),
        (plane_point_port(0, 0), 0.0),
        (plane_point_port(0, 1), 0.0),
        (plane_point_port(1, 0), 1.0),
        (plane_point_port(1, 1), 0.0),
        (plane_point_port(2, 0), 0.0),
        (plane_point_port(2, 1), 1.0),
    ];
    c.push((plane_triangle_port(0), plane_pack_triangle(0, 1, 2)));
    for blend in [0.0f32, 4000.0] {
        let mut c = c.clone();
        c.push((PLANE_BLEND, blend));
        let at = |x: f32, y: f32| run(x, y, &c).0;
        assert_eq!(at(10.0, 0.0), at(1.0, 0.0), "blend {blend}");
        assert!((at(10.0, 0.0) - 3.0).abs() < 1e-6, "blend {blend}");
        assert_eq!(at(-5.0, -5.0), at(0.0, 0.0), "blend {blend}");
        assert!((at(1.0e6, 0.25) - 3.0).abs() < 1e-6, "blend {blend}");
        // inside the box nothing changes
        assert!((at(0.25, 0.25) - 1.5).abs() < 1e-6, "blend {blend}");
    }
}
