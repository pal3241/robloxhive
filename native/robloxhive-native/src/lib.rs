#[no_mangle]
pub extern "C" fn rh_predict_axis(
    position: f64,
    velocity: f64,
    lead_seconds: f64,
    max_lead: f64,
) -> f64 {
    let lead = (velocity * lead_seconds).clamp(-max_lead, max_lead);
    position + lead
}

#[no_mangle]
pub extern "C" fn rh_iou(
    ax: f64,
    ay: f64,
    aw: f64,
    ah: f64,
    bx: f64,
    by: f64,
    bw: f64,
    bh: f64,
) -> f64 {
    let ax2 = ax + aw;
    let ay2 = ay + ah;
    let bx2 = bx + bw;
    let by2 = by + bh;
    let x1 = ax.max(bx);
    let y1 = ay.max(by);
    let x2 = ax2.min(bx2);
    let y2 = ay2.min(by2);
    let iw = (x2 - x1).max(0.0);
    let ih = (y2 - y1).max(0.0);
    let inter = iw * ih;
    let union = aw.max(0.0) * ah.max(0.0) + bw.max(0.0) * bh.max(0.0) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}
