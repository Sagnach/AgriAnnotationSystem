"""B1: direct, inline class-label editor for the Gradio 6 image review canvas.

Uses only Gradio's built-in HTML component (no CDN/npm dependency).
The component value is a server-authored, JSON-compatible view model. It is NOT
trusted as an annotation record: all changes are checked against AnnotationManager.
"""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image

from src.annotation.annotation_manager import AnnotationManager


CANVAS_HTML = r'''
<div class="aa-editor">
  <div class="aa-help">Click a colored label to edit its class in place. Click the image to use the selected Add/Refine/Select tool.</div>
  <div class="aa-stage" role="group" aria-label="Interactive agricultural annotation image">
    <img class="aa-image" alt="Annotated agricultural image" draggable="false" />
    <div class="aa-box-layer" aria-hidden="true"></div>
    <div class="aa-label-layer"></div>
    <div class="aa-popover" role="dialog" aria-label="Edit instance class" hidden>
      <div class="aa-popover-heading"></div>
      <label class="aa-input-label" for="aa-inline-label-input">Class label</label>
      <input class="aa-inline-label-input" id="aa-inline-label-input" type="text"
             maxlength="80" autocomplete="off" spellcheck="false" list="aa-class-options" />
      <datalist id="aa-class-options"></datalist>
      <div class="aa-popover-buttons">
        <button type="button" class="aa-cancel">Cancel</button>
        <button type="button" class="aa-save">Apply label</button>
      </div>
      <div class="aa-error" aria-live="polite"></div>
    </div>
    <div class="aa-placeholder">Upload an image and run SAM 3 to begin editing.</div>
  </div>
  <div class="aa-footer" aria-live="polite">No active annotation session.</div>
</div>
'''

CANVAS_CSS = r'''
.aa-editor { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; width: 100%; }
.aa-help {font-size:12px; line-height:1.5; color: var(--body-text-color-subdued,#64748b); margin-bottom:8px;}
.aa-stage {position:relative; display:table; margin:0 auto; max-width:100%; background:#111827; border-radius:10px; overflow:visible; line-height:0;}
.aa-image {display:block; width:auto; max-width:100%; max-height:75vh; height:auto; user-select:none; -webkit-user-drag:none; cursor:crosshair; border-radius:10px;}
.aa-box-layer, .aa-label-layer {position:absolute; inset:0; pointer-events:none;}
.aa-box {position:absolute; pointer-events:none; box-sizing:border-box; border:1.7px solid var(--aa-border,#60a5fa); border-radius:1px;}
.aa-box.aa-selected {border:3px solid #fff; box-shadow:0 0 0 2px #0f172a, 0 0 5px #fff; z-index:6;}
.aa-chip {position:absolute; z-index:12; transform:translateY(-100%); min-width:24px; max-width:210px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; border-radius:4px 4px 0 0; padding:2px 5px; font-size:11px; font-weight:700; line-height:1.35; color:#fff; background:rgba(15,23,42,.93); border:1px solid var(--aa-border,#60a5fa); box-shadow:0 1px 3px rgba(0,0,0,.5); cursor:pointer; pointer-events:auto;}
.aa-chip.aa-low {border-style:dashed; opacity:.85;}
.aa-chip.aa-top-edge {transform:none; border-radius:0 0 4px 4px;}
.aa-chip.aa-selected {z-index:18; background:#fef08a; color:#172554; border-color:#fde047;}
.aa-chip:hover,.aa-chip:focus-visible {z-index:30; outline:2px solid #fff;}
.aa-popover {position:absolute; z-index:100; width:240px; max-width:calc(100% - 12px); background:#fff; color:#0f172a; border:1px solid #94a3b8; border-radius:9px; padding:12px; box-shadow:0 7px 22px rgba(0,0,0,.36); line-height:1.4; pointer-events:auto;}
.aa-popover[hidden], .aa-placeholder[hidden] {display:none !important;}
.aa-popover-heading {font-weight:700; font-size:13px; margin-bottom:8px;}
.aa-input-label {display:block; font-size:12px; font-weight:600; margin-bottom:4px;}
.aa-inline-label-input {width:100%; box-sizing:border-box; min-height:34px; border:1px solid #94a3b8; border-radius:5px; color:#111827; background:#fff; padding:6px 8px; font-size:14px;}
.aa-popover-buttons {display:flex; justify-content:flex-end; gap:7px; margin-top:10px;}
.aa-popover-buttons button {border:1px solid #cbd5e1; border-radius:5px; padding:5px 9px; cursor:pointer; line-height:1.5; font-size:12px;}
.aa-cancel {background:#e2e8f0; color:#0f172a;}
.aa-save {background:#2563eb; color:white; font-weight:700;}
.aa-error {font-size:11px; color:#b91c1c; margin-top:5px; min-height:0;}
.aa-placeholder {font-size:13px; color:#cbd5e1; padding:60px 16px; line-height:1.5; text-align:center;}
.aa-footer {margin-top:8px; font-size:12px; color:var(--body-text-color-subdued,#64748b); line-height:1.5;}
'''

# Deliberately delegated listeners: Gradio refreshes value without needing to
# re-attach handlers, and labels are rebuilt from server data after every change.
CANVAS_JS = r'''
const root = element.querySelector('.aa-editor');
const stage = element.querySelector('.aa-stage');
const image = element.querySelector('.aa-image');
const boxLayer = element.querySelector('.aa-box-layer');
const labelLayer = element.querySelector('.aa-label-layer');
const popover = element.querySelector('.aa-popover');
const editorInput = element.querySelector('.aa-inline-label-input');
const classOptions = element.querySelector('#aa-class-options');
const heading = element.querySelector('.aa-popover-heading');
const error = element.querySelector('.aa-error');
const footer = element.querySelector('.aa-footer');
const placeholder = element.querySelector('.aa-placeholder');
let editingId = null;

function view() { return props.value || {}; }
function findAnnotation(id) {
  return (view().annotations || []).find(a => Number(a.id) === Number(id));
}
function closePopover() { popover.hidden = true; editingId = null; error.textContent = ''; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function repaint() {
  const data = view();
  const available = Boolean(data.image_data_url && data.width > 0 && data.height > 0);
  placeholder.hidden = available;
  image.hidden = !available;
  boxLayer.replaceChildren();
  labelLayer.replaceChildren();
  closePopover();
  if (!available) {
    footer.textContent = 'No active annotation session.';
    return;
  }
  image.src = data.image_data_url;
  const W = Number(data.width), H = Number(data.height);
  const annotations = data.annotations || [];
  const selected = Number(data.selected_id || 0);
  footer.textContent = `${annotations.length} visible instances · ${data.image_name || 'Image'} · click a label to edit its class`;
  const knownClasses = [...new Set(['apple', 'orange', 'pear', 'peach', ...annotations.map(a => a.label)])].filter(Boolean);
  classOptions.replaceChildren();
  knownClasses.forEach(label => {
    const opt = document.createElement('option'); opt.value = label; classOptions.appendChild(opt);
  });
  annotations.forEach(a => {
    const x0 = clamp(Number(a.box[0]), 0, W), y0 = clamp(Number(a.box[1]), 0, H);
    const x1 = clamp(Number(a.box[2]), 0, W), y1 = clamp(Number(a.box[3]), 0, H);
    const rect = document.createElement('div');
    rect.className = 'aa-box' + (Number(a.id) === selected ? ' aa-selected' : '');
    rect.style.left = `${100 * x0 / W}%`;
    rect.style.top = `${100 * y0 / H}%`;
    rect.style.width = `${100 * Math.max(0, x1 - x0) / W}%`;
    rect.style.height = `${100 * Math.max(0, y1 - y0) / H}%`;
    rect.style.setProperty('--aa-border', a.color || '#60a5fa');
    boxLayer.appendChild(rect);
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'aa-chip' + (a.tier === 'candidate' ? ' aa-low' : '') +
                     (Number(a.id) === selected ? ' aa-selected' : '') +
                     (y0 <= 20 ? ' aa-top-edge' : '');
    chip.dataset.instanceId = String(a.id);
    chip.style.left = `${100 * x0 / W}%`;
    chip.style.top = `${100 * y0 / H}%`;
    chip.style.setProperty('--aa-border', a.color || '#60a5fa');
    chip.textContent = `${a.label} #${a.id}`;
    chip.title = `Edit class: ${a.label} #${a.id} (${a.status})`;
    chip.setAttribute('aria-label', `Edit class label for instance ${a.id}, currently ${a.label}`);
    labelLayer.appendChild(chip);
  });
}
function openInline(id, chip) {
  const a = findAnnotation(id);
  if (!a) return;
  editingId = Number(a.id);
  heading.textContent = `Edit label for instance #${a.id}`;
  editorInput.value = a.label;
  error.textContent = '';
  popover.hidden = false;
  const s = stage.getBoundingClientRect();
  const r = chip.getBoundingClientRect();
  const width = Math.min(240, s.width - 12);
  const left = clamp(r.left - s.left, 6, Math.max(6, s.width - width - 6));
  let top = (r.bottom - s.top) + 6;
  if (top + 175 > s.height && r.top - s.top > 180) top = r.top - s.top - 175;
  popover.style.left = `${Math.round(left)}px`;
  popover.style.top = `${Math.max(6, Math.round(top))}px`;
  labelLayer.querySelectorAll('.aa-chip').forEach(b => b.classList.toggle('aa-selected', b === chip));
  editorInput.focus(); editorInput.select();
  trigger('edit', {instance_id: editingId});
}
function submitInline() {
  if (editingId === null) return;
  const label = editorInput.value.trim();
  if (!label) { error.textContent = 'Class label cannot be empty.'; editorInput.focus(); return; }
  if (label.length > 80) { error.textContent = 'Maximum 80 characters.'; return; }
  const current = view();
  props.value = {
    ...current,
    action: {type: 'relabel', instance_id: editingId, new_label: label,
             event_id: `${Date.now()}-${Math.random().toString(36).slice(2)}`}
  };
  trigger('submit');
  closePopover();
}
root.addEventListener('click', e => {
  const chip = e.target.closest('.aa-chip');
  if (chip && root.contains(chip)) {
    e.preventDefault(); e.stopPropagation();
    openInline(Number(chip.dataset.instanceId), chip);
    return;
  }
  if (e.target.closest('.aa-popover')) return;
  if (e.target === image || e.target === stage || e.target === boxLayer || e.target === labelLayer) {
    if (!view().image_data_url) return;
    const r = image.getBoundingClientRect();
    if (!r.width || !r.height) return;
    const x = clamp((e.clientX - r.left) * Number(view().width) / r.width, 0, view().width - 1);
    const y = clamp((e.clientY - r.top) * Number(view().height) / r.height, 0, view().height - 1);
    closePopover();
    trigger('select', {index: [Math.round(x), Math.round(y)]});
  }
});
root.querySelector('.aa-cancel').addEventListener('click', closePopover);
root.querySelector('.aa-save').addEventListener('click', submitInline);
editorInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); submitInline(); }
  if (e.key === 'Escape') { e.preventDefault(); closePopover(); }
});
watch('value', repaint);
repaint();
'''


def empty_canvas_payload() -> dict[str, Any]:
    return {"image_data_url": None, "width": 0, "height": 0,
            "session_id": "", "revision": 0, "annotations": [],
            "selected_id": 0, "action": None}


def encode_overlay_as_data_url(image: Image.Image, jpeg_quality: int = 88) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def make_canvas_payload(
    overlay: Image.Image,
    manager: AnnotationManager,
    *,
    session_id: str,
    selected_id: int | None,
    show_candidates: bool,
) -> dict[str, Any]:
    """Serialize read-only geometry + background image for client rendering.

    The JS may submit an edit request, but the backend never trusts client-side
    masks, boxes, session IDs or class metadata as authoritative.
    """
    annotations = []
    for ann in manager.active_annotations(show_candidates=show_candidates):
        from hashlib import sha256
        digest = sha256(f"{ann.class_label}:{ann.instance_id}".encode("utf-8")).digest()
        color = "#{:02x}{:02x}{:02x}".format(
            60 + digest[0] % 176, 60 + digest[1] % 176, 60 + digest[2] % 176
        )
        annotations.append({
            "id": int(ann.instance_id),
            "label": ann.class_label,
            "box": [float(v) for v in ann.bbox_xyxy],
            "color": color,
            "score": None if ann.score is None else float(ann.score),
            "tier": ann.tier,
            "status": ann.review_status,
        })
    return {
        "image_data_url": encode_overlay_as_data_url(overlay),
        "width": int(overlay.width),
        "height": int(overlay.height),
        "image_name": manager.image_name,
        "session_id": session_id,
        "revision": len(manager.history),
        "selected_id": int(selected_id or 0),
        "annotations": annotations,
        "action": None,
    }
