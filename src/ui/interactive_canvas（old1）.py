"""B2 interactive image canvas: B1 inline labels + draggable SAM-guided boxes.

Only display geometry travels to the browser. The browser proposes an edited
rectangle; the Python backend validates it and recomputes an instance mask.
The mask, not the draft rectangle, remains authoritative for the COCO bbox.

Requires Gradio 6 HTML custom events: select, edit and submit.
"""
from __future__ import annotations

import base64
from hashlib import sha256
from io import BytesIO
from typing import Any

from PIL import Image

from src.annotation.annotation_manager import AnnotationManager


CANVAS_HTML = r'''
<div class="aa-editor">
  <div class="aa-help">Click a colored class tag to edit the label. Select an instance, then choose <b>Edit selected box</b> to move or resize its prompt rectangle. The new box will guide SAM 3 to recompute the mask.</div>
  <div class="aa-toolbar" role="toolbar" aria-label="Bounding box tools">
    <button type="button" class="aa-start-box">Edit selected box</button>
    <button type="button" class="aa-cancel-box" hidden>Cancel</button>
    <button type="button" class="aa-apply-box" hidden>Apply box with SAM 3</button>
    <span class="aa-geometry" aria-live="polite">Select an instance to edit its box.</span>
  </div>
  <div class="aa-stage" role="group" aria-label="Interactive agricultural annotation image">
    <img class="aa-image" alt="Annotated agricultural image" draggable="false" />
    <div class="aa-box-layer" aria-hidden="true"></div>
    <div class="aa-label-layer"></div>
    <div class="aa-draft-layer" aria-label="Editable bounding box"></div>
    <div class="aa-popover" role="dialog" aria-label="Edit instance class" hidden>
      <div class="aa-popover-heading"></div>
      <label class="aa-input-label" for="aa-inline-label-input">Class label</label>
      <input class="aa-inline-label-input" id="aa-inline-label-input" type="text"
             maxlength="80" autocomplete="off" spellcheck="false" list="aa-class-options" />
      <datalist id="aa-class-options"></datalist>
      <div class="aa-popover-buttons">
        <button type="button" class="aa-cancel">Cancel</button>
        <button type="button" class="aa-edit-box-from-label">Edit box</button>
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
.aa-editor {font-family:system-ui,-apple-system,"Segoe UI",sans-serif; width:100%;}
.aa-help {font-size:12px; line-height:1.5; color:var(--body-text-color-subdued,#64748b); margin-bottom:8px;}
.aa-toolbar {display:flex; flex-wrap:wrap; align-items:center; gap:7px; margin-bottom:10px; line-height:1.4;}
.aa-toolbar button {padding:6px 10px; font-size:12px; font-weight:650; border:1px solid #64748b; border-radius:6px; background:#e2e8f0; color:#0f172a; cursor:pointer;}
.aa-toolbar .aa-apply-box {background:#2563eb; color:#fff; border-color:#1d4ed8;}
.aa-toolbar .aa-cancel-box {background:#f8fafc;}
.aa-toolbar button:disabled {opacity:.5; cursor:not-allowed;}
.aa-toolbar button[hidden] {display:none !important;}
.aa-geometry {font-size:11px; color:var(--body-text-color-subdued,#64748b); word-break:break-word;}
.aa-stage {position:relative; display:table; margin:0 auto; max-width:100%; background:#111827; border-radius:10px; overflow:visible; line-height:0;}
.aa-image {display:block; width:auto; max-width:100%; max-height:75vh; height:auto; user-select:none; -webkit-user-drag:none; cursor:crosshair; border-radius:10px;}
.aa-box-layer,.aa-label-layer,.aa-draft-layer {position:absolute; inset:0; pointer-events:none;}
.aa-box {position:absolute; pointer-events:none; box-sizing:border-box; border:1.7px solid var(--aa-border,#60a5fa); border-radius:1px;}
.aa-box.aa-selected {border:3px solid #fff; box-shadow:0 0 0 2px #0f172a,0 0 5px #fff; z-index:6;}
.aa-label-layer {z-index:13;}
.aa-chip {position:absolute; z-index:12; transform:translateY(-100%); min-width:24px; max-width:210px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; border-radius:4px 4px 0 0; padding:2px 5px; font-size:11px; font-weight:700; line-height:1.35; color:#fff; background:rgba(15,23,42,.93); border:1px solid var(--aa-border,#60a5fa); box-shadow:0 1px 3px rgba(0,0,0,.5); cursor:pointer; pointer-events:auto;}
.aa-chip.aa-low {border-style:dashed; opacity:.85;}
.aa-chip.aa-top-edge {transform:none; border-radius:0 0 4px 4px;}
.aa-chip.aa-selected {z-index:18; background:#fef08a; color:#172554; border-color:#fde047;}
.aa-chip:hover,.aa-chip:focus-visible {z-index:30; outline:2px solid #fff;}
.aa-draft-layer {z-index:20;}
.aa-draft-box {position:absolute; box-sizing:border-box; border:2px dashed #facc15; background:rgba(250,204,21,.08); box-shadow:0 0 0 1px #111827,0 0 9px rgba(250,204,21,.8); cursor:move; pointer-events:auto; touch-action:none;}
.aa-draft-tag {position:absolute; top:0; left:0; padding:2px 4px; font-size:11px; line-height:1.4; font-weight:700; color:#111827; background:#facc15; pointer-events:none; transform:translateY(-100%); white-space:nowrap;}
.aa-handle {position:absolute; box-sizing:border-box; width:12px; height:12px; background:#fff; border:2px solid #2563eb; border-radius:2px; transform:translate(-50%,-50%); pointer-events:auto; touch-action:none; z-index:22;}
.aa-handle[data-handle="nw"] {left:0; top:0; cursor:nwse-resize;}
.aa-handle[data-handle="n"]  {left:50%; top:0; cursor:ns-resize;}
.aa-handle[data-handle="ne"] {left:100%; top:0; cursor:nesw-resize;}
.aa-handle[data-handle="e"]  {left:100%; top:50%; cursor:ew-resize;}
.aa-handle[data-handle="se"] {left:100%; top:100%; cursor:nwse-resize;}
.aa-handle[data-handle="s"]  {left:50%; top:100%; cursor:ns-resize;}
.aa-handle[data-handle="sw"] {left:0; top:100%; cursor:nesw-resize;}
.aa-handle[data-handle="w"]  {left:0; top:50%; cursor:ew-resize;}
.aa-popover {position:absolute; z-index:100; width:250px; max-width:calc(100% - 12px); background:#fff; color:#0f172a; border:1px solid #94a3b8; border-radius:9px; padding:12px; box-shadow:0 7px 22px rgba(0,0,0,.36); line-height:1.4; pointer-events:auto;}
.aa-popover[hidden],.aa-placeholder[hidden] {display:none !important;}
.aa-popover-heading {font-weight:700; font-size:13px; margin-bottom:8px;}
.aa-input-label {display:block; font-size:12px; font-weight:600; margin-bottom:4px;}
.aa-inline-label-input {width:100%; box-sizing:border-box; min-height:34px; border:1px solid #94a3b8; border-radius:5px; color:#111827; background:#fff; padding:6px 8px; font-size:14px;}
.aa-popover-buttons {display:flex; justify-content:flex-end; flex-wrap:wrap; gap:6px; margin-top:10px;}
.aa-popover-buttons button {border:1px solid #cbd5e1; border-radius:5px; padding:5px 8px; cursor:pointer; line-height:1.5; font-size:12px;}
.aa-cancel {background:#e2e8f0; color:#0f172a;}
.aa-edit-box-from-label {background:#fef08a; color:#713f12;}
.aa-save {background:#2563eb; color:white; font-weight:700;}
.aa-error {font-size:11px; color:#b91c1c; margin-top:5px; min-height:0;}
.aa-placeholder {font-size:13px; color:#cbd5e1; padding:60px 16px; line-height:1.5; text-align:center;}
.aa-footer {margin-top:8px; font-size:12px; color:var(--body-text-color-subdued,#64748b); line-height:1.5;}
'''

CANVAS_JS = r'''
const root = element.querySelector('.aa-editor');
const stage = element.querySelector('.aa-stage');
const image = element.querySelector('.aa-image');
const boxLayer = element.querySelector('.aa-box-layer');
const labelLayer = element.querySelector('.aa-label-layer');
const draftLayer = element.querySelector('.aa-draft-layer');
const popover = element.querySelector('.aa-popover');
const editorInput = element.querySelector('.aa-inline-label-input');
const classOptions = element.querySelector('#aa-class-options');
const heading = element.querySelector('.aa-popover-heading');
const error = element.querySelector('.aa-error');
const footer = element.querySelector('.aa-footer');
const placeholder = element.querySelector('.aa-placeholder');
const startBoxButton = element.querySelector('.aa-start-box');
const cancelBoxButton = element.querySelector('.aa-cancel-box');
const applyBoxButton = element.querySelector('.aa-apply-box');
const geometry = element.querySelector('.aa-geometry');
let editingId = null;
let selectedPreviewId = 0;
let draft = null; // {instanceId, box:[x0,y0,x1,y1], initialBox:[...], dragging:null, div:null}
let applying = false;

function view() {return props.value || {};}
function findAnnotation(id) {return (view().annotations || []).find(a => Number(a.id) === Number(id));}
function clamp(v,lo,hi) {return Math.max(lo,Math.min(hi,v));}
function closePopover() {popover.hidden=true; editingId=null; error.textContent='';}
function dims() {return [Number(view().width),Number(view().height)];}
function boxString(box) {
  const [x0,y0,x1,y1]=box;
  return `x=${Math.round(x0)}, y=${Math.round(y0)}, w=${Math.round(x1-x0)}, h=${Math.round(y1-y0)} px`;
}
function refreshToolbar() {
  const item=findAnnotation(selectedPreviewId);
  startBoxButton.disabled = !item || Boolean(draft) || applying;
  cancelBoxButton.hidden = !draft;
  applyBoxButton.hidden = !draft;
  applyBoxButton.disabled = applying || !draft;
  if (draft) geometry.textContent = `#${draft.instanceId} · ${boxString(draft.box)} · drag inside to move, handles to resize` + (applying ? ' · Processing…' : '');
  else geometry.textContent = item ? `Selected ${item.label} #${item.id} · ${boxString(item.box)}` : 'Select an instance to edit its box.';
}
function renderDraftGeometry() {
  if (!draft || !draft.div) return;
  const [W,H]=dims();
  const b=draft.box;
  draft.div.style.left=`${100*b[0]/W}%`;
  draft.div.style.top=`${100*b[1]/H}%`;
  draft.div.style.width=`${100*(b[2]-b[0])/W}%`;
  draft.div.style.height=`${100*(b[3]-b[1])/H}%`;
  refreshToolbar();
}
function cancelDraft() {
  draft = null; applying = false; draftLayer.replaceChildren(); refreshToolbar();
}
function startDraft(id) {
  if (applying) return;
  const item=findAnnotation(id);
  if (!item || !view().image_data_url) {footer.textContent='Select an existing instance first.'; return;}
  closePopover();
  cancelDraft();
  selectedPreviewId=Number(item.id);
  const [W,H]=dims();
  const raw=item.box.map(Number);
  const b=[clamp(raw[0],0,W),clamp(raw[1],0,H),clamp(raw[2],0,W),clamp(raw[3],0,H)];
  if (b[2]-b[0]<4 || b[3]-b[1]<4) {footer.textContent='This instance has an invalid editing box.'; return;}
  const div=document.createElement('div');
  div.className='aa-draft-box';
  div.setAttribute('role','group');
  div.setAttribute('aria-label',`Drag or resize bounding box for instance ${item.id}`);
  const tag=document.createElement('span'); tag.className='aa-draft-tag'; tag.textContent=`${item.label} #${item.id} · edit box`; div.appendChild(tag);
  for (const h of ['nw','n','ne','e','se','s','sw','w']) {
    const handle=document.createElement('div');
    handle.className='aa-handle'; handle.dataset.handle=h;
    handle.setAttribute('aria-label',`Resize ${h} handle`);
    handle.setAttribute('title',`Drag ${h} to resize`);
    div.appendChild(handle);
  }
  draftLayer.replaceChildren(div);
  draft={instanceId:Number(item.id),initialBox:[...b],box:[...b],dragging:null,div};
  renderDraftGeometry();
  footer.textContent=`Editing box for ${item.label} #${item.id}. The draft does not modify the mask until Apply is pressed.`;
}
function repaint() {
  const data=view();
  const available=Boolean(data.image_data_url && data.width>0 && data.height>0);
  placeholder.hidden=available;
  image.hidden=!available;
  boxLayer.replaceChildren(); labelLayer.replaceChildren(); closePopover(); cancelDraft();
  applying=Boolean(data.action && data.action.type==='bbox_refine');
  selectedPreviewId=Number(data.selected_id || 0);
  if (!available) {footer.textContent='No active annotation session.'; refreshToolbar(); return;}
  image.src=data.image_data_url;
  const W=Number(data.width), H=Number(data.height);
  const annotations=data.annotations || [];
  footer.textContent=applying ? 'Applying box with SAM 3… please wait for the server.' :
    `${annotations.length} visible instances · ${data.image_name || 'Image'} · click label to relabel, select then edit box`;
  const knownClasses=[...new Set(['apple','orange','pear','peach',...annotations.map(a=>a.label)])].filter(Boolean);
  classOptions.replaceChildren();
  knownClasses.forEach(label=>{const opt=document.createElement('option');opt.value=label;classOptions.appendChild(opt);});
  annotations.forEach(a=>{
    const x0=clamp(Number(a.box[0]),0,W), y0=clamp(Number(a.box[1]),0,H);
    const x1=clamp(Number(a.box[2]),0,W), y1=clamp(Number(a.box[3]),0,H);
    const rect=document.createElement('div');
    rect.className='aa-box'+(Number(a.id)===selectedPreviewId?' aa-selected':'');
    rect.style.left=`${100*x0/W}%`; rect.style.top=`${100*y0/H}%`;
    rect.style.width=`${100*Math.max(0,x1-x0)/W}%`;
    rect.style.height=`${100*Math.max(0,y1-y0)/H}%`;
    rect.style.setProperty('--aa-border',a.color||'#60a5fa');boxLayer.appendChild(rect);
    const chip=document.createElement('button');chip.type='button';
    chip.className='aa-chip'+(a.tier==='candidate'?' aa-low':'')+
                   (Number(a.id)===selectedPreviewId?' aa-selected':'')+
                   (y0<=20?' aa-top-edge':'');
    chip.dataset.instanceId=String(a.id);
    chip.style.left=`${100*x0/W}%`; chip.style.top=`${100*y0/H}%`;
    chip.style.setProperty('--aa-border',a.color||'#60a5fa');
    chip.textContent=`${a.label} #${a.id}`;
    chip.title=`Edit class or box: ${a.label} #${a.id} (${a.status})`;
    chip.setAttribute('aria-label',`Edit annotation ${a.id}, currently ${a.label}`);
    labelLayer.appendChild(chip);
  });
  refreshToolbar();
}
function openInline(id,chip) {
  if (applying || draft) {footer.textContent=applying ? 'SAM 3 is processing the edited box…' : 'Apply or cancel the current box edit first.';return;}
  const a=findAnnotation(id); if (!a) return;
  editingId=Number(a.id);selectedPreviewId=editingId;
  heading.textContent=`Edit label for instance #${a.id}`;
  editorInput.value=a.label;error.textContent='';popover.hidden=false;
  const s=stage.getBoundingClientRect(),r=chip.getBoundingClientRect();
  const width=Math.min(250,s.width-12);
  const left=clamp(r.left-s.left,6,Math.max(6,s.width-width-6));
  let top=r.bottom-s.top+6;
  if (top+175>s.height&&r.top-s.top>180) top=r.top-s.top-175;
  popover.style.left=`${Math.round(left)}px`;popover.style.top=`${Math.max(6,Math.round(top))}px`;
  labelLayer.querySelectorAll('.aa-chip').forEach(b=>b.classList.toggle('aa-selected',b===chip));
  refreshToolbar();editorInput.focus();editorInput.select();
  trigger('edit',{instance_id:editingId});
}
function submitInline() {
  if (editingId===null || applying) return;
  const label=editorInput.value.trim();
  if (!label) {error.textContent='Class label cannot be empty.';editorInput.focus();return;}
  if (label.length>80) {error.textContent='Maximum 80 characters.';return;}
  props.value={...view(),action:{type:'relabel',instance_id:editingId,new_label:label,
    event_id:`label-${Date.now()}-${Math.random().toString(36).slice(2)}`}};
  trigger('submit');closePopover();
}
function submitBox() {
  if (!draft||applying) return;
  const diff=Math.max(...draft.box.map((v,i)=>Math.abs(v-draft.initialBox[i])));
  if (diff<0.5) {footer.textContent='Box is unchanged. Drag the box or a resize handle first.';return;}
  const [W,H]=dims();
  const b=draft.box.map(v=>Math.round(v*100)/100);
  if (b.some(v=>!Number.isFinite(v))||b[0]<0||b[1]<0||b[2]>W||b[3]>H||b[2]-b[0]<4||b[3]-b[1]<4) {
    footer.textContent='Edited box must stay inside the image and be at least 4 px wide/high.';return;
  }
  applying=true;refreshToolbar();
  footer.textContent='Applying edited box with SAM 3. The mask will be regenerated on the server…';
  props.value={...view(),action:{type:'bbox_refine',instance_id:draft.instanceId,
    box_xyxy:b,original_box:draft.initialBox,
    event_id:`box-${Date.now()}-${Math.random().toString(36).slice(2)}`}};
  trigger('submit');
}
function clientCoords(e) {
  const [W,H]=dims(),r=image.getBoundingClientRect();
  return {x:clamp((e.clientX-r.left)*W/r.width,0,W), y:clamp((e.clientY-r.top)*H/r.height,0,H)};
}
function clampDragBox(box,mode,dx,dy) {
  const [W,H]=dims();const min=4;
  let [x0,y0,x1,y1]=box;
  if (mode==='move') {
    dx=clamp(dx,-x0,W-x1);dy=clamp(dy,-y0,H-y1);
    return [x0+dx,y0+dy,x1+dx,y1+dy];
  }
  if (mode.includes('w'))x0=clamp(x0+dx,0,x1-min);
  if (mode.includes('e'))x1=clamp(x1+dx,x0+min,W);
  if (mode.includes('n'))y0=clamp(y0+dy,0,y1-min);
  if (mode.includes('s'))y1=clamp(y1+dy,y0+min,H);
  return [x0,y0,x1,y1];
}

root.addEventListener('click',e=>{
  const chip=e.target.closest('.aa-chip');
  if (chip&&root.contains(chip)) {e.preventDefault();e.stopPropagation();openInline(Number(chip.dataset.instanceId),chip);return;}
  if (e.target.closest('.aa-popover')||e.target.closest('.aa-toolbar')||e.target.closest('.aa-draft-box'))return;
  if (draft || applying)return;
  if (e.target===image||e.target===stage||e.target===boxLayer||e.target===labelLayer) {
    if (!view().image_data_url)return;
    const r=image.getBoundingClientRect();if (!r.width||!r.height)return;
    const p=clientCoords(e);
    closePopover();trigger('select',{index:[Math.round(clamp(p.x,0,view().width-1)),Math.round(clamp(p.y,0,view().height-1))]});
  }
});
root.addEventListener('pointerdown',e=>{
  if (!draft||applying)return;
  const h=e.target.closest('.aa-handle'),d=e.target.closest('.aa-draft-box');
  if (!h&&!d)return;
  const mode=h?h.dataset.handle:'move';
  draft.dragging={pointerId:e.pointerId,mode,clientX:e.clientX,clientY:e.clientY,box:[...draft.box]};
  root.setPointerCapture(e.pointerId);
  e.preventDefault();e.stopPropagation();
});
root.addEventListener('pointermove',e=>{
  if (!draft||!draft.dragging||draft.dragging.pointerId!==e.pointerId||applying)return;
  const d=draft.dragging;
  const [W,H]=dims(),r=image.getBoundingClientRect();
  const dx=(e.clientX-d.clientX)*W/r.width;
  const dy=(e.clientY-d.clientY)*H/r.height;
  draft.box=clampDragBox(d.box,d.mode,dx,dy);
  renderDraftGeometry();e.preventDefault();
});
function endPointer(e) {
  if (draft&&draft.dragging&&draft.dragging.pointerId===e.pointerId) {
    draft.dragging=null;
    if (root.hasPointerCapture(e.pointerId))root.releasePointerCapture(e.pointerId);
  }
}
root.addEventListener('pointerup',endPointer);
root.addEventListener('pointercancel',endPointer);
root.querySelector('.aa-cancel').addEventListener('click',closePopover);
root.querySelector('.aa-save').addEventListener('click',submitInline);
root.querySelector('.aa-edit-box-from-label').addEventListener('click',()=>{
  const id=editingId;if (id!==null)startDraft(id);
});
startBoxButton.addEventListener('click',()=>startDraft(selectedPreviewId));
cancelBoxButton.addEventListener('click',()=>{cancelDraft();footer.textContent='Box edit canceled; annotation unchanged.';});
applyBoxButton.addEventListener('click',submitBox);
editorInput.addEventListener('keydown',e=>{
  if (e.key==='Enter'){e.preventDefault();submitInline();}
  if (e.key==='Escape'){e.preventDefault();closePopover();}
});
root.addEventListener('keydown',e=>{
  if (e.target===editorInput)return;
  if (e.key==='Escape'&&draft){cancelDraft();footer.textContent='Box edit canceled.';e.preventDefault();return;}
  if (!draft||applying||!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;
  const amount=e.shiftKey?10:1;
  const dx=e.key==='ArrowLeft'?-amount:e.key==='ArrowRight'?amount:0;
  const dy=e.key==='ArrowUp'?-amount:e.key==='ArrowDown'?amount:0;
  draft.box=clampDragBox(draft.box,'move',dx,dy);renderDraftGeometry();e.preventDefault();
});
watch('value',repaint);
repaint();
'''


def empty_canvas_payload() -> dict[str, Any]:
    return {
        'image_data_url': None, 'width': 0, 'height': 0, 'session_id': '',
        'revision': 0, 'annotations': [], 'selected_id': 0, 'action': None,
    }


def encode_overlay_as_data_url(image: Image.Image, jpeg_quality: int = 88) -> str:
    buffer = BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=jpeg_quality, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def make_canvas_payload(
    overlay: Image.Image,
    manager: AnnotationManager,
    *,
    session_id: str,
    selected_id: int | None,
    show_candidates: bool,
) -> dict[str, Any]:
    annotations = []
    for ann in manager.active_annotations(show_candidates=show_candidates):
        digest = sha256(f'{ann.class_label}:{ann.instance_id}'.encode('utf-8')).digest()
        color = '#{:02x}{:02x}{:02x}'.format(
            60 + digest[0] % 176, 60 + digest[1] % 176, 60 + digest[2] % 176,
        )
        annotations.append({
            'id': int(ann.instance_id),
            'label': ann.class_label,
            'box': [float(v) for v in ann.bbox_xyxy],
            'color': color,
            'score': None if ann.score is None else float(ann.score),
            'tier': ann.tier,
            'status': ann.review_status,
        })
    return {
        'image_data_url': encode_overlay_as_data_url(overlay),
        'width': int(overlay.width), 'height': int(overlay.height),
        'image_name': manager.image_name, 'session_id': session_id,
        'revision': len(manager.history),
        'selected_id': int(selected_id or 0),
        'annotations': annotations, 'action': None,
    }
