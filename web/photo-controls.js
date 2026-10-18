// Local camera hints and a reversible crop editor. No photo leaves this server.
export function mountPhotoControls(container, photo, onChange=()=>{}, onError=()=>{}) {
  if (!document.getElementById('photo-controls-styles')) {
    const style=document.createElement('style');style.id='photo-controls-styles';
    style.textContent='.photo-controls{margin-top:10px;border-top:1px solid #35444c;padding-top:8px}.photo-controls summary{cursor:pointer;font-size:12px}.photo-controls-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}.photo-controls label{display:flex;flex-direction:column;gap:4px;font-size:11px}.photo-controls input,.photo-controls select{width:100%;min-width:0;padding:6px;box-sizing:border-box}.photo-controls canvas{width:100%;height:auto;display:block;margin:10px 0;touch-action:none;cursor:crosshair;background:#15242b}.photo-controls p{font-size:11px;line-height:1.5}.photo-controls button{margin:8px 6px 0 0;padding:7px 10px}.photo-controls-status{color:#c8ded2}';
    document.head.append(style);
  }
  const details=document.createElement('details');details.className='photo-controls';
  const summary=document.createElement('summary');summary.textContent='Photo direction, tree style & crop';details.append(summary);
  const grid=document.createElement('div');grid.className='photo-controls-grid';details.append(grid);
  function field(title, options) {
    const label=document.createElement('label');label.textContent=title;
    const input=document.createElement(options ? 'select' : 'input');
    if(options) for(const [value,text] of options){const option=document.createElement('option');option.value=value;option.textContent=text;input.append(option);}
    else input.type='number';
    label.append(input);grid.append(label);return input;
  }
  const projection=field('Projection',[['auto','Auto (metadata)'],['perspective','Normal photo / screenshot'],['panorama','360° panorama']]);
  projection.value=photo.projection_setting || photo.projection || 'auto';
  const heading=field('True heading (°; blank = unknown)');heading.min='0';heading.max='359.99';heading.step='0.1';heading.value=photo.heading??'';
  const fov=field('Horizontal field of view (°)');fov.required=true;fov.min='10';fov.max='360';fov.value=photo.fov||70;
  const distance=field('Estimated object distance (m)');distance.required=true;distance.min='5';distance.max='150';distance.value=photo.estimated_distance_m||25;
  const species=field('Nearby tree style',['auto','oak','maple','birch','spruce','pine','dark_oak','bare','cherry','shrub'].map(s=>[s,s==='auto'?'Auto / map species':s==='bare'?'Bare winter':s.replace('_',' ')]));species.value=photo.tree_style||'auto';
  function sync() {
    if (![heading,fov,distance].every(input=>input.checkValidity())) return false;
    photo.heading=heading.value===''?null:Number(heading.value);
    photo.projection=projection.value;photo.projection_setting=projection.value;
    photo.fov=projection.value==='panorama'?360:Number(fov.value);
    photo.estimated_distance_m=Number(distance.value);photo.tree_style=species.value;
    fov.disabled=projection.value==='panorama';onChange(photo);return true;
  }
  for(const input of [projection,heading,fov,distance,species]) input.addEventListener('change',sync);
  fov.disabled=projection.value==='panorama';
  const note=document.createElement('p');
  note.textContent='Estimates guide nearby materials and vegetation. They do not recover exact 3D geometry. Drag a rectangle below to exclude screenshot toolbars, then check the heading and field of view.';
  if(photo.exif_heading_reference==='magnetic') note.textContent+=' EXIF heading is magnetic; enter a true-north heading to position objects.';
  details.append(note);
  const canvas=document.createElement('canvas');canvas.width=640;canvas.height=360;canvas.setAttribute('aria-label','Photo crop preview. Drag a rectangle or use percentage fields below.');details.append(canvas);
  const image=new Image();let ready=false, start=null;
  let rect=photo.crop?[...photo.crop]:photo.applied_crop&&photo.original_width?[photo.applied_crop[0]/photo.original_width,photo.applied_crop[1]/photo.original_height,photo.applied_crop[2]/photo.original_width,photo.applied_crop[3]/photo.original_height]:[0,0,1,1];
  const cropGrid=document.createElement('div');cropGrid.className='photo-controls-grid';details.append(cropGrid);
  const cropFields=['Left %','Top %','Right %','Bottom %'].map((title,i)=>{
    const label=document.createElement('label');label.textContent=title;
    const input=document.createElement('input');input.type='number';input.min='0';input.max='100';input.step='0.1';input.value=(rect[i]*100).toFixed(1);
    input.addEventListener('input',()=>{rect[i]=Math.min(1,Math.max(0,Number(input.value)/100));draw();});label.append(input);cropGrid.append(label);return input;
  });
  function draw() {
    if(!ready)return;
    const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
    ctx.clearRect(0,0,w,h);ctx.drawImage(image,0,0,w,h);
    const [left,top,right,bottom]=rect;ctx.fillStyle='rgba(0,0,0,.48)';
    ctx.fillRect(0,0,w,top*h);ctx.fillRect(0,bottom*h,w,(1-bottom)*h);ctx.fillRect(0,top*h,left*w,(bottom-top)*h);ctx.fillRect(right*w,top*h,(1-right)*w,(bottom-top)*h);
    ctx.strokeStyle='#99e59a';ctx.lineWidth=3;ctx.strokeRect(left*w,top*h,(right-left)*w,(bottom-top)*h);
  }
  function point(event){const box=canvas.getBoundingClientRect();return [Math.min(1,Math.max(0,(event.clientX-box.left)/box.width)),Math.min(1,Math.max(0,(event.clientY-box.top)/box.height))];}
  function setRect(value){rect=value;cropFields.forEach((field,i)=>field.value=(rect[i]*100).toFixed(1));draw();}
  canvas.addEventListener('pointerdown',event=>{if(!ready)return;start=point(event);canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener('pointermove',event=>{if(!start)return;const p=point(event);setRect([Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])]);});
  canvas.addEventListener('pointerup',()=>{start=null;});canvas.addEventListener('pointercancel',()=>{start=null;});
  image.onload=()=>{const scale=Math.min(640/image.width,480/image.height);canvas.width=Math.max(1,Math.round(image.width*scale));canvas.height=Math.max(1,Math.round(image.height*scale));canvas.style.maxWidth=`${canvas.width}px`;ready=true;draw();};
  image.onerror=()=>onError('The original photo could not be loaded.',true);image.src=photo.original_url||`/api/photos/${photo.id}/original`;
  const reset=document.createElement('button');reset.type='button';reset.textContent='Select full image';reset.onclick=()=>setRect([0,0,1,1]);
  const apply=document.createElement('button');apply.type='button';apply.textContent='Apply crop & analyze';
  const status=document.createElement('p');status.className='photo-controls-status';status.textContent=photo.summary||'';status.setAttribute('aria-live','polite');
  apply.onclick=async()=>{
    if(!sync()||rect[2]<=rect[0]||rect[3]<=rect[1]){onError('Enter valid photo settings and a crop with positive width and height.',true);return;}
    apply.disabled=true;status.textContent='Analyzing cropped photo locally…';
    try{
      const response=await fetch(`/api/photos/${photo.id}/analyze`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({crop:rect,projection:projection.value,heading:photo.heading,fov:photo.fov,tree_style:photo.tree_style})});
      const result=await response.json();if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:'Photo settings could not be applied.');
      Object.assign(photo,result);photo.projection_setting=projection.value;photo.projection=projection.value;
      const thumb=container.querySelector('img');if(thumb)thumb.src=result.url;
      status.textContent=result.summary;onChange(photo);
    }catch(error){status.textContent='Crop could not be applied.';onError(error.message,true);}finally{apply.disabled=false;}
  };
  details.append(reset,apply,status);container.append(details);
}
