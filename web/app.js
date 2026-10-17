import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = id => document.getElementById(id);
const state = {source:'live', size:250, photos:[], busy:false, scene:null, job:null};
let map, pin, rectangle, toastTimer;
function toast(message, error=false) {
  $('toast').textContent=message; $('toast').classList.toggle('error',error); $('toast').hidden=false;
  clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,error?12000:5000);
}
async function api(url, options={}) {
  const response=await fetch(url,options);
  if(!response.ok) {
    let message=`Request failed (${response.status})`;
    try { const body=await response.json(); message=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail); } catch {}
    throw new Error(message);
  }
  return response.json();
}
const post=(url,body)=>api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function center() {return [Number($('latitude').value),Number($('longitude').value)];}
function bbox() {
  if($('use-bbox').checked) return ['west','south','east','north'].map(side=>Number($(`bbox-${side}`).value));
  const [lat,lon]=center(), dy=state.size/111320/2, dx=dy/Math.cos(lat*Math.PI/180);
  return [lon-dx,lat-dy,lon+dx,lat+dy];
}
function updateLocation(fit=false) {
  const [lat,lon]=center(); if(!Number.isFinite(lat)||!Number.isFinite(lon)||Math.abs(lat)>85||Math.abs(lon)>180)return;
  const b=bbox(), scale=Number($('resolution').value);
  if(!b.every(Number.isFinite))return;
  if(!$('use-bbox').checked) ['west','south','east','north'].forEach((side,i)=>$(`bbox-${side}`).value=b[i].toFixed(7));
  $('bounds-fields').classList.toggle('enabled',$('use-bbox').checked);
  for(const side of ['west','south','east','north'])$(`bbox-${side}`).disabled=!$('use-bbox').checked;
  if(map){pin.setLatLng([lat,lon]);rectangle.setBounds([[b[1],b[0]],[b[3],b[2]]]);if(fit)map.fitBounds(rectangle.getBounds(),{padding:[12,12],maxZoom:17});}
  const width=(b[2]-b[0])*111320*Math.cos((b[1]+b[3])/2*Math.PI/180),depth=(b[3]-b[1])*111320;
  $('size-summary').textContent=`${Math.ceil(width/scale)} × ${Math.ceil(depth/scale)} blocks`;
  $('area-summary').textContent=`${(width*depth/1e6).toFixed(2)} km²`;
  $('google-map-link').href=`https://www.google.com/maps/search/?api=1&query=${lat},${lon}`;
  $('streetview-link').href=`https://www.google.com/maps/@?api=1&map_action=pano&viewpoint=${lat},${lon}`;
}
function setSource(source) {
  state.source=source;
  $('demo-button').innerHTML=source==='demo'?'Use live map':'Try a demo';
  $('generate-label').textContent=source==='demo'?'Generate demo world':'Generate Java world';
  $('provenance-note').textContent=source==='demo'?'Offline demo selected. The neighbourhood is synthetic; coordinates are only an anchor. Photo-guided styling still works.':'Live generation uses OpenStreetMap at your selected coordinates. Photos guide materials; hidden geometry and architectural details are approximated.';
}
function initializeMap(){
  if(!window.L){$('map-offline').hidden=false;return;}
  map=L.map('map',{zoomControl:true,scrollWheelZoom:false}).setView(center(),16);
  const tiles=L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}).addTo(map);
  tiles.on('tileerror',()=>$('map-offline').hidden=false);tiles.on('tileload',()=>$('map-offline').hidden=true);
  pin=L.marker(center(),{icon:L.divIcon({className:'map-centre',iconSize:[13,13],iconAnchor:[6,6]})}).addTo(map);
  rectangle=L.rectangle([[0,0],[0,0]],{color:'#5e7e32',weight:2,fillColor:'#b5de70',fillOpacity:.18}).addTo(map);
  map.on('click',e=>{if(state.busy)return;$('latitude').value=e.latlng.lat.toFixed(6);$('longitude').value=e.latlng.lng.toFixed(6);$('use-bbox').checked=false;setSource('live');updateLocation();});
  updateLocation(true);
}

// The preview consumes the same semantic model as the voxel generator.
let renderer, worldScene, camera, controls, model, grid;
const colors={bricks:0xa35b48,stone_bricks:0x92958b,oak_planks:0xb39462,white_concrete:0xdce0d3,sandstone:0xd7c392,
  deepslate_tiles:0x3f4a52,gray_concrete:0x424c50,smooth_stone:0xa9afa2,grass_block:0x789650,oak_log:0x786043,
  oak_leaves:0x557e38,water:0x488ea1,light_blue_stained_glass:0x8cc1d1,quartz_block:0xe0ded0};
const materialCache=new Map();
function material(name){
  if(!materialCache.has(name)){const key=(name||'stone_bricks').split(':').pop().split('[')[0];materialCache.set(name,new THREE.MeshStandardMaterial({color:colors[key]||0x939a88,roughness:.9,metalness:key.includes('glass')?.15:0,side:THREE.DoubleSide}));}
  return materialCache.get(name);
}
function cube(x,y,z,w,h,d,mat){
  const mesh=new THREE.Mesh(new THREE.BoxGeometry(Math.max(.05,w),Math.max(.05,h),Math.max(.05,d)),material(mat));
  mesh.position.set(x+w/2,y+h/2,z+d/2);mesh.castShadow=true;mesh.receiveShadow=true;model.add(mesh);return mesh;
}
function shapeFor(object){
  const points=object.geometry.coordinates;
  const shape=new THREE.Shape(points.map(p=>new THREE.Vector2(p[0],-p[1])));
  for(const hole of object.geometry.holes||[])shape.holes.push(new THREE.Path(hole.map(p=>new THREE.Vector2(p[0],-p[1]))));
  return shape;
}
function polygon(object,y,height,mat){
  const mesh=new THREE.Mesh(new THREE.ExtrudeGeometry(shapeFor(object),{depth:Math.max(.05,height),bevelEnabled:false}),material(mat));
  mesh.rotation.x=-Math.PI/2;mesh.position.y=y;mesh.castShadow=true;mesh.receiveShadow=true;model.add(mesh);return mesh;
}
function groundAt(scene,x,z){
  const heights=scene.terrain.heights, rows=heights.length, cols=heights[0].length;
  const gx=Math.max(0,Math.min(cols-1,x/Math.max(1,scene.bounds.width-1)*(cols-1)));
  const gz=Math.max(0,Math.min(rows-1,z/Math.max(1,scene.bounds.depth-1)*(rows-1)));
  const a=Math.floor(gx),b=Math.floor(gz),tx=gx-a,tz=gz-b;
  return (heights[b][a]*(1-tx)+heights[b][Math.min(a+1,cols-1)]*tx)*(1-tz)+(heights[Math.min(b+1,rows-1)][a]*(1-tx)+heights[Math.min(b+1,rows-1)][Math.min(a+1,cols-1)]*tx)*tz;
}
function segment(a,b,width,y,mat){
  const dx=b[0]-a[0],dz=b[1]-a[1],length=Math.hypot(dx,dz);
  const mesh=cube(-length/2,0,-width/2,length,.22,width,mat);
  mesh.position.set((a[0]+b[0])/2,y+.11,(a[1]+b[1])/2);mesh.rotation.y=-Math.atan2(dz,dx);return mesh;
}
function resetCamera(top=false){
  if(!state.scene||!camera)return;
  const {width,depth}=state.scene.bounds,size=Math.max(width,depth);
  const distance=Math.hypot(width,depth)/2/Math.sin(camera.fov*Math.PI/360)*1.08/Math.min(1,camera.aspect);
  const direction=new THREE.Vector3(top?0:.82,top?1:.95,top?.001:.95).normalize().multiplyScalar(distance);
  camera.position.set(width/2+direction.x,7+direction.y,depth/2+direction.z);
  controls.target.set(width/2,7,depth/2);camera.near=.1;camera.far=size*15;camera.updateProjectionMatrix();controls.update();
}
function setupPreview(){
  try{
    renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
    renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;renderer.outputColorSpace=THREE.SRGBColorSpace;
    $('viewport').prepend(renderer.domElement);worldScene=new THREE.Scene();
    camera=new THREE.PerspectiveCamera(38,1,.1,10000);controls=new OrbitControls(camera,renderer.domElement);
    controls.enableDamping=true;controls.maxPolarAngle=Math.PI*.48;controls.minDistance=10;
    worldScene.add(new THREE.HemisphereLight(0xe5f0ff,0x64704c,2.7));
    const sunlight=new THREE.DirectionalLight(0xfff1d6,3);sunlight.position.set(-150,400,200);sunlight.castShadow=true;
    sunlight.shadow.mapSize.set(2048,2048);sunlight.shadow.camera.left=-600;sunlight.shadow.camera.right=600;sunlight.shadow.camera.top=600;sunlight.shadow.camera.bottom=-600;sunlight.shadow.camera.far=1500;sunlight.shadow.bias=-.001;worldScene.add(sunlight);
    new ResizeObserver(()=>{const {width,height}=$('viewport').getBoundingClientRect();renderer.setSize(width,height);camera.aspect=width/height;camera.updateProjectionMatrix();}).observe($('viewport'));
    renderer.setAnimationLoop(()=>{controls.update();renderer.render(worldScene,camera);});
  }catch(error){$('webgl-error').hidden=false;console.warn('WebGL preview unavailable',error);}
}
function renderScene(scene){
  state.scene=scene;$('preview-loading').hidden=true;
  $('scene-name').textContent=scene.name||'Generated neighbourhood';$('scene-source').textContent=scene.source==='demo'?'SYNTHETIC DEMO':'LIVE MAP DATA';
  $('scene-dimensions').textContent=`${scene.bounds.width} × ${scene.bounds.depth} blocks`;
  for(const [id,key] of [['stat-buildings','building'],['stat-roads','road'],['stat-trees','tree']])$(id).textContent=(scene.stats[key]||0).toLocaleString();
  $('stat-coverage').textContent=`${Math.round(Math.max(scene.bounds.width,scene.bounds.depth)*scene.meters_per_block)} m`;
  $('stats-caption').textContent=scene.stats.photo_dressed_buildings?`${scene.stats.photo_dressed_buildings} buildings dressed from photos.`:'Map-shaped. Hand-built details.';
  if(!renderer)return;
  if(model){model.traverse(o=>{if(o.geometry)o.geometry.dispose();});worldScene.remove(model);}
  if(grid){worldScene.remove(grid);grid.geometry.dispose();grid.material.dispose();}
  model=new THREE.Group();worldScene.add(model);
  const {width,depth}=scene.bounds;
  if(scene.terrain.kind==='flat')cube(0,0,0,width,5,depth,'minecraft:grass_block');
  else{
    cube(0,-1,0,width,5,depth,'minecraft:stone_bricks');
    const geometry=new THREE.PlaneGeometry(width,depth,48,48);geometry.rotateX(-Math.PI/2);geometry.translate(width/2,0,depth/2);
    const positions=geometry.attributes.position;for(let i=0;i<positions.count;i++)positions.setY(i,groundAt(scene,positions.getX(i),positions.getZ(i))+1);
    geometry.computeVertexNormals();const terrain=new THREE.Mesh(geometry,material('minecraft:grass_block'));terrain.receiveShadow=true;model.add(terrain);
  }
  let detailedBuildings=0;
  if(scene.objects.length>2000)$('stats-caption').textContent='Preview limited to 2,000 objects; exports include all.';
  for(const object of scene.objects.slice(0,2000)){
    const m=object.materials,points=object.geometry.coordinates,base=object.base_y;
    if(object.kind==='building'){
      polygon(object,base,object.height,m.wall);
      const top=base+object.height;
      if(object.roof_shape==='flat')polygon(object,top,1,m.roof);
      else{
        const xs=points.map(p=>p[0]),zs=points.map(p=>p[1]),axis=Math.max(...xs)-Math.min(...xs)<Math.max(...zs)-Math.min(...zs)?0:1;
        const values=points.map(p=>p[axis]),low=Math.min(...values),high=Math.max(...values),mid=(low+high)/2;
        const halves=[true,false].map(less=>{const out=[];let prev=points.at(-1),was=less?prev[axis]<=mid:prev[axis]>=mid;
          for(const p of points){const inside=less?p[axis]<=mid:p[axis]>=mid;if(inside!==was){const t=(mid-prev[axis])/(p[axis]-prev[axis]);out.push([prev[0]+t*(p[0]-prev[0]),prev[1]+t*(p[1]-prev[1])]);}if(inside)out.push(p);prev=p;was=inside;}return out;});
        for(const half of halves){if(half.length<3)continue;const vertices=half.flatMap(p=>[p[0],top+(1-Math.abs((p[axis]-low)/Math.max(.01,high-low)*2-1))*object.roof_height,p[1]]);
          const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(vertices,3));geometry.setIndex(THREE.ShapeUtils.triangulateShape(half.map(p=>new THREE.Vector2(...p)),[]).flat());geometry.computeVertexNormals();const roof=new THREE.Mesh(geometry,material(m.roof));roof.castShadow=true;model.add(roof);}
      }
      const levels=Math.min(object.levels||2,8),count=points.length;
      if(detailedBuildings++>=120)continue;
      for(let edge=0;edge<count;edge++){
        const a=points[edge],b=points[(edge+1)%count],length=Math.hypot(b[0]-a[0],b[1]-a[1]);
        if(length<4)continue;
        const windows=Math.min(12,Math.floor(length/4));
        for(let level=0;level<levels;level++)for(let i=0;i<windows;i++){
          const t=(i+1)/(windows+1),x=a[0]+(b[0]-a[0])*t,z=a[1]+(b[1]-a[1])*t;
          const win=cube(-.85,0,-.12,1.7,1.6,.24,m.window);win.position.set(x,base+1.8+level*object.height/levels,z);win.rotation.y=-Math.atan2(b[1]-a[1],b[0]-a[0]);
        }
      }
    }else if(object.kind==='road'){
      for(let i=0;i<points.length-1;i++){
        const a=points[i],b=points[i+1],y=groundAt(scene,(a[0]+b[0])/2,(a[1]+b[1])/2)+1.1;
        segment(a,b,object.width+2*object.sidewalk_width,y,m.sidewalk);segment(a,b,object.width,y+.12,m.road);
        if(object.markings){const length=Math.hypot(b[0]-a[0],b[1]-a[1]);for(let d=0;d<length;d+=10){const p=t=>[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t];segment(p(d/length),p(Math.min(d+4,length)/length),.45,y+.28,'minecraft:quartz_block');}}
      }
    }else if(object.kind==='tree'){
      const [x,z]=points,r=object.canopy_radius||2.5,h=object.height;
      cube(x-.45,base,z-.45,.9,h*.8,.9,m.trunk);cube(x-r,base+h*.4,z-r,r*2,h*.4,r*2,m.leaves);cube(x-r*.7,base+h*.75,z-r*.7,r*1.4,h*.25,r*1.4,m.leaves);
    }else if(object.kind==='water'||object.kind==='park')polygon(object,base+(object.kind==='water'?.2:.05),.1,m[object.kind==='water'?'water':'grass']);
  }
  grid=new THREE.GridHelper(Math.max(width,depth)*1.3,26,0x69816a,0x425753);grid.position.set(width/2,-.1,depth/2);grid.visible=$('toggle-grid').getAttribute('aria-pressed')==='true';worldScene.add(grid);
  resetCamera();
}

function renderPhotos(){
  $('photo-list').replaceChildren();
  for(const photo of state.photos){
    const card=document.createElement('div');card.className='photo-card';
    const img=document.createElement('img');img.src=photo.url;img.alt='Uploaded reference photo';
    const info=document.createElement('div');info.className='photo-info';
    const title=document.createElement('strong');title.textContent=photo.filename;
    const summary=document.createElement('p');summary.className='helper';summary.textContent=`${photo.projection} · ${photo.suggested_style} suggestion`;
    const remove=document.createElement('button');remove.type='button';remove.textContent='Remove';remove.className='photo-remove';remove.setAttribute('aria-label',`Remove ${photo.filename}`);remove.onclick=()=>{state.photos=state.photos.filter(p=>p.id!==photo.id);renderPhotos();};
    const select=document.createElement('select');select.setAttribute('aria-label',`Material style for ${photo.filename}`);
    for(const style of ['auto','brick','stone','wood','modern']){const option=new Option(style==='auto'?'Suggested palette':style[0].toUpperCase()+style.slice(1),style);select.add(option);}
    select.value=photo.style;select.onchange=()=>photo.style=select.value;
    const controls=document.createElement('div');controls.className='photo-fields';
    for(const [key,label,min,max] of [['lat','Latitude',-85,85],['lon','Longitude',-180,180],['radius_m','Reach (m)',5,250]]){
      const wrap=document.createElement('label');wrap.textContent=label;const input=document.createElement('input');input.type='number';input.step='any';input.min=min;input.max=max;input.value=photo[key];input.required=true;input.onchange=()=>photo[key]=Number(input.value);wrap.append(input);controls.append(wrap);
    }
    info.append(title,summary,select,remove);card.append(img,info,controls);$('photo-list').append(card);
  }
}
async function upload(files){
  if(state.busy||!files.length)return;
  if(files.length+state.photos.length>12){toast('Choose up to 12 photos.',true);return;}
  const form=new FormData();for(const file of files)form.append('files',file);
  $('photo-files').disabled=true;toast('Reading photo palettes locally…');
  try{const result=await api('/api/photos',{method:'POST',body:form});const [lat,lon]=center();for(const photo of result.photos)state.photos.push({...photo,lat,lon,style:'auto',radius_m:80});renderPhotos();toast('Photos added. Check their position and material style.');}
  catch(error){toast(error.message,true);}finally{$('photo-files').disabled=false;$('photo-files').value='';}
}
function progress(job){
  $('pipeline-message').textContent=job.error||job.message;$('pipeline-percent').textContent=job.status==='failed'?'NEEDS ATTENTION':`${job.progress}%`;
  $('progress-bar').style.width=`${job.progress}%`;
  document.querySelector('.progress-track').setAttribute('aria-valuenow',job.progress);
  const stage=job.progress<30?0:job.progress<65?1:2;
  document.querySelectorAll('.pipeline-step').forEach((step,i)=>{step.classList.toggle('active',i===stage&&job.status!=='complete');step.classList.toggle('complete',i<stage||job.status==='complete');});
}
async function generate(event){
  event.preventDefault();if(state.busy)return;
  state.busy=true;$('generate-button').disabled=true;$('demo-button').disabled=true;$('exports-panel').hidden=true;
  try{
    const photos=state.photos.map(({id,lat,lon,style,radius_m})=>({id,lat,lon,style,radius_m}));
    const job=await post('/api/generate',{name:$('world-name').value,bbox:bbox(),source:state.source,photos,terrain:$('terrain').checked,meters_per_block:Number($('resolution').value)});
    state.job=job.id;sessionStorage.setItem('realityforge-job',job.id);
    await pollJob(job.id);
  }catch(error){toast(error.message,true);$('pipeline-message').textContent=error.message;}
  finally{state.busy=false;$('generate-button').disabled=false;$('demo-button').disabled=false;}
}
async function pollJob(id){
  while(true){
    const job=await api(`/api/jobs/${id}`);progress(job);
    if(job.status==='failed')throw new Error(job.error||'Generation failed.');
    if(job.status==='complete'){
      state.job=id;
      const scene=await api(job.scene_url);renderScene(scene);
      for(const key of ['world','schematic','model','scene','report'])$(`download-${key}`).href=job.files[key];
      let mtl=$('download-materials');if(job.files.materials){if(!mtl){mtl=document.createElement('a');mtl.id='download-materials';mtl.textContent='Model colours .mtl';mtl.download='';document.querySelector('.secondary-downloads').append(mtl);}mtl.href=job.files.materials;}
      $('exports-panel').hidden=false;$('provenance-note').textContent=(scene.source==='demo'?'Synthetic demo. ':'Live map generation. ')+scene.limitations.join(' ');
      let install=$('install-world');if(!install){install=document.createElement('button');install.type='button';install.id='install-world';install.className='button button-outline';install.textContent='Install in Minecraft saves';install.onclick=async()=>{install.disabled=true;try{const result=await post(`/api/jobs/${state.job}/install`,{});$('pipeline-message').textContent=`${result.message} ${result.path}`;toast('Installed in a new Minecraft save folder.');}catch(error){toast(error.message,true);}finally{install.disabled=false;}};document.querySelector('.secondary-downloads').append(install);}
      toast('Your Minecraft Java world is ready.');return;
    }
    await new Promise(resolve=>setTimeout(resolve,800));
  }
}

$('world-form').addEventListener('submit',generate);
for(const id of ['latitude','longitude','coverage','resolution','use-bbox','bbox-west','bbox-south','bbox-east','bbox-north'])$(id).addEventListener('change',()=>{if(id==='coverage')state.size=Math.max(50,Math.min(1000,Number($('coverage').value)));if(['latitude','longitude'].includes(id))setSource('live');updateLocation(true);});
document.querySelectorAll('[data-coverage]').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('[data-coverage]').forEach(b=>{b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
  const custom=button.dataset.coverage==='custom';$('custom-coverage').hidden=!custom;state.size=custom?Number($('coverage').value):Number(button.dataset.coverage);$('use-bbox').checked=false;updateLocation(true);
}));
$('parse-location').onclick=async()=>{try{const location=await post('/api/location',{value:$('google-location').value});$('latitude').value=location.lat;$('longitude').value=location.lon;$('use-bbox').checked=false;setSource('live');updateLocation(true);toast('Location set.');}catch(error){toast(error.message,true);}};
$('demo-button').onclick=async()=>{if(state.source==='demo'){setSource('live');toast('Live map mode selected.');return;}try{const demo=await api('/api/demo');setSource('demo');$('world-name').value=demo.name;renderScene(demo.scene);toast('Offline demo selected. Click Generate demo world to export.');}catch(error){toast(error.message,true);}};
$('photo-files').onchange=event=>upload([...event.target.files]);
for(const name of ['dragenter','dragover'])$('upload-zone').addEventListener(name,e=>{e.preventDefault();$('upload-zone').classList.add('dragging');});
for(const name of ['dragleave','drop'])$('upload-zone').addEventListener(name,e=>{e.preventDefault();$('upload-zone').classList.remove('dragging');if(name==='drop')upload([...e.dataTransfer.files]);});
$('reset-view').onclick=()=>resetCamera();$('top-view').onclick=()=>resetCamera(true);
$('toggle-grid').onclick=()=>{const visible=$('toggle-grid').getAttribute('aria-pressed')!=='true';$('toggle-grid').setAttribute('aria-pressed',visible);if(grid)grid.visible=visible;};
$('fullscreen').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('preview-panel').requestFullscreen();}catch{toast('Fullscreen is unavailable in this browser.');}};
setupPreview();initializeMap();updateLocation();
try{await api('/api/health');$('engine-dot').classList.add('online');$('engine-status').textContent='Local engine online';const demo=await api('/api/demo');renderScene(demo.scene);
  const previous=sessionStorage.getItem('realityforge-job');if(previous){const job=await api(`/api/jobs/${previous}`);if(job.status==='complete')await pollJob(previous);}
}catch(error){$('engine-status').textContent='Engine unavailable';$('preview-loading').hidden=true;toast(error.message,true);}
