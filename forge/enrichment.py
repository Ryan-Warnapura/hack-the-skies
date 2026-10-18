"""Photo-informed, reproducible vegetation and geographic street details.

Species selected without OSM evidence are labelled estimates. A photo provides
appearance hints, never a claim of recovered three-dimensional geometry.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from .assets import BLOCK_COLORS

SPECIES = ('oak', 'maple', 'birch', 'spruce', 'pine', 'dark_oak', 'bare', 'cherry', 'shrub')
TRUNKS = {'birch':'birch', 'spruce':'spruce', 'pine':'spruce', 'dark_oak':'dark_oak', 'cherry':'cherry'}
LEAVES = {'maple':'azalea', 'birch':'birch', 'spruce':'spruce', 'pine':'spruce', 'dark_oak':'dark_oak', 'cherry':'cherry', 'shrub':'azalea'}
WALKWAYS = {'footway','path','pedestrian','cycleway','steps','bridleway'}


def rng_for(value):
    return random.Random(int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], 'big'))


def _number(value, default):
    match = re.search(r'\d+(?:\.\d+)?', str(value))
    return float(match[0]) if match else default


def _base(scene, identifier, kind, point, tags=None, source='osm', confidence=.65):
    from .scene import _terrain_height
    x,z=point
    return {'id':identifier,'kind':kind,'center':list(point),'name':(tags or {}).get('name',kind.title()),
            'geometry':{'type':'point','coordinates':list(point),'holes':[]},
            'base_y':round(_terrain_height(scene['terrain'],x,z,scene['bounds']['width'],scene['bounds']['depth']))+1,
            'height':0,'materials':dict(scene['materials']),'tags':tags or {},'source':source,'confidence':confidence}


def import_details(scene, osm):
    from .scene import _projection, _clip_polygon, _clip_road
    width,depth,project=_projection(scene['bbox'],scene['meters_per_block'])
    nodes={e['id']:e for e in osm.get('elements',[]) if e['type']=='node'}
    existing={o['id'] for o in scene['objects']}
    for el in osm.get('elements',[]):
        tags=el.get('tags',{}); identifier=f"{el['type']}/{el['id']}"
        if el['type']=='node':
            p=project(el['lon'],el['lat'])
            if not (0<=p[0]<width and 0<=p[1]<depth):continue
            subtype={'street_lamp':'streetlight','bus_stop':'bus_stop','crossing':'crossing'}.get(tags.get('highway'))
            if tags.get('amenity')=='bench':subtype='bench'
            if tags.get('barrier')=='bollard':subtype='bollard'
            if subtype:
                obj=_base(scene,identifier,'street_detail',p,tags);obj['subtype']=subtype;scene['objects'].append(obj)
            elif tags.get('natural')=='shrub':
                obj=_base(scene,identifier,'tree',p,tags);obj['requested_species']='shrub';scene['objects'].append(obj)
            continue
        if el['type']!='way':continue
        geom=el.get('geometry') or [nodes[n] for n in el.get('nodes',[]) if n in nodes]
        points=[project(p['lon'],p['lat']) for p in geom if 'lon' in p and 'lat' in p]
        if len(points)<2:continue
        if tags.get('natural')=='tree_row' or tags.get('barrier')=='hedge':
            step=max(2,(6 if tags.get('natural')=='tree_row' else 2)/scene['meters_per_block'])
            index=0
            for a,b in zip(points,points[1:]):
                length=math.dist(a,b)
                for d in np.arange(0,length,step):
                    point=[a[k]+(b[k]-a[k])*d/max(.01,length) for k in (0,1)]
                    if not (0<=point[0]<width and 0<=point[1]<depth):continue
                    obj=_base(scene,f'{identifier}/tree/{index}','tree',point,tags,source='osm_tree_row')
                    if tags.get('barrier')=='hedge':obj['requested_species']='shrub'
                    scene['objects'].append(obj);index+=1
        elif tags.get('railway')=='tram' or tags.get('barrier')=='fence':
            for i,(a,b) in enumerate(zip(points,points[1:])):
                clipped=_clip_road(a,b,width,depth)
                if not clipped:continue
                center=[sum(p[k] for p in clipped)/2 for k in (0,1)]
                kind='tram_track' if tags.get('railway')=='tram' else 'street_detail'
                obj=_base(scene,f'{identifier}/{i}',kind,center,tags)
                obj['geometry']={'type':'polyline','coordinates':clipped,'holes':[]}
                obj.update(subtype='fence',width=max(1,2/scene['meters_per_block']))
                scene['objects'].append(obj)
        elif tags.get('amenity')=='parking' or tags.get('natural')=='scrub':
            if tags.get('amenity')=='parking' and (tags.get('parking') in {'underground','multi-storey','rooftop'} or str(tags.get('layer','')).startswith('-') or str(tags.get('level','')).startswith('-')):continue
            polygon=_clip_polygon(points,width,depth)
            if len(polygon)<3 or identifier in existing:continue
            center=[sum(p[k] for p in polygon)/len(polygon) for k in (0,1)]
            kind='parking' if tags.get('amenity')=='parking' else 'park'
            obj=_base(scene,identifier,kind,center,tags)
            obj['geometry']={'type':'polygon','coordinates':polygon,'holes':[]}
            obj['materials']['surface']='minecraft:gray_concrete';obj['vegetation']='scrub'
            scene['objects'].append(obj)
            if kind=='park':
                from .scene import _point_inside
                step=max(3,5/scene['meters_per_block'])
                for x in np.arange(max(1,min(p[0] for p in polygon)),min(width-1,max(p[0] for p in polygon)),step):
                    for z in np.arange(max(1,min(p[1] for p in polygon)),min(depth-1,max(p[1] for p in polygon)),step):
                        if _point_inside([x,z],polygon):
                            shrub=_base(scene,f'{identifier}/shrub/{x:.1f}/{z:.1f}','tree',[float(x),float(z)],source='procedural_park',confidence=.25)
                            shrub['requested_species']='shrub';scene['objects'].append(shrub)
        if tags.get('leisure')=='playground':
            center=[sum(p[k] for p in points)/len(points) for k in (0,1)]
            if 0<=center[0]<width and 0<=center[1]<depth:
                obj=_base(scene,identifier+'/equipment','street_detail',center,tags);obj['subtype']='playground';scene['objects'].append(obj)


def tree_masks(scene):
    """Sidewalks remain plantable. Only carriageways, water and buildings block roots."""
    from .scene import _polygon_mask
    w,d=scene['bounds']['width'],scene['bounds']['depth']
    solid=np.zeros((d,w),bool);road=np.zeros_like(solid)
    for obj in scene['objects']:
        if obj['kind'] in {'building','water'}:solid|=_polygon_mask(obj,w,d)
        elif obj['kind']=='road' and obj.get('road_type') not in WALKWAYS:
            im=Image.new('1',(w,d));ImageDraw.Draw(im).line([tuple(p) for p in obj['geometry']['coordinates']],fill=1,width=max(1,round(obj['width'])),joint='curve')
            road|=np.asarray(im,dtype=bool)
    return solid,road


def nearest_photo(obj, photos, scale):
    candidates=[]
    for photo in photos:
        dx,dz=[obj['center'][k]-photo['position'][k] for k in (0,1)]
        distance=math.hypot(dx,dz)*scale
        if distance>float(photo.get('radius_m',80)):continue
        if photo.get('heading') is not None and photo.get('projection')!='panorama':
            bearing=math.degrees(math.atan2(dx,-dz))%360
            if abs((bearing-float(photo['heading'])+180)%360-180)>float(photo.get('fov',70))/2:continue
        candidates.append((distance,photo))
    return min(candidates,key=lambda p:p[0])[1] if candidates else None


def choose_tree(obj, photo, rng):
    tags=obj.get('tags',{})
    description=' '.join(str(tags.get(k,'')) for k in ('species','species:en','genus','taxon')).lower()
    species=obj.get('requested_species'); method='user_annotation' if obj['source']=='user_annotation' else 'osm_attribute'
    if species not in SPECIES:species=None
    if not species:
        for key,terms in [('maple',('acer','maple')),('birch',('betula','birch')),('spruce',('picea','spruce')),
                          ('pine',('pinus','pine')),('cherry',('prunus','cherry')),('dark_oak',('quercus robur','dark oak')),('oak',('quercus','oak'))]:
            if any(term in description for term in terms):species=key;break
    if not species and tags.get('leaf_type')=='needleleaved':species=rng.choice(['spruce','pine'])
    if not species and photo and photo.get('tree_style','auto')!='auto':species=photo['tree_style'];method='photo_override'
    if not species:
        pool=['oak','maple','birch','dark_oak','cherry'] if tags.get('leaf_type')=='broadleaved' else ['oak','oak','maple','maple','birch','spruce','pine','dark_oak','cherry']
        species=rng.choice(pool);method='seeded_approximation'
    state=str(tags.get('seasonal_state',tags.get('leaf_state',''))).lower()
    if state in {'leafless','bare','winter'}:state='bare'
    elif state not in {'autumn','green'}:state='green'
    # A bare override is explicit; colour heuristics alone are too ambiguous to remove leaves.
    if species=='bare' or photo and photo.get('tree_style')=='bare' and species not in {'spruce','pine'}:state='bare'
    elif photo and photo.get('vegetation_state')=='autumn' and species not in {'spruce','pine'}:state='autumn'
    return species,state,method


def dress_tree(scene, obj, photo=None):
    """Original hand-built tree grammar: roots, branching limbs, uneven clusters."""
    rng=rng_for(obj['id']);world_scale=scene['meters_per_block'];scale=1;tags=obj.get('tags',{})
    species,state,method=choose_tree(obj,photo,rng)
    defaults={'oak':10,'maple':11,'birch':12,'spruce':14,'pine':15,'dark_oak':12,'bare':10,'cherry':7,'shrub':2}
    height=max(1.4 if species=='shrub' else 4,min(26/scale,_number(tags.get('height'),defaults[species]*rng.uniform(.8,1.3))/scale))
    diameter=_number(tags.get('diameter_crown'),height*scale*({'spruce':.36,'pine':.42,'birch':.38,'shrub':1.5}.get(species,.65)))/scale
    radius=max(.8,min(7/scale,diameter/2));base=obj['base_y'];x,z=obj['center']
    trunk='minecraft:'+TRUNKS.get(species,'oak')+'_log[axis=y]'
    leaf='minecraft:'+LEAVES.get(species,'oak')+'_leaves[persistent=true]'
    leaf_options=[leaf]
    if state=='autumn':
        leaf_options=['minecraft:yellow_terracotta','minecraft:orange_terracotta','minecraft:red_terracotta','minecraft:mangrove_leaves[persistent=true]']
        if photo and photo.get('foliage_colors'):
            def difference(material):
                a=BLOCK_COLORS[material.split(':')[-1].split('[')[0]]
                return min(sum((int(a[i:i+2],16)-int(c[i:i+2],16))**2 for i in (1,3,5)) for c in photo['foliage_colors'] if re.fullmatch(r'#[0-9a-fA-F]{6}',c))
            try:leaf_options.sort(key=difference)
            except ValueError:pass
        leaf=leaf_options[0]
    elif species in {'oak','maple','dark_oak','shrub'}:
        leaf_options += ['minecraft:azalea_leaves[persistent=true]' if species!='dark_oak' else 'minecraft:oak_leaves[persistent=true]']
        if photo:
            colors=[c for c in photo.get('foliage_colors',[]) if re.fullmatch(r'#[0-9a-fA-F]{6}',c)]
            if colors:
                choices=['oak','birch','dark_oak','azalea','mangrove']
                best=min(choices,key=lambda name:min(sum((int(BLOCK_COLORS[name+'_leaves'][i:i+2],16)-int(color[i:i+2],16))**2 for i in (1,3,5)) for color in colors))
                leaf='minecraft:'+best+'_leaves[persistent=true]';leaf_options=[leaf,leaf,leaf_options[0]]
    obj.update(species=species,vegetation_state=state,species_source=method,height=round(height,2),canopy_radius=round(radius,2),parts=[],appearance_method='branching_tree_asset_v2')
    obj['materials'].update(trunk=trunk,leaves=leaf)
    obj['color']=BLOCK_COLORS[leaf.split(':')[-1].split('[')[0]] if state!='bare' else BLOCK_COLORS[trunk.split(':')[-1].split('[')[0]]
    if photo:
        obj.update(photo_id=photo['id'],appearance_confidence=.3,photo_vegetation_state=photo.get('vegetation_state','unknown'))
    def part(center,size,material,shape='box'):
        obj['parts'].append({'shape':shape,'center':[round(v,3) for v in center],'size':[round(v,3) for v in size],'material':material,'role':'foliage' if material in leaf_options else 'wood'})
    def finish():
        obj['height_m']=round(height,2);obj['canopy_radius_m']=round(radius,2)
        obj['height']=round(height/world_scale,3);obj['canopy_radius']=round(radius/world_scale,3)
        for p in obj['parts']:
            p['center']=[round(origin+(v-origin)/world_scale,3) for v,origin in zip(p['center'],[x,base,z])]
            p['size']=[round(v/world_scale,4) for v in p['size']]
    def branch(a,b,thickness):
        length=math.dist(a,b)
        for t in np.linspace(0,1,max(2,math.ceil(length/.65))):
            p=[a[k]+(b[k]-a[k])*float(t) for k in range(3)]
            axis='x' if abs(b[0]-a[0])>abs(b[2]-a[2]) else 'z'
            part(p,[thickness]*3,trunk.replace('axis=y','axis='+axis))
    if species=='shrub':
        part([x,base+height*.35,z],[.65,height*.7,.65],trunk)
        for i in range(3):part([x+rng.uniform(-.3,.3)*radius,base+height*.65,z+rng.uniform(-.3,.3)*radius],[radius*1.4,height*.9,radius*1.4],rng.choice(leaf_options),'ellipsoid')
        finish();return
    thickness=max(.8,min(2.2,_number(tags.get('circumference'),2.8)/math.pi/scale))
    trunk_h=height*(.9 if species in {'spruce','pine','bare'} else .66)
    part([x,base+trunk_h/2,z],[thickness,trunk_h,thickness],trunk)
    if thickness>1.15:
        for dx,dz in [(1,0),(-1,0),(0,1),(0,-1)]:
            part([x+dx*thickness*.6,base+.35,z+dz*thickness*.6],[thickness*.8,.7,thickness*.8],trunk)
    if species in {'spruce','pine'}:
        tiers=5 if species=='spruce' else 3
        for i in range(tiers):
            t=i/max(1,tiers-1);y=base+height*(.3+t*.59 if species=='spruce' else .56+t*.32)
            r=radius*(1-t*.78)
            angle=rng.random()*math.tau
            for j in range(4):
                a=angle+j*math.pi/2;bx=x+math.cos(a)*r*.65;bz=z+math.sin(a)*r*.65
                branch([x,y-.4,z],[bx,y,bz],.7)
                part([bx,y+.35,bz],[r*1.25,max(1,height*.14),r*1.25],leaf,'ellipsoid')
        part([x,base+height*.96,z],[max(1,radius*.7),height*.14,max(1,radius*.7)],leaf,'ellipsoid')
    else:
        # Separate uneven lobes expose branch forks and avoid identical ball crowns.
        count=rng.randint(5,8)
        for i in range(count):
            angle=math.tau*i/count+rng.uniform(-.35,.35)
            reach=radius*rng.uniform(.55,1.0)
            tip=[x+math.cos(angle)*reach,base+height*rng.uniform(.61,.84),z+math.sin(angle)*reach]
            start=[x,base+height*rng.uniform(.36,.57),z]
            elbow=[start[k]+(tip[k]-start[k])*.56 for k in range(3)];elbow[1]-=.35
            branch(start,elbow,max(.65,thickness*.7));branch(elbow,tip,.65)
            if state=='bare':
                for sign in [-1,1]:branch(tip,[tip[0]+math.cos(angle+sign*.6)*radius*.4,tip[1]+height*.14,tip[2]+math.sin(angle+sign*.6)*radius*.4],.5)
            else:
                part([tip[0],tip[1]+height*.05,tip[2]],[radius*rng.uniform(.95,1.4),height*rng.uniform(.23,.36),radius*rng.uniform(.9,1.35)],rng.choice(leaf_options),'ellipsoid')
        if state!='bare':part([x+radius*.1,base+height*.87,z-radius*.08],[radius*1.2,height*.29,radius*1.25],leaf,'ellipsoid')
    finish()


def enrich_scene(scene, osm, photos, annotations):
    from .scene import _projection, _terrain_height
    from .population import populate_scene
    import_details(scene,osm)
    w,d,project=_projection(scene['bbox'],scene['meters_per_block'])
    # Annotated trees are authoritative species choices, subject to root placement safety.
    replacements={a.get('replace_id') for a in annotations if a['kind']=='tree' and a.get('replace_id')}
    originals={o['id']:o for o in scene['objects'] if o['id'] in replacements and o['kind']=='tree'}
    for i,annotation in enumerate(annotations):
        if annotation['kind']!='tree':continue
        point=project(annotation['lon'],annotation['lat'])
        obj=_base(scene,'annotation/'+str(annotation.get('id') or i),'tree',point,source='user_annotation',confidence=1.)
        obj['requested_species']=annotation.get('tree_species') or 'oak'
        obj['replace_id']=annotation.get('replace_id');scene['objects'].append(obj)
    # Estimate only bounded nearby trees from image evidence; positions are not observations.
    for photo in photos:
        count=(photo.get('estimated_counts') or {}).get('trees') or 0
        if count<1 or photo.get('heading') is None:continue
        distance=min(150,max(5,float(photo.get('estimated_distance_m',25))))/scene['meters_per_block']
        nearby=sum(o['kind']=='tree' and math.dist(o['center'],photo['position'])<=distance*1.4 for o in scene['objects'])
        for i in range(min(12,max(0,int(count)-nearby))):
            fov=360 if photo.get('projection')=='panorama' else float(photo.get('fov',70))
            angle=math.radians(float(photo['heading'])+((i+.5)/min(12,max(1,count))-0.5)*fov)
            p=[photo['position'][0]+math.sin(angle)*distance,photo['position'][1]-math.cos(angle)*distance]
            obj=_base(scene,f"photo/{photo['id']}/tree/{i}",'tree',p,source='photo_estimate',confidence=.2)
            obj['photo_id']=photo['id'];scene['objects'].append(obj)
    solid,road=tree_masks(scene);occupied=set();records=[];kept=[]
    candidates=[o for o in scene['objects'] if o['kind']=='tree']
    # OSM/annotations take priority over dense procedural planting.
    candidates.sort(key=lambda o:(0 if o['source']=='user_annotation' else 1 if o['source'].startswith('osm') else 2,o['id']))
    removed=set()
    for obj in candidates:
        if obj['id'] in removed:continue
        original=list(obj['center']);x,z=original;ix,iz=math.floor(x),math.floor(z)
        reason=None;snapped=False
        if not (0<=ix<w and 0<=iz<d):reason='outside_bounds'
        elif solid[iz,ix]:reason='building_or_water'
        elif road[iz,ix] or (ix,iz) in occupied:
            options=[]
            for dz in range(-3,4):
                for dx in range(-3,4):
                    nx,nz=ix+dx,iz+dz;distance=math.hypot(nx+.5-x,nz+.5-z)
                    if distance<=3 and 0<=nx<w and 0<=nz<d and not solid[nz,nx] and not road[nz,nx] and (nx,nz) not in occupied:options.append((distance,nx,nz))
            if options:
                _,ix,iz=min(options);obj['center']=[ix+.5,iz+.5];obj['geometry']['coordinates']=obj['center'];snapped=True
            else:reason='no_free_verge_within_3_blocks' if road[iz,ix] else 'duplicate_root'
        record={'id':obj['id'],'source':obj['source'],'status':'dropped' if reason else 'snapped' if snapped else 'placed','original_position':original}
        if reason:record['reason']=reason
        else:
            if obj.get('replace_id') in originals:removed.add(obj['replace_id'])
            occupied.add((ix,iz));obj['base_y']=round(_terrain_height(scene['terrain'],*obj['center'],w,d))+1
            photo=nearest_photo(obj,photos,scene['meters_per_block'])
            dress_tree(scene,obj,photo)
            from .tree_templates import apply_tree_template
            apply_tree_template(obj)
            kept.append(obj);record['position']=obj['center'];record['species']=obj['species']
        records.append(record)
    # A successfully replaced map tree is counted explicitly, rather than silently dropped.
    for identifier in sorted(removed):
        records.append({'id':identifier,'source':originals[identifier]['source'],'status':'dropped','reason':'replaced_by_user_annotation','original_position':originals[identifier]['center']})
    scene['objects']=[o for o in scene['objects'] if o['kind']!='tree']+kept
    scene['tree_placement']=records
    stats=scene['stats'];stats.update(tree_osm_total=sum(r['source']=='osm' for r in records),tree_candidates=len(records),
        trees_placed=len(kept),trees_snapped=sum(r['status']=='snapped' for r in records),trees_dropped=sum(r['status']=='dropped' for r in records),
        tree_osm_placed=sum(r['source']=='osm' and r['status']!='dropped' for r in records),
        tree_osm_snapped=sum(r['source']=='osm' and r['status']=='snapped' for r in records),
        tree_osm_dropped=sum(r['source']=='osm' and r['status']=='dropped' for r in records),
        tree_species=dict(sorted(Counter(o['species'] for o in kept).items())),
        tree_drop_reasons=dict(Counter(r['reason'] for r in records if r.get('reason'))))
    # Expand plausible masonry palettes only where neither OSM nor photographs specify it.
    for obj in scene['objects']:
        if obj['kind']=='building' and obj['source']=='osm' and not obj.get('tags',{}).get('building:material'):
            rng=rng_for(obj['id']);mat=obj['materials']
            pool=['bricks','stone_bricks','white_terracotta','smooth_sandstone','light_gray_terracotta','polished_andesite']
            if obj.get('building_type') in {'house','detached','residential'}:pool+=['red_sandstone','oak_planks']
            mat['wall']='minecraft:'+rng.choice(pool);mat['trim']='minecraft:'+rng.choice(['quartz_block','smooth_stone','stone_bricks','sandstone'])
            mat['roof']='minecraft:'+rng.choice(['deepslate_tiles','deepslate_bricks','dark_oak_planks','gray_concrete'])
            obj['color']=BLOCK_COLORS[mat['wall'].split(':')[-1]];obj['appearance_method']='seeded_material_estimate'
    populate_scene(scene,annotations,photos=photos)
    scale=scene['meters_per_block']
    if scale>=4:
        people=[o for o in scene['objects'] if o['kind']=='person']
        scene['objects']=[o for o in scene['objects'] if o['kind']!='person']
        stats.update(pedestrians=0,person=0,pedestrians_omitted_at_overview_scale=len(people))
        stats['annotations_dropped'] += [{'id':o['id'],'reason':'person_below_resolution_at_overview_scale'} for o in people if o['source']=='user_annotation']
        scene['limitations'].append('Pedestrians are omitted at 4 and 8 m/block because Java 1.20.4 armour stands cannot be scaled down enough to represent human-sized people.')
    elif scale==2:
        for obj in scene['objects']:
            if obj['kind']!='person':continue
            anchor=[obj['center'][0],obj['base_y'],obj['center'][1]]
            for p in obj['parts']:
                p['center']=[a+(v-a)/2 for a,v in zip(anchor,p['center'])]
                p['size']=[v/2 for v in p['size']]
            obj['height']/=2;obj['entity_small']=True
    scene['limitations']=[s for s in scene['limitations'] if not s.startswith('Minecraft armour stands remain')]
    scene['block_colors']=BLOCK_COLORS
    stats['template_trees']=sum(o.get('appearance_method')=='local_schematic_tree_template' for o in kept)
    stats['annotations_dropped'] += [{'id':r['id'],'reason':r['reason']} for r in records if r['source']=='user_annotation' and r['status']=='dropped']
    stats.update(dict(Counter(o['kind'] for o in scene['objects'])));stats['objects']=len(scene['objects'])
    stats['photos_used']=len({o['photo_id'] for o in scene['objects'] if o.get('photo_id')})
    heights=[scene['bounds']['height']]+[math.ceil(p['center'][1]+p['size'][1]/2)+1 for o in scene['objects'] for p in o.get('parts',[])]
    scene['bounds']['height']=min(128,max(heights))
    scene['asset_version']='branching-trees-and-population-v2'
    if scene['meters_per_block']>1:
        scene['limitations'].append(f"Overview scale: 1 block represents {scene['meters_per_block']:g} metres. Small details are quantized or omitted; use 1 m/block in a smaller area for street-level detail.")
    scene['limitations'].append('Unspecified species, pedestrians and traffic are seeded approximations, not observed real objects. Photo tree positions are low-confidence estimates.')
    scene['limitations'].append('Preview/OBJ use shared continuous asset primitives; Minecraft rasterizes them to blocks. Armour stands are static, not walking AI NPCs. In-game compatibility is not yet verified.')
