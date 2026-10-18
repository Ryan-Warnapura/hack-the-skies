"""One cuboid/ellipsoid asset representation shared by mesh and voxel exports."""
import math

import numpy as np

BLOCK_COLORS = {
    'oak_log':'#775b37','birch_log':'#e1d9b5','spruce_log':'#624729','dark_oak_log':'#493321',
    'cherry_log':'#643d43','oak_leaves':'#568a35','birch_leaves':'#80a64b','spruce_leaves':'#345b47',
    'dark_oak_leaves':'#375e32','azalea_leaves':'#638c42','flowering_azalea_leaves':'#936993',
    'cherry_leaves':'#e7abc0','mangrove_leaves':'#667e2e','moss_block':'#627d32',
    'lime_terracotta':'#778d39','yellow_terracotta':'#b88c32','orange_terracotta':'#b6652e',
    'red_terracotta':'#a4503b','brown_terracotta':'#785137','terracotta':'#aa765d',
    'white_terracotta':'#d2b5a6','light_gray_terracotta':'#9c8273','gray_terracotta':'#54433c',
    'white_concrete':'#d9ded4','orange_concrete':'#e87921','magenta_concrete':'#aa35a1',
    'light_blue_concrete':'#3aafd9','yellow_concrete':'#f1c532','lime_concrete':'#6baa1e',
    'pink_concrete':'#d96792','gray_concrete':'#41474c','light_gray_concrete':'#90928b',
    'cyan_concrete':'#168695','purple_concrete':'#6f259b','blue_concrete':'#344aa4',
    'brown_concrete':'#6b4729','green_concrete':'#506428','red_concrete':'#ac302b','black_concrete':'#22262d',
    'black_wool':'#26262a','light_gray_wool':'#93948c','white_wool':'#e9e9e3',
    'black_stained_glass':'#35424c','light_blue_stained_glass':'#7fb8c6','glass':'#bedbdd',
    'yellow_stained_glass':'#f8dc69','red_stained_glass':'#c93832','blue_stained_glass':'#364fc4',
    'sea_lantern':'#d4f2df','glowstone':'#e1bd68','redstone_block':'#bb2828',
    'iron_block':'#c5cbcb','iron_bars':'#989e9d','polished_andesite':'#969b97','andesite':'#858b84',
    'polished_diorite':'#c5c5bb','diorite':'#bbbdb1','polished_granite':'#a87967',
    'smooth_sandstone':'#dfd0a0','sandstone':'#d6c693','red_sandstone':'#b56632',
    'stone_bricks':'#898c85','mossy_stone_bricks':'#75836b','bricks':'#a45f4b','polished_blackstone':'#35333a',
    'oak_slab':'#ae8c59','spruce_slab':'#80603d','stone_slab':'#92928d','smooth_stone_slab':'#a5a69d',
    'deepslate_bricks':'#484a4e','deepslate_tiles':'#41454a','copper_block':'#ba775c',
    'weathered_copper':'#63a386','waxed_weathered_copper':'#63a386','prismarine_bricks':'#75a994',
    'oak_planks':'#ae8c59','spruce_planks':'#80603d','dark_oak_planks':'#4e3926','birch_planks':'#ccbb81',
    'quartz_block':'#e5e1d3','smooth_stone':'#a5a69d','grass_block':'#78964b','water':'#488fa9',
}

for _species in ('oak','birch','spruce','dark_oak','cherry','acacia','jungle','mangrove'):
    _color=BLOCK_COLORS.get(_species+'_log','#775b37')
    for _suffix in ('wood','fence','fence_gate','slab','stairs'):
        BLOCK_COLORS.setdefault(_species+'_'+_suffix,_color)
    for _suffix in ('log','wood'):
        BLOCK_COLORS.setdefault('stripped_'+_species+'_'+_suffix,BLOCK_COLORS.get(_species+'_planks',_color))


def part_mesh(part):
    """Return vertices/faces for the canonical asset primitive."""
    sx, sy, sz = part['size']
    if part.get('shape') == 'ellipsoid':
        vertices = []
        rows, cols = 6, 10
        for row in range(rows + 1):
            phi = math.pi * row / rows
            for col in range(cols):
                theta = 2 * math.pi * col / cols
                vertices.append([math.sin(phi)*math.cos(theta)*sx/2, math.cos(phi)*sy/2, math.sin(phi)*math.sin(theta)*sz/2])
        faces = [(r*cols+c, r*cols+(c+1)%cols, (r+1)*cols+(c+1)%cols, (r+1)*cols+c)
                 for r in range(rows) for c in range(cols)]
    else:
        vertices = [[x*sx/2,y*sy/2,z*sz/2] for y in [-1,1] for z in [-1,1] for x in [-1,1]]
        faces = [(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]
    angle = math.radians(part.get('yaw', 0)); c,s=math.cos(angle),math.sin(angle)
    cx,cy,cz=part['center']
    return [(cx+x*c+z*s,cy+y,cz-x*s+z*c) for x,y,z in vertices],faces


def voxelize_parts(blocks, obj, block_id):
    """Rasterize the same rotated asset primitives used in OBJ and the viewer."""
    height, depth, width = blocks.shape
    for part in obj.get('parts', []):
        cx,cy,cz=part['center']; sx,sy,sz=part['size']
        angle=math.radians(part.get('yaw',0));c,s=math.cos(angle),math.sin(angle)
        rx=abs(c)*sx/2+abs(s)*sz/2;rz=abs(s)*sx/2+abs(c)*sz/2
        x0,x1=max(0,math.floor(cx-rx)),min(width,math.ceil(cx+rx))
        z0,z1=max(0,math.floor(cz-rz)),min(depth,math.ceil(cz+rz))
        y0,y1=max(1,math.floor(cy-sy/2)),min(height,math.ceil(cy+sy/2))
        if x1<=x0 or z1<=z0 or y1<=y0:continue
        yy,zz,xx=np.ogrid[y0:y1,z0:z1,x0:x1]
        dx,dz=xx+.5-cx,zz+.5-cz
        lx,lz=dx*c-dz*s,dx*s+dz*c
        if part.get('shape')=='ellipsoid':
            mask=(lx/max(.5,sx/2))**2+((yy+.5-cy)/max(.5,sy/2))**2+(lz/max(.5,sz/2))**2<=1.05
        else:
            # Half-open bounds prevent an integer-centred thin trunk becoming 2x2.
            hx,hy,hz=max(.5,sx/2),max(.5,sy/2),max(.5,sz/2)
            mask=(lx>=-hx)&(lx<hx)&(lz>=-hz)&(lz<hz)&(yy+.5-cy>=-hy)&(yy+.5-cy<hy)
        target=blocks[y0:y1,z0:z1,x0:x1]
        # Tree crowns may overhang sidewalks but must not erase buildings/vehicles.
        if obj['kind'] in {'tree','shrub'} and ('leaves' in part['material'] or part.get('role')=='foliage'):
            mask=mask&(target==0)
        target[mask]=block_id(part['material'])
