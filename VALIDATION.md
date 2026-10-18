# RealityForge detail and scale validation

Validated on Windows with Python 3.11 on 17 October 2026.

## Completed

- Sidewalk trees survive export. Carriageway trees snap within three blocks or carry a drop reason. Reports distinguish mapped, placed, snapped and dropped vegetation.
- Nine supported tree/shrub styles, tagged species/genus/leaf type/height, seeded branching/crowns, bare winter override, photograph foliage hints, denser parks and forests. Preview, OBJ and voxels consume shared parts.
- Five user-provided local tree templates are cropped/resampled to real map scale; 58 of Toronto's trees use them. Conifers, bare/flowering trees and shrubs retain suitable original assets.
- Ten vehicle models, varied paints/windows/wheels/lights, road-aligned footprints, intersection/sidewalk avoidance, parking and limited traffic. Streetcars require mapped tram tracks.
- Dressed static armour-stand pedestrians, deterministic poses/orientations, density bounds, Sponge v 2 Entities and Java 1.20.4 separate entities/*.mca files.
- Extended OSM detail query and matching small-area fallback filters, versioned cache, benches with slab seats, fences, lights, bus stops, crossings and playground assets. Underground garages are excluded from surface parking.
- Reversible manual crops, conservative overlay trim, metadata/explicit panorama selection, true EXIF heading, FOV, estimated object distance, low-confidence vegetation estimates, per-photo tree overrides and manual map annotations.
- Photo-derived people/vehicles are supported only when compatible segmentation supplies counts. No people/car counts are guessed from image colour.
- 5 km square selection at 8 m/block; physical-scale trees/vehicles/furniture. Vehicles under 2 blocks long are omitted, people use small stands at 2 m/block and are omitted at 4/8 m to avoid giant markers.

## Toronto before / after

Same bounding box: [-79.3847,43.6521,-79.3817,43.6543], approximately 242×245 metres,1 m/block, public terrain enabled.

| Measurement | Before | After |
|---|---:|---:|
| Vegetation scene candidates |49|132|
| Visible old center-column trunks / new rooted assets checked |23|132|
| Mapped OSM trees retained |not separately audited|34/34|
| Vegetation styles present |1|8|
| Pedestrians |0|29|
| Vehicles |0|19|
| Street furniture/detail objects |0|54|
| Block palette entries |23|59|
| Non-air blocks |542,565|550,001|

After contains 100 trees plus 32 shrubs: birch 11, cherry 12, darkoak 11, maple 25, oak 22, pine 11, spruce 8, shrubs 32. All 132 root neighbourhoods contain wood in schematic readback. Two procedural trees snapped; zero dropped in this sample. Species without map evidence are estimates. Expanded OSM queries contribute additional features, so this measures the complete improvement, not only the occupancy fix.

19 vehicles comprise 10 parked at roads,8 in surface lots and 1 traffic vehicle. Four vehicle types fit this particular sample; allten models are covered by tests. Pedestrians use 11,980 m² of mapped walkable area, giving a 29 person procedural limit.

## Tests and runs

- Full automated suite:83 passed. One upstream Starlette TestClient deprecation warning remains.
- JavaScript syntax checks for app.js and photo-controls.js pass; git diff --check passes.
- Tests cover deterministic scene/voxel output, Toronto species diversity, actual sidewalk tree blocks, snap/drop reasons, physical scales, five-kilometre bounds, car headings/full-footprint collisions, annotations and crop/metadata/ONNX fallback behavior.
- Independent binary readback tests cover actual Sponge entity tags and Java entity regions, coordinates, UUIDs, poses, outfits, chunk/region boundaries, and small armour stands.
- Final Toronto export independently read back 29 armour stands from the schematic and 29 from entities/*.mca.
- The offline 250 m sample was regenerated:51 trees,21 pedestrians,30 vehicles,50 palette entries,377,792 non-air blocks.
- Full synthetic 5 km export at 8 m/block completed:625×625 blocks, approximately 32.44 seconds on this run. No live 5 km city export was verified; public-service limits and complex geometry can make it slower or fail.
- Browser smoke checks: enriched instanced preview, 5 km automatically selecting 8 m, return to 250 m restoring 1 m, map tree annotation without moving the area, successful generation to 100%, and world/schematic/OBJ/MTL/JSON/report links.
- Updated live sample: data/live-toronto-final. Previous sample retained: data/live-toronto-before-detail. Offline sample: data/offline-demo-enriched. Final overview check: data/five-km-verified.

## Deliberate omissions and limits

- sampletree.schem is a large multi-tree display/collection; shrinking it into one tree would be incorrect. The five individual ZIP trees are used instead, sized to the map.
- No licence file accompanied the reference ZIP, so original reference files are not bundled in the source archive. Local sample worlds include scaled geometry as requested; missing local references fall back to original procedural assets.
- Exact tree species/shape matching from arbitrary screenshots is not reliable. OSM attributes, photo appearance hints and explicit overrides are used. Bare-looking images are flagged conservatively; users can select bare winter style.
- Complex screenshot UI is not automatically erased; manual crop exists to avoid discarding real scene content. Aspect ratio alone no longer labels a panorama.
- Optional offline ONNX support is an adapter for an explicitly supplied compatible model, not a bundled verified model. No real weights were installed or tested.
- Armour stands are static decorative figures, not walking AI NPCs. Their mesh preview is a silhouette, not a literal rendering of Minecraft armour geometry.
- Roof stairs/slabs throughout buildings, balconies/storefronts and biome-aware leaf tinting remain deferred. Civic building defaults were corrected to avoid invented pitched roofs, but landmark architecture remains approximate.
- Google Maps URLs locate the area and open reference viewers; automated Street View imagery import/reconstruction is not implemented or verified.
- MinecraftJava 1.20.4/DataVersion 3700 is the target. Binary structure and roundtrip readback are verified. **No Minecraft client/server was launched; in-game compatibility remains unverified.**
