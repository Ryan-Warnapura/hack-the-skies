# RealityForge

Turn a bounding box into a Minecraft **Java Edition** world, with local photos guiding nearby building materials and vegetation estimates. Original Python + browser JavaScript implementation, inspired by [Arnis](https://github.com/louis-e/arnis). No Rust and no Bedrock output.

**Working MVP:** select a location, upload optional phone photos/panoramas, inspect the 3D scene, and download a `.schem` or Java world. No API key, paid service, GPU, or Minecraft mod is needed to generate a world.

## Start on Windows

1. Install **Python 3.11 or newer**, with “Add Python to PATH” enabled.
2. Double-click **`start.bat`** in this folder. The first launch installs Python dependencies.
3. Open **http://127.0.0.1:8000** in your browser. Keep the terminal open.
4. Try **Offline demo**, then generate. The demonstration geometry is synthetic and clearly labelled.
5. For a real location, choose live map mode, click the map or paste coordinates/a full Google Maps location URL, and generate.

From PowerShell in this folder, run `npm run dev` or `.\start.bat`. The npm command is a Windows shortcut to the Python launcher; `npm install` is not required.

If RealityForge is already running, the launcher prints its address and exits successfully. If another application occupies port 8000, run `npm run dev -- --port 8001` and open http://127.0.0.1:8001 instead.

On macOS/Linux: `python3 -m venv .venv`, activate the environment, `python -m pip install -r requirements-lock.txt`, then `python -m forge serve`.

## What the pipeline actually does

```text
Bounding box + OpenStreetMap + optional public elevation + local photos
  -> semantic 3D scene (scene.json + Wavefront scene.obj / scene.mtl)
  -> rasterized Minecraft block volume
  -> Sponge v2 schematic (realityforge.schem)
  -> read that schematic back
  -> Minecraft Java Anvil world (level.dat + region/*.mca + entities/*.mca)
  -> RealityForge-world.zip + report.json
```

The semantic scene is a 3D model of building footprints/heights, road paths, trees, green space, water and terrain. It is the authoritative model consumed by the voxelizer; the OBJ is its mesh export. Buildings have rule-based walls, windows, floors and roofs; trees use species-specific branching assets, asymmetric foliage and optional locally supplied tree schematics. People are static dressed armour stands; vehicles and street furniture have shared preview/OBJ/block geometry. This is **not photogrammetry or a learned 3D reconstruction model**.

Photos supply deterministic material, foliage-colour and low-confidence vegetation estimates. A reversible crop editor can remove screenshot overlays. Panorama mode comes from metadata or explicit selection, not aspect ratio. True EXIF camera heading, field of view and estimated object distance constrain nearby estimates; magnetic headings require correction. Per-photo tree styles include bare winter trees. Trees are chosen from OSM species/genus/leaf type first, then photo hints and seeded variation. Photos do not identify exact species reliably.

Photos supply a deterministic colour/material estimate. You can correct it using brick, stone, wood or modern style. That choice dresses nearby mapped buildings inside the photo's influence radius. Photos are anchored to the selected position; adjust their position when captured elsewhere. One panorama does not reveal hidden facades or distant geometry. Map features beyond the photos use map tags and reasonable defaults.

For the hackathon, use map coverage of **250 m across** first. Coverage is the side length of the selected square, **not a radius**. 250 m, 500 m, 1 km and **5 km** are supported, with custom bounds from 32 m to 5 km per side. The browser automatically selects a coarser scale when necessary. A 5 km square uses **8 metres/block**, producing 625 × 625 blocks. Use 1 metre/block in a smaller area for street-level detail. At coarse scales, vehicles shorter than two blocks are omitted, people use small armour stands at 2 m/block and are omitted at 4/8 m/block, and tiny features are quantized. Geometry is not inflated merely to keep details visible. Buildings and terrain heights are capped to the prototype's 128-block vertical budget; see each generation report.

## Added map detail and annotations

The generator supports oak, maple, birch, spruce, pine, dark oak, bare winter, flowering/cherry and shrub styles. Unknown species are estimates, not botanical observations. Sidewalk roots are allowed; carriageway roots move no more than three blocks or receive a recorded drop reason. Reports include OSM totals, placed/snapped/dropped counts and species distribution.

Map data includes tree rows, shrubs, gardens, playgrounds, surface parking, streetlights, benches, bollards, fences, bus stops, crossings and tram tracks. Underground parking is deliberately not drawn as a surface car park. Ten vehicle assets are available, placed only where their full footprints fit. Pedestrian density is capped near one per 400 m² of mapped walkable area and 300 total; explicit annotations may exceed the density estimate but not the hard cap.

Expand **Add or correct map details**, choose a tree, vehicle or person, and click inside the area. Choose species, vehicle type/colour or outfit; optionally correct a nearby existing object. Valid annotations enter `scene.json` as `user_annotation`, confidence 1. Invalid placements are explained in the report. Confidence 1 records a user decision, not independently verified geography.

## Local tree references

The supplied five-tree ZIP is used locally through gitignored `data/tree_templates.json`. Oversized trees (up to 115 blocks tall) are cropped and resampled to the selected tree height before placement. Wood, leaves, supported slab/stair states and orientation are retained. The large `sampletree.schem` appears to contain a collection/display and is not placed as one enormous tree.

Without the local ZIP/configuration, original procedural trees remain fully functional. Reference files are not included in the source archive because the provided bundle contains no licence file. The locally generated sample worlds do contain scaled reference-tree geometry. See `forge/tree_templates.py` for the optional config format.

## Import into Minecraft Java

You can click **Install in Minecraft saves** after generation to create a new save automatically. Each job gets a unique folder name; the app refuses to replace an existing folder. Custom launchers may use a different saves directory, in which case use the ZIP instructions below.

1. Download `RealityForge-world.zip` and unzip it.
2. Copy the enclosed **RealityForge** folder (the one containing `level.dat`) into `%APPDATA%\.minecraft\saves` on Windows. Use a unique folder name if RealityForge already exists.
3. Start **Minecraft Java 1.20.4**, select Singleplayer, and open your world. Creative mode and peaceful difficulty are configured.

macOS saves: `~/Library/Application Support/minecraft/saves`. Linux saves: `~/.minecraft/saves`.

Newer Java versions may upgrade the save; keep a backup. Older versions and Bedrock are not supported. The world is a bounded generated patch, with empty terrain beyond it. The app never modifies your existing Minecraft saves.

Alternatively, place `realityforge.schem` in your compatible WorldEdit `schematics` folder, then run `//schem load realityforge` and `//paste`. The schematic origin is its bottom northwest corner. `scene.obj` and `scene.mtl` should be kept together when opening the model in a 3D editor.

## Google Maps and no-photo mode

Paste a full Google Maps URL with coordinates, or `43.6532, -79.3832`. Short `maps.app.goo.gl` links must first be opened in your browser; copy the full address-bar URL. Maps and Street View links open Google's viewer for reference. Actual world geometry comes from OpenStreetMap, and photos are optional.

This prototype does not download or reconstruct Google Street View, satellite imagery or Google 3D tiles. [Google Maps Platform terms](https://cloud.google.com/maps-platform/terms) restrict derived content such as 3D models. A future imagery provider must explicitly license reconstruction. See [Maps URLs documentation](https://developers.google.com/maps/documentation/urls/get-started).

## Demo in two minutes

See `DEMO.md`. Always prepare the offline export before presenting. Public Overpass and elevation services can be unavailable or rate limited. For areas up to 250 m per side, an Overpass failure triggers one small OSM API download, cached locally. Larger areas require Overpass availability. Live map errors are shown instead of silently substituting the demo. Failed elevation retrieval uses flat terrain and records that fallback in the report.

## Command line

Run these from this folder:

```powershell
.\.venv\Scripts\python.exe -m forge generate --demo --output data\my-demo
.\.venv\Scripts\python.exe -m forge generate --bbox=-79.3847,43.6521,-79.3817,43.6543 --name "My neighbourhood" --terrain --output data\my-neighbourhood
.\.venv\Scripts\python.exe -m pytest -q
```

Choose an empty output folder. Bounding box order is **west longitude, south latitude, east longitude, north latitude**. The API is documented at `/api/docs`. Photos are supported through the browser/API; the command-line shortcut is map-only.

## Files and privacy

`forge/geography.py` fetches bounded map/elevation data. `photos.py` estimates appearance and handles crop/camera hints. `enrichment.py`, `population.py`, `assets.py` and `tree_templates.py` build shared detailed assets. `scene.py` builds models and voxelizes them. `minecraft.py` writes schematic and Java formats. `pipeline.py` enforces the conversion sequence. `app.py` is the local FastAPI service. `web/` contains the interface and vendored browser libraries.

Uploaded images remain in `data/uploads/` on this computer, resized to at most 4,096 pixels and stripped of embedded EXIF metadata. A sanitized original supports reversible cropping; selected GPS/direction/panorama metadata is retained in local JSON sidecars for analysis. They are not sent to AI services. `data/jobs/` contains exports; `data/cache/` contains public map/elevation responses. All three are excluded from Git. Delete these folders to clear your local history while the app is stopped. The map loads public OSM tiles; live generation sends the selected bounding box to Overpass and requests public terrain tiles if enabled.

The server binds to `127.0.0.1`. It is a local hackathon tool, not a multi-user hosted service. Deployment needs authentication, isolated workers, storage quotas and a job system.

## Beginner Git guide

Git has been initialized in this folder on branch `main`. You can use Git locally without GitHub. Dependencies, personal photos and generated worlds are ignored.

```powershell
git status                      # Show changed files
git diff                        # Review your changes
git config user.name "Your Name"
git config user.email "you@example.com"
git add .                       # Stage the project files
git commit -m "Describe what changed"
git log --oneline                # View saved checkpoints
```

To upload later, create an empty repository on GitHub, then use `git remote add origin YOUR_REPOSITORY_URL` and `git push -u origin main`. No remote repository is created by this project setup.

## Limitations and next steps

- Default photo analysis uses conservative colour/component heuristics, not recovered depth or exact geometry. People/vehicle counts stay unknown unless a compatible optional local ONNX model is supplied. The adapter has graceful fallback but no real model weights were installed or validated. Complex screenshot overlays require manual cropping. Panoramas are not projected into a textured mesh.
- Unmapped objects cannot be recovered from map data. Architectural details, interiors and landmarks use generic assets; complex multipolygon buildings, bridges, tunnels and water levels are approximate.
- Public DEM terrain is coarse, not survey-quality. Optional height clipping is documented in the scene report.
- Browser preview is a lightweight semantic view; exported Minecraft blocks add voxel-specific detailing. Terrain and assets are approximations.
- File structure and round-trip exports are tested automatically. Actual Minecraft client loading is a separate verification step; see `VALIDATION.md` for the checks completed in this workspace.

After the demo: collect multi-view photo sets with camera poses, introduce segmentation/depth, align reconstructions to geographic footprints, build a richer asset library, and add in-game regression fixtures.

Detailed before/after measurements, completed checks and deferred items are in `VALIDATION.md`. Optional roof-stair upgrades, balconies, storefronts and biome-aware leaf tinting remain future work.

## Credits

Arnis by Louis Erbkamm is the architectural inspiration. Its provided ZIP and local source were inspected for project scope; no Rust code, Arnis assets or bundled models are included. Live map data: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL. Terrain: [Mapzen/AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) and [source dataset credits](https://github.com/tilezen/joerd/blob/master/docs/attribution.md). Browser library licences are in `web/vendor/`. See `THIRD_PARTY.md`.

Not an official Minecraft product. Not approved by or associated with Mojang or Microsoft.
