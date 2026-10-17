"""Explicit model -> schematic -> world pipeline, shared by HTTP and CLI."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .geography import demo_osm, fetch_elevation, fetch_osm, validate_bbox
from .minecraft import write_java_world, write_schematic
from .scene import build_scene, export_obj, scene_to_voxels


def generate(request: dict, output: Path, cache: Path, progress=lambda *args: None):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    bbox = validate_bbox(request["bbox"])
    demo = request.get("source") == "demo"
    warnings = []
    progress(8, "geography", "Loading synthetic demo geometry" if demo else "Fetching real streets, footprints and green spaces")
    osm = demo_osm(bbox) if demo else fetch_osm(bbox, cache)
    elevation = None
    if request.get("terrain") and not demo:
        progress(20, "terrain", "Sampling public terrain elevation tiles")
        try:
            elevation = fetch_elevation(bbox, cache)
        except Exception as exc:
            warnings.append(f"Terrain service unavailable; explicit flat-ground fallback used: {str(exc)[:160]}")
    if demo:
        warnings.append("Synthetic demo: these buildings and streets are invented, not a reconstruction of the selected coordinates.")
    progress(30, "model", "Building semantic 3D geometry and applying nearby photo materials")
    scene = build_scene(osm, bbox, request.get("meters_per_block", 1), request.get("photos", []), elevation)
    scene["name"] = request.get("name", "RealityForge")
    scene["source"] = "demo" if demo else "live"
    scene["data_attribution"] = osm["realityforge_source"]
    scene.setdefault("limitations", []).extend(warnings)
    if not osm["elements"]:
        scene["limitations"].append("OpenStreetMap returned no mapped objects for this area; only terrain was generated.")
    (output / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    export_obj(scene, output / "scene.obj")
    # Persist and reload the 3D model description: voxel generation consumes this model.
    model = json.loads((output / "scene.json").read_text(encoding="utf-8"))
    bounds = model["bounds"]
    if bounds["width"] * bounds["depth"] * bounds["height"] > 100_000_000:
        raise ValueError("This detailed area exceeds the 100-million-voxel memory budget. Select 2 or 4 metres per block, or reduce the bounding box.")
    progress(48, "voxelize", "Converting the semantic 3D model into Minecraft blocks")
    blocks, palette = scene_to_voxels(model)
    progress(65, "schematic", "Writing a Sponge v2 schematic")
    metadata = {"name": scene["name"], "bbox": bbox, "meters_per_block": request.get("meters_per_block", 1),
                "source": scene["data_attribution"], "pipeline": "semantic 3D scene -> Sponge schematic -> Java Anvil world"}
    write_schematic(output / "realityforge.schem", blocks, palette, metadata)
    non_air = int((blocks != palette.index("minecraft:air")).sum())
    del blocks
    progress(76, "world", "Reading the schematic and writing Minecraft Java region files")
    world = write_java_world(output / "realityforge.schem", output / "RealityForge", scene["name"])
    report = {
        "name": scene["name"], "edition": "Minecraft Java", "target_version": "1.20.4", "data_version": 3700,
        "pipeline": metadata["pipeline"], "bbox": bbox, "meters_per_block": request.get("meters_per_block", 1),
        "source": scene["data_attribution"], "terrain": scene["terrain"].get("source", "Flat approximation"),
        "bounds": scene["bounds"], "statistics": {**scene.get("stats", {}), "non_air_blocks": non_air, "palette_size": len(palette)},
        "photo_count": len(request.get("photos", [])), "limitations": scene["limitations"],
        "attribution": ["RealityForge: original implementation inspired by Arnis (https://github.com/louis-e/arnis).",
                        "Live map data © OpenStreetMap contributors, ODbL: https://www.openstreetmap.org/copyright",
                        "Terrain: Mapzen/AWS Terrain Tiles. Source dataset attribution: https://github.com/tilezen/joerd/blob/master/docs/attribution.md"],
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "import": "Unzip RealityForge-world.zip and copy the enclosed RealityForge folder into your Minecraft Java saves directory. Open in Java 1.20.4. Newer Java versions may upgrade the save; keep a backup."
    }
    report_text = json.dumps(report, indent=2)
    (output / "report.json").write_text(report_text, encoding="utf-8")
    (world / "REALITYFORGE_REPORT.json").write_text(report_text, encoding="utf-8")
    (world / "README.txt").write_text(report["import"] + "\n\n" + "\n".join(report["attribution"]) + "\n\n" + "\n".join(report["limitations"]), encoding="utf-8")
    progress(94, "package", "Packaging the Java world and generation report")
    shutil.make_archive(str(output / "RealityForge-world"), "zip", root_dir=output, base_dir=world.name)
    return {"scene": scene, "stats": report["statistics"], "report": report}
