# Validation record

Validated on Windows with Python 3.11 on 17 October 2026.

- `python -m pytest -q`: 12 tests passed. One upstream Starlette TestClient deprecation warning remains; runtime generation is unaffected.
- `node --check web/app.js`: passed.
- Tests decode Sponge varints, palettes and dimensions; parse Anvil chunks across chunk and region boundaries; verify signed/padded block-state arrays, heightmaps, spawn selection and schematic readback.
- Pipeline tests verify semantic geometry, image-derived material influence bounds, terrain, OBJ/MTL export, the actual schematic → world conversion, ZIP contents and provenance.
- HTTP integration tests upload an image, generate a world, download every artifact, reject malformed requests and cross-origin writes, and install into a temporary saves directory. A second install is rejected without overwriting the first.
- Browser QA: the local interface rendered its map and WebGL scene; Offline demo generation reached 100% and displayed the world, schematic, OBJ, MTL, scene and report links. The 3D camera was adjusted to frame narrow viewports.
- Live geographic generation succeeded for a roughly 242 × 245 m area near Toronto City Hall, with public terrain elevations. Both Overpass endpoints failed at test time; the small-area OSM API fallback succeeded. This is recorded in the export report.
- The offline 250 m fixture exported about 370,000 non-air blocks. The live Toronto sample exported 542,565 non-air blocks after final road clipping refinements.
- A 1 km synthetic selection at 2 metres/block completed the full export pipeline in approximately 3 seconds on this computer. Live dense areas can be substantially slower.

Target metadata: Minecraft Java 1.20.4, DataVersion 3700, verified against Mojang's release metadata. Generated files are structurally validated. **No Minecraft client or server was launched to verify in-game loading.** Open an exported world in Java 1.20.4 before the presentation; preserve its ZIP as a backup. No EULA was accepted and no existing Minecraft save was changed during testing.

Larger selections are bounded to 1 km per side and a 100-million-voxel memory budget. Use 2 or 4 metres/block for tall or dense 1 km scenes. Public service availability and performance on other computers are not guaranteed by these checks.
