# RealityForge hackathon demo

Before presenting: run `start.bat`, open http://127.0.0.1:8000, generate the offline demo, and import the resulting world into Minecraft Java 1.20.4. Keep that save and export ZIP ready in case venue Wi-Fi fails. Test it in your Minecraft client before the presentation.

1. **The idea (15 seconds):** “RealityForge combines real map geometry with photo-guided materials to turn a chosen neighbourhood into a Minecraft Java world.”
2. **Location (20 seconds):** click the map, show 250 m / 500 m / 1 km coverage, or paste a Google Maps coordinate URL. Explain that no-photo mode still works from OpenStreetMap.
3. **Photos (25 seconds):** add your own street photo. Show the suggested palette and style override. Explain that nearby buildings take that material, while unseen geometry comes from geographic data and procedural assets.
4. **Pipeline (30 seconds):** generate the offline fixture for dependable timing, explicitly calling it synthetic. Show model → schematic → Java world progress. Orbit the preview.
5. **Result (30 seconds):** download the `.schem` and world ZIP; switch to the prepared Minecraft save and fly along a street past houses and trees.

Use accurate language: “photo-guided semantic reconstruction prototype,” not “we recover an entire city from one panorama.” Explain that the prototype uses hand-built assets and colour heuristics; learned photo geometry is future work.

If live services fail, the app displays the failure. Choose Offline demo explicitly. The local demo and bundled 3D viewer need no network after installation; map tiles may be unavailable offline, but generation still works.

Judges may ask:

- **Why a semantic scene?** Streets, buildings and trees retain object types before voxel conversion, so they become meaningful Minecraft assets.
- **Why model → schematic → world?** The model is inspectable and reusable; the schematic is portable; the world is ready to explore.
- **What do photos change today?** Local material choices, with manual correction. They do not currently recover geometry or guarantee facade matching.
- **How is it different from Arnis?** A separate Python implementation centered on photo-guided local dressing and an explicit, downloadable 3D intermediate stage.
- **What is next?** Segmentation, camera pose/depth estimation, multi-view alignment and a broader semantic asset library.
