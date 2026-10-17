# Optional tasks to paste into Claude Pro

These tasks are independent of the application implementation. Return Claude's Markdown here for review and integration. Do not send private photos or account credentials. No source files are needed for these two tasks.

## Task 1: practical reconstruction roadmap

```text
I’m building RealityForge for a hackathon. It is a local Python + browser JavaScript app, with no Rust, targeting Minecraft Java 1.20.4.

The implemented MVP pipeline is:
OpenStreetMap bounding box + optional terrain elevation + phone photos/panoramas
→ semantic 3D scene and OBJ
→ Minecraft voxel volume
→ Sponge v2 .schem
→ Java Anvil world.

Photos currently guide nearby building materials using colour heuristics and manual style choices. Buildings and trees use hand-built procedural assets. We do NOT yet perform photogrammetry, depth estimation, segmentation or camera-pose recovery.

Write a practical ROADMAP.md for improving photo-based reconstruction after the hackathon. Compare a single panorama with multiple overlapping photos. Recommend concrete stages for segmentation, depth, camera alignment, semantic asset selection and evaluation. Clearly distinguish achievable results from guesses, discuss hidden geometry and scale, and avoid proposing Google imagery extraction without appropriate licensing. Prioritize a solo beginner developer. Limit to 900 words. Do not write application code or claim these features already exist.
```

## Task 2: pitch and judge questions

```text
Write a truthful 90-second hackathon pitch and answers to 6 likely judge questions for RealityForge.

RealityForge is a Python + JavaScript prototype inspired by Arnis. Users select a real-world bounding box, optionally upload phone photos/panoramas, and export a Minecraft Java world. Its pipeline is geographic data → semantic 3D model → schematic → Java world. OpenStreetMap supplies geometry; optional public elevation supplies terrain; photo colour heuristics and manual choices guide nearby materials. Houses and trees use procedural assets. There is also a clearly labelled synthetic offline demo.

The distinction is photo-guided local dressing plus an inspectable 3D intermediate model. Do not claim AI geometry reconstruction, photorealism, exact landmarks, Google Street View import, or proven in-game compatibility. Explain the current limitations confidently and identify multi-view reconstruction as future work. Deliver plain Markdown, no application code, under 700 words.
```
