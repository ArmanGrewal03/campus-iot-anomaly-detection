"use strict";
const fs = require("fs");
const path = require("path");
const topojson = require("topojson-client");
const d3geo = require("d3-geo");

const land110m = JSON.parse(
  fs.readFileSync(require.resolve("world-atlas/land-110m.json"), "utf8")
);
const land = topojson.feature(land110m, land110m.objects.land);

const TARGET_POINTS = 14_000;
const points = [];

if (!land.features || land.features.length === 0) {
  console.error("No features in land GeoJSON");
  process.exit(1);
}

const totalArea = land.features.reduce((s, f) => s + d3geo.geoArea(f), 0);

for (const feature of land.features) {
  const area = d3geo.geoArea(feature);
  const n = Math.max(1, Math.round((area / totalArea) * TARGET_POINTS));
  const bbox = d3geo.geoBounds(feature);
  const [minLon, minLat] = bbox[0];
  const [maxLon, maxLat] = bbox[1];
  let added = 0;
  let attempts = 0;
  const maxAttempts = n * 50;
  while (added < n && attempts < maxAttempts) {
    const lon = minLon + Math.random() * (maxLon - minLon);
    const lat = minLat + Math.random() * (maxLat - minLat);
    if (d3geo.geoContains(feature, [lon, lat])) {
      points.push([lat, lon]);
      added++;
    }
    attempts++;
  }
}

const outPath = path.join(__dirname, "..", "src", "data", "globeLandPoints.json");
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(points), "utf8");
console.log(`Wrote ${points.length} land points to ${outPath}`);
