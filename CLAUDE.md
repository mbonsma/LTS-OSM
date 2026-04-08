# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Calculates cycling [Level of Traffic Stress (LTS)](https://peterfurth.sites.northeastern.edu/level-of-traffic-stress/) for road/path segments in a region using OpenStreetMap data. LTS values (1–4) represent how stressful a segment is for cyclists, based on Furth (2016). Adapted from [Bike Ottawa's stressmodel](https://github.com/BikeOttawa/stressmodel) and [mbonsma's LTS-OSM](https://github.com/mbonsma/LTS-OSM).

## Setup

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The `OVERPASS_API_URL` and `LTS_OUTPUT_DIR` environment variables can override defaults (`http://overpass-api.de/api/interpreter` and `output/`, respectively).

## Common commands

```shell
# Run LTS calculation for one or more areas defined in a query JSON file
python lts_osm/lts_osm.py --query-json-file query/gta.json

# Re-run LTS calculation using already-downloaded XML files (skips Overpass download)
# Pass the areas_xml_file_path.json written to the run directory by a prior run
python lts_osm/lts_osm.py --downloaded-xml-json-map output/runs/<timestamp>/areas_xml_file_path.json

# Re-run using a single previously downloaded OSM XML file
python lts_osm/lts_osm.py --osm-file output/runs/<timestamp>/xmls/toronto.xml

# Plot results
python lts_osm/lts_plot.py --lts-csv-file output/runs/<timestamp>/lts_csv/all_lts_toronto.csv --gdf-nodes-file output/runs/<timestamp>/lts_csv/gdf_nodes_toronto.csv --city "Toronto"

# Run tests
pytest lts_osm/lts_functions.py
```

## Architecture

### Data flow

1. **Query JSON file** (`query/*.json`) — JSON file listing one or more areas by name and wikidata ID (e.g. `query/gta.json`). The Overpass QL query itself is built at runtime from `query/query_template.overpass`.
2. **`lts_osm/lts_osm.py`** — Entry point. Downloads OSM XML per area via Overpass API → builds an `osmnx` graphml per area → converts to GeoPandas GeoDataFrames (`gdf_nodes`, `gdf_edges`) → applies LTS classification pipeline → writes outputs. All outputs for a run go into a timestamped directory `output/runs/<timestamp>/`.
3. **`lts_osm/lts_functions.py`** — All LTS classification logic. Pure functions operating on GeoDataFrame slices.
4. **`lts_osm/lts_plot.py`** — Reads the CSV outputs from step 2 and produces PDF/PNG maps.
5. **`lts_osm/isochrone.py`** — Standalone notebook-style script for isochrone analysis from a point on the LTS graph.

### LTS classification pipeline (in `lts_osm.py` → `lts_functions.py`)

Each stage splits edges and assigns a `rule` code (e.g. `p2`, `s3`, `b1`, `m5`); the rule maps to an LTS integer:

1. `biking_permitted` — removes ways where cycling is banned (rule prefix `p`); remainder proceeds.
2. `is_separated_path` — detects cycleways/paths/tracks (prefix `s`); assigned LTS 1.
3. `is_bike_lane` — detects bike lanes (cycleway tags); remainder goes to mixed traffic.
4. `parking_present` — splits bike-lane edges by parking presence.
5. `bike_lane_analysis_with_parking` / `bike_lane_analysis_no_parking` — assign LTS 1–4 based on speed, lanes, width (prefix `b`/`c`).
6. `mixed_traffic` — assigns LTS 1–4 based on highway type, speed, lanes (prefix `m`).
7. Node LTS — assigned in `lts_osm.py` as max of intersecting edge LTS (vectorized), reduced by traffic signals or stop signs.

### OSM tag parsing

`get_lanes` and `get_max_speed` (in `lts_functions.py`) normalize messy OSM values before comparisons:
- Lanes: handles `None`, lists, delimited strings (`"2;1"`, `"2|1"`), decimals — takes `ceil(max(...))` to stay conservative.
- Speed: handles lists and unparseable strings — defaults to 50 km/h (`local` parameter).
- Speed defaults by highway type: `national`→40, `motorway`→100, `primary`/`secondary`→80, otherwise→50.

### Outputs (written to `output/runs/<timestamp>/`)

Each run creates a timestamped directory. Within it:

| Path | Content |
|------|---------|
| `xmls/<area>.xml` | Raw Overpass XML download per area |
| `areas_xml_file_path.json` | Manifest mapping area names → XML paths (pass to `--downloaded-xml-json-map` to skip re-downloading) |
| `graphml/<area>.graphml` | osmnx graph (pre-LTS) per area |
| `lts_csv/all_lts_<area>.csv` | Edge LTS with rule codes |
| `lts_csv/gdf_nodes_<area>.csv` | Node LTS |
| `lts_geojson/all_lts_<area>.geojson` | Edge LTS (filtered LTS 1–4) |
| `lts_geojson/gdf_nodes_<area>.geojson` | Node LTS (filtered LTS 1–4) |
| `lts_geojson/combined/all_lts_combined.geojson` | Combined edges across all areas (multi-area runs only) |
| `lts_graphml/<area>_lts.graphml` | osmnx graph with LTS attributes |
| `lts_outputs.json` | Manifest of all output file paths for the run |

If `graphml/<area>.graphml` already exists in the run directory, the osmnx graph build step is skipped and it is loaded directly.

## Converting output to PMTiles (for web rendering)

```shell
tippecanoe -l lts_toronto_filtered_1_4 -n "Level of Traffic Stress" \
  --no-feature-limit --extend-zooms-if-still-dropping \
  --coalesce-densest-as-needed --maximum-tile-bytes=2000000 \
  -P -zg -D12 -o output.mbtiles input.geojson

pmtiles convert output.mbtiles output.pmtiles
```
