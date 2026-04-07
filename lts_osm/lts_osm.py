"""Calculate Level of Traffic Stress maps with Open Street Map"""
import argparse
import json
import logging
import numpy as np
import os
import osmnx as ox
import pandas as pd
import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from string import Template
from typing import Any

# import lts calculation functions
from lts_functions import (biking_permitted, is_separated_path, is_bike_lane, parking_present,
                           bike_lane_analysis_no_parking, bike_lane_analysis_with_parking, mixed_traffic)

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)

OVERPASS_API_URL = os.environ.get("OVERPASS_API_URL", "http://overpass-api.de/api/interpreter")

OUTPUT_DIR = os.environ.get("LTS_OUTPUT_DIR", "output")
OSM_FILES_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "osm")
LTS_FILES_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "lts")
OVERPASS_QUERY_TEMPLATE = os.path.join("query/query_template.overpass", "")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OSM_FILES_OUTPUT_DIR, exist_ok=True)
os.makedirs(LTS_FILES_OUTPUT_DIR, exist_ok=True)

def build_run_dir_name(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"{now:%Y%m%dT%H%M%S}"

def create_run_directory(args: argparse.Namespace, now: datetime | None = None) -> Path:
    output_root = Path(OUTPUT_DIR) 
    output_root = output_root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / build_run_dir_name(now=now)
    run_dir.mkdir()
    return run_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculates the Level of Traffic Stress from Open Street Map data."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run",
    )
    osm_file_group = parser.add_mutually_exclusive_group(required=True)
    osm_file_group.add_argument(
        "--query-json-file",
        type=str,
        help="Path to query json file indicating which area to download from osm",
    )
    osm_file_group.add_argument(
        "--osm-file",
        type=str,
        help="Path to the downloaded osm file to calculate the level of stress for",
    )
    osm_file_group.add_argument(
        "--downloaded-xml-json-map",
        type=str,
        help="Path to a json file map of areas and it's downloaded xml path",
    )
    return parser.parse_args()

def _make_overpass_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist={429, 502, 503, 504},
        respect_retry_after_header=True,
        allowed_methods={"POST"},
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def build_overpass_query(area_id: str) -> str:
    template_path = Path(OVERPASS_QUERY_TEMPLATE)
    template_text = template_path.read_text()
    query = Template(template_text).substitute(key="wikidata", value=area_id)
    return query

def write_json_to_run_dir(
    run_dir: Path,
    filename: str,
    payload: Any,
    *,
    log_label: str | None = None,
) -> Path:
    output_path = run_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved %s to %s", log_label or filename, output_path)
    return output_path

def download_osm_data_from_overpass_api(
    query_str: str, output_file_path: str, session: requests.Session
) -> str:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        with session.post(
            OVERPASS_API_URL,
            data=query_str,
            timeout=300,
            stream=True,
        ) as overpass_request:
            overpass_request.raise_for_status()
            total = int(overpass_request.headers.get("Content-Length", 0))
            task = progress.add_task("Requesting Overpass data ...", total=total if total > 0 else None)
            with open(output_file_path, "wb") as output_file:
                for chunk in overpass_request.iter_content(chunk_size=1024 * 128):
                    if not chunk:
                        continue
                    output_file.write(chunk)
                    if total > 0:
                        progress.update(task, advance=len(chunk))
    logger.info(f"Downloaded osm data to: {output_file_path}")
    return output_file_path

def generate_graphml(run_dir: Path, area_name: str, osm_data_xml_path: str) -> str:
    graphml_dir =  run_dir / "graphml"
    graphml_dir.mkdir(parents=True, exist_ok=True)

     # extract all unique way tag keys so osmnx retains them when parsing the graph
    osm_xml_tree = ET.parse(osm_data_xml_path)
    osm_xml_root = osm_xml_tree.getroot()
    if osm_xml_root.tag != 'osm':
        os.remove(osm_data_xml_path)
        logger.error("Downloaded file is not valid OSM XML (root tag: <%s>). File deleted, please retry.", osm_xml_root.tag)
    remark = osm_xml_root.find('remark')
    if remark is not None and remark.text:
        logger.warning("Overpass API remark: %s", remark.text.strip())
    logger.info("Total osm elements: %s", len(osm_xml_root))

    # dataframe of tags
    dfs_tags = []

    for way in osm_xml_root.findall('way'):
        tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
        if not tags:
            continue
        if "name" in tags:
            logger.debug("Adding road %s with tags: %s", tags["name"], tags)
        df = pd.DataFrame.from_dict(tags, orient='index')
        dfs_tags.append(df)

    tags_df = pd.concat(dfs_tags).reset_index()
    tags_df.columns = ["tag", "tagvalue"]
    #logger.info("Tags dataframe: %s", tags_df)

    tag_counts = tags_df['tag'].value_counts().reset_index() # count all the unique tags
    way_tags = list(tag_counts['tag']) # all unique tags from the OSM download

    # add the above list to the global osmnx settings
    ox.settings.useful_tags_way += way_tags
    ox.settings.osm_xml_way_tags = way_tags

    # build graph from the downloaded OSM XML, or load cached graphml
    area_graphml_filepath = os.path.join(graphml_dir, f"{area_name}.graphml")

    if os.path.exists(area_graphml_filepath):
        logger.info("Loading saved graph %s", area_graphml_filepath)
        area_graphml = ox.load_graphml(area_graphml_filepath)
    else:
        logger.info("Building graph from %s", osm_data_xml_path)
        area_graphml = ox.graph_from_xml(osm_data_xml_path, retain_all=True, simplify=False)
        logger.info("Saving graph %s", area_graphml_filepath)
        ox.save_graphml(area_graphml, area_graphml_filepath)
    return area_graphml_filepath

def calculate_lts(run_dir: Path, area_name: str, area_graphml_path: str) -> dict[str, Any]:
    geojson_dir = run_dir / "lts_geojson"
    geojson_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = run_dir / "lts_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    graphml_dir = run_dir / "lts_graphml"
    graphml_dir.mkdir(parents=True, exist_ok=True)

    area_graphml = ox.load_graphml(area_graphml_path)
    # plot downloaded graph - this is slow for a large area
    #fig, ax = ox.plot_graph(city_graphml, node_size=0, edge_color="w", edge_linewidth=0.2)

    # ## Analyze LTS
    #
    # Start with is biking allowed, get edges where biking is not *not* allowed.

    # convert graph to node and edge GeoPandas GeoDataFrames
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(area_graphml)
    logger.info(f"Gdf edges shape: {gdf_edges.shape}", )
    gdf_allowed, gdf_not_allowed = biking_permitted(gdf_edges)
    logger.info(f"Gdf allowed shape: {gdf_allowed.shape}")
    logger.info(f"Gdf not allowed shape: {gdf_not_allowed.shape}")

    # check for separated path
    separated_edges, unseparated_edges = is_separated_path(gdf_allowed)
    #assign separated ways lts = 1
    separated_edges['lts'] = 1
    logger.info(f"Separated edges shape: {separated_edges.shape}", )
    logger.info(f"Unseparated edges shape: {unseparated_edges.shape}")

    to_analyze, no_lane = is_bike_lane(unseparated_edges)
    logger.info(f"To analyze shape: {to_analyze.shape}")
    logger.info(f"No lane shape: {no_lane.shape}")

    parking_detected, parking_not_detected = parking_present(to_analyze)
    logger.info(f"Parking detected {parking_detected.shape}")
    logger.info(f"Parking not detected {parking_not_detected.shape}")

    parking_lts = bike_lane_analysis_with_parking(parking_detected)

    no_parking_lts = bike_lane_analysis_no_parking(parking_not_detected)

    # Next, go to the last step - mixed traffic
    lts_no_lane = mixed_traffic(no_lane)

    # final components: lts_no_lane, parking_lts, no_parking_lts, separated_edges
    # these should all add up to gdf_allowed
    logger.info(f"Gdf allowed {gdf_allowed.shape}")
    lts_no_lane.shape[0] + parking_lts.shape[0] + no_parking_lts.shape[0] + separated_edges.shape[0]

    gdf_not_allowed['lts'] = 0

    all_lts = pd.concat([separated_edges, parking_lts, no_parking_lts, lts_no_lane, gdf_not_allowed])

    # decision rule glossary
    # these are from Bike Ottawa's stressmodel code
    rule_message_dict = {'p2':'Cycling not permitted due to bicycle=\'no\' tag.',
                        'p6':'Cycling not permitted due to access=\'no\' tag.',
                        'p3':'Cycling not permitted due to highway=\'motorway\' tag.',
                        'p4':'Cycling not permitted due to highway=\'motorway_link\' tag.',
                        'p7':'Cycling not permitted due to highway=\'proposed\' tag.',
                        'p5':'Cycling not permitted. When footway="sidewalk" is present, there must be a bicycle="yes" when the highway is "footway" or "path".',
                        's3':'This way is a separated path because highway=\'cycleway\'.',
                        's1':'This way is a separated path because highway=\'path\'.',
                        's2':'This way is a separated path because highway=\'footway\' but it is not a crossing.',
                        's7':'This way is a separated path because cycleway* is defined as \'track\'.',
                        's8':'This way is a separated path because cycleway* is defined as \'opposite_track\'.',
                        'b1':'LTS is 1 because there is parking present, the maxspeed is less than or equal to 40, highway="residential", and there are 2 lanes or less.',
                        'b2':'Increasing LTS to 3 because there are 3 or more lanes and parking present.',
                        'b3':'Increasing LTS to 3 because the bike lane width is less than 4.1m and parking present.',
                        'b4':'Increasing LTS to 2 because the bike lane width is less than 4.25m and parking present.',
                        'b5':'Increasing LTS to 2 because the bike lane width is less than 4.5m, maxspeed is less than 40 on a residential street and parking present.',
                        'b6':'Increasing LTS to 2 because the maxspeed is between 41-50 km/h and parking present.',
                        'b7':'Increasing LTS to 3 because the maxspeed is between 51-54 km/h and parking present.',
                        'b8':'Increasing LTS to 4 because the maxspeed is over 55 km/h and parking present.',
                        'b9':'Increasing LTS to 3 because highway is not \'residential\'.',
                        'c1':'LTS is 1 because there is no parking, maxspeed is less than or equal to 50, highway=\'residential\', and there are 2 lanes or less.',
                        'c3':'Increasing LTS to 3 because there are 3 or more lanes and no parking.',
                        'c4':'Increasing LTS to 2 because the bike lane width is less than 1.7 metres and no parking.',
                        'c5':'Increasing LTS to 3 because the maxspeed is between 51-64 km/h and no parking.',
                        'c6':'Increasing LTS to 4 because the maxspeed is over 65 km/h and no parking.',
                        'c7':'Increasing LTS to 3 because highway with bike lane is not \'residential\' and no parking.',
                        'm17':'Setting LTS to 1 because motor_vehicle=\'no\'.',
                        'm13':'Setting LTS to 1 because highway=\'pedestrian\'.',
                        'm14':'Setting LTS to 2 because highway=\'footway\' and footway=\'crossing\'.',
                        'm2':'Setting LTS to 1 because highway=\'service\' and service=\'alley\'.',
                        'm15':'Setting LTS to 2 because highway=\'track\'.',
                        'm3':'Setting LTS to 2 because maxspeed is 50 km/h or less and service is \'parking_aisle\'.',
                        'm4':'Setting LTS to 2 because maxspeed is 50 km/h or less and service is \'driveway\'.',
                        'm16':'Setting LTS to 2 because maxspeed is less than 35 km/h and highway=\'service\'.',
                        'm5':'Setting LTS to 1 because maxspeed is up to 40 km/h, 3 or fewer lanes and highway=\'residential\'.',
                        'm6':'Setting LTS to 3 because maxspeed is up to 40 km/h and 3 or fewer lanes on non-residential highway.',
                        'm7':'Setting LTS to 3 because maxspeed is up to 40 km/h and 4 or 5 lanes.',
                        'm8':'Setting LTS to 4 because maxspeed is up to 40 km/h and the number of lanes is greater than 5.',
                        'm9':'Setting LTS to 2 because maxspeed is up to 50 km/h and lanes are 2 or less and highway=\'residential\'.',
                        'm10':'Setting LTS to 3 because maxspeed is up to 50 km/h and lanes are 3 or less on non-residential highway.',
                        'm11':'Setting LTS to 4 because the number of lanes is greater than 3.',
                        'm12':'Setting LTS to 4 because maxspeed is greater than 50 km/h.'}

    simplified_message_dict = {'p2':r'bicycle $=$ "no"',
                     'p6':r'access $=$ "no"',
                     'p3':r'highway $=$ "motorway"',
                     'p4':r'highway $=$ "motorway_link"',
                     'p7':r'highway $=$ "proposed"',
                     'p5':r'footway $=$ "sidewalk", bicycle$\neq$"yes"',
                     's3':r'highway $=$ "cycleway"',
                     's1':r'highway $=$" path"',
                     's2':r'separated, highway $=$" footway", not a crossing',
                     's7':r'cycleway* $=$ "track"',
                     's8':r'cycleway* $=$ "opposite_track"',
                     'b1':r'bike lane w/ parking, $\leq$ 40 km/h, highway $=$ "residential", $\leq$ 2 lanes',
                     'b2':r'bike lane w/ parking, 3 or more lanes',
                     'b3':r'bike lane width $<$ 4.1m, parking',
                     'b4':r'bike lane width $<$ 4.25m, parking',
                     'b5':r'bike lane width $<$ 4.5m, $\leq$ 40 km/h, residential, parking',
                     'b6':r'bike lane w/ parking, speed 41-50 km/h',
                     'b7':r'bike lane w/ parking, speed 51-54 km/h',
                     'b8':r'bike lane w/ parking, speed $>$ 55 km/h',
                     'b9':r'bike lane w/ parking, highway $\neq$ "residential"',
                     'c1':r'bike lane no parking, $\leq$ 50 km/h, highway $=$ "residential", $\leq$ 2 lanes',
                     'c3':r'bike lane no parking, $\leq$ 65 km/h, $\geq$ 3 lanes',
                     'c4':r'bike lane width $<$ 1.7m, no parking',
                     'c5':r'bike lane no parking, speed 51-64 km/h',
                     'c6':r'bike lane no parking, speed $>$ 65 km/h',
                     'c7':r'bike lane no parking, highway $\neq$ "residential"',
                     'm17':r'mixed traffic, motor_vehicle $=$ "no"',
                     'm13':r'mixed traffic, highway $=$ "pedestrian"',
                     'm14':r'mixed traffic, highway $=$ "footway", footway $=$ "crossing"',
                     'm2':r'mixed traffic, highway $=$ "service", service $=$ "alley"',
                     'm15':r'mixed traffic, highway $=$ "track"',
                     'm3':r'mixed traffic, speed $\leq$ 50 km/h, service $=$ "parking_aisle"',
                     'm4':r'mixed traffic, speed $\leq$ 50 km/h, service $=$ "driveway"',
                     'm16':r'mixed traffic, speed $\leq$ 35 km/h, highway $=$ "service"',
                     'm5':r'mixed traffic, speed $\leq$ 40 km/h, highway $=$ "residential", $\leq$ 3 lanes',
                     'm6':r'mixed traffic, speed $\leq$ 40 km/h, highway $\neq$ "residential", $\leq$ 3 lanes',
                     'm7':r'mixed traffic, speed $\leq$ 40 km/h, 4 or 5 lanes',
                     'm8':r'mixed traffic, speed $\leq$ 40 km/h, lanes $>$ 5',
                     'm9':r'mixed traffic, speed $\leq$ 50 km/h, highway $=$ "residential",$\leq$ 2 lanes',
                     'm10':r'mixed traffic, speed $\leq$ 50 km/h, highway $\neq$ "residential", $\leq$ 3 lanes',
                     'm11':r'mixed traffic, speed $\leq$ 50 km/h, lanes $>$ 3',
                     'm12':r'mixed traffic, speed $>$ 50 km/h'}

    all_lts['message'] = all_lts['rule'].map(rule_message_dict)
    all_lts['short_message'] = all_lts['rule'].map(simplified_message_dict)

    # ## Node LTS
    #
    # Calculate node LTS.
    #
    # - An intersection without either was assigned the highest LTS of its intersecting roads.
    # - Stop signs reduced an otherwise LTS2 intersection to LTS1.
    # - A signalized intersection of two lowstress links was assigned LTS1.
    # - Assigned LTS2 to signalized intersections where a low-stress (LTS1/ 2) link crosses a high-stress (LTS3/4) link.

    gdf_nodes['highway'].value_counts()
    gdf_nodes['lts'] = np.nan # make lts column
    gdf_nodes["message"] = ""  # make message column

    for node in tqdm(gdf_nodes.index):
        try:
            edges = all_lts.loc[node]
        except:
            #print("Node not found in edges: %s" %node)
            gdf_nodes.loc[node, 'message'] = "Node not found in edges"
            continue
        control = gdf_nodes.loc[node,'highway'] # if there is a traffic control
        max_lts = edges['lts'].max()
        node_lts = int(max_lts) # set to max of intersecting roads
        message = "Node LTS is max intersecting LTS"
        if node_lts > 2:
            if control == 'traffic_signals':
                node_lts = 2
                message = "LTS 3-4 with traffic signals"
        elif node_lts <= 2:
            if control == 'traffic_signals' or control == 'stop':
                node_lts = 1
                message = "LTS 1-2 with traffic signals or stop"

        gdf_nodes.loc[node,'message'] = message
        gdf_nodes.loc[node,'lts'] = node_lts # assign node lts

    gdf_nodes_filtered = gdf_nodes[gdf_nodes['lts'].between(1, 4, inclusive='both')]

    # Save data for plotting
    gdf_nodes_csv_file_path = os.path.join(csv_dir, f"gdf_nodes_{area_name}.csv")
    logger.info(f"Saving lts nodes csv to: {gdf_nodes_csv_file_path}")
    gdf_nodes_filtered.to_csv(gdf_nodes_csv_file_path, index=True)

    # Save GeoJSON
    gdf_nodes_geojson_file_path = os.path.join(geojson_dir, f"gdf_nodes_{area_name}.geojson")
    logger.info(f"Saving lts nodes to: {gdf_nodes_geojson_file_path}")
    gdf_nodes_filtered.to_file(gdf_nodes_geojson_file_path, driver="GeoJSON")

    all_lts_small = all_lts[['osmid', 'lanes', 'name', 'highway', 'maxspeed', 'geometry', 'length', 'rule', 'lts',
                            'lanes_assumed', 'maxspeed_assumed', 'message', 'short_message']]

    # Filter to keep lts only between 1-4
    all_lts_small_filtered = all_lts_small[all_lts_small['lts'].between(1, 4)]

    all_lts_csv_file_path = os.path.join(csv_dir, f"all_lts_{area_name}.csv")
    logger.info(f"Saving all lts csv to: {all_lts_csv_file_path}")
    all_lts_small_filtered.to_csv(all_lts_csv_file_path, index=True)

    all_lts_geojson_file_path = os.path.join(geojson_dir, f"all_lts_{area_name}.geojson")
    logger.info(f"Saving all lts geojson to: {all_lts_geojson_file_path}")
    all_lts_small_filtered.to_file(all_lts_geojson_file_path, driver="GeoJSON")

    # make graph with LTS information
    all_lts_graphml = ox.graph_from_gdfs(gdf_nodes, all_lts_small_filtered)

    # save LTS graph
    all_lts_graphml_filepath = os.path.join(graphml_dir, f"{area_name}_lts.graphml")
    logger.info(f"Saving city lts graphml to: {all_lts_graphml_filepath}")
    ox.save_graphml(all_lts_graphml, all_lts_graphml_filepath)

    lts_outputs = {"area_name": area_name}
    lts_outputs["lts_csv"] = all_lts_csv_file_path
    lts_outputs["lts_geojson"] = all_lts_geojson_file_path
    lts_outputs["lts_graphml"] = all_lts_graphml_filepath
    lts_outputs["gdf_nodes_geojson"] = gdf_nodes_geojson_file_path

    return lts_outputs

def main(args: argparse.Namespace) -> int:
    run_dir = create_run_directory(args)
    query_file = args.query_json_file
    areas_processed_dict = {}

    if args.osm_file:
        if not os.path.isfile(args.osm_file):
            logger.error("Invalid --osm-file path: %a", args.osm_file)
            return 2
        osm_data_xml_path = args.osm_file
        logger.info("Using existing osm file %s", osm_data_xml_path)
    elif args.downloaded_xml_json_map:
        if not os.path.isfile(args.downloaded_xml_json_map):
            logger.error("Invalid --downloaded-xml-json-map: %a", args.downloaded_xml_json_map)
            return 2
        areas_processed_dict = json.loads(Path(args.downloaded_xml_json_map).read_text("utf-8"))
    elif args.query_json_file:
        if not os.path.isfile(args.query_json_file):
            logger.error("Invalid --query-json-file path: %a", args.query_json_file)
            return 2
        logger.info("Using query file %s to download osm data from overpass api", args.query_json_file)
        query_json = json.loads(Path(query_file).read_text(encoding="utf-8"))
        logger.info(f"Number of areas to download: {len(query_json.get("areas"))}")
        areas_xml_list = []
        xmls_download_dir = run_dir / "xmls"
        xmls_download_dir.mkdir()
        session = _make_overpass_session()
        for area in query_json.get("areas"):
            area_dict = {}
            logger.info(f"Processing area: {area.get("name")}, Aread Id: {area.get("wikidata_id")}")
            overpass_query = build_overpass_query(area.get("wikidata_id"))
            logger.info(f"Downloading osm data using overpass query: {overpass_query}")
            osm_data_xml_path = download_osm_data_from_overpass_api(
                overpass_query,
                os.path.join(xmls_download_dir, f"{area.get("name")}.xml"),
                session,
            )
            area_dict["name"] = area.get("name")
            area_dict["xml_file_path"] = osm_data_xml_path
            areas_xml_list.append(area_dict)
        areas_processed_dict["areas"] = areas_xml_list
        write_json_to_run_dir(run_dir, "areas_xml_file_path.json", areas_processed_dict)
    areas_lts_outputs = {"areas": []}
    areas_lts_outputs_list =  []
    for processed_area in areas_processed_dict.get("areas"):
        logger.info(f"Processing area {processed_area.get("name")}, XML filepath: {processed_area.get("xml_file_path")}")
        area_graphml_filepath = generate_graphml(run_dir, processed_area.get("name"), processed_area.get("xml_file_path"))
        area_lts_output = calculate_lts(run_dir, processed_area.get("name"), area_graphml_filepath)
        areas_lts_outputs_list.append(area_lts_output)
    areas_lts_outputs["areas"] = areas_lts_outputs_list

    write_json_to_run_dir(run_dir, "lts_outputs.json", areas_lts_outputs)

    if len(areas_lts_outputs.get("areas")) > 1:
        #Combining geojsons
        lts_combined_geojson_dir = run_dir / "lts_geojson" / "combined"
        lts_combined_geojson_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Combining geojsons")
        lts_gdf_list = []
        for lts_area in areas_lts_outputs.get("areas"):
            logger.info(f"Adding lts for {lts_area['area_name']}, gejson: {lts_area['lts_geojson']}")
            lts_gdf_list.append(gpd.read_file(lts_area["lts_geojson"]))
        lts_combined_gdf = pd.concat(lts_gdf_list, ignore_index=True)
        lts_combined_file_path = os.path.join(lts_combined_geojson_dir, "all_lts_combined.geojson")
        lts_combined_gdf.to_file(lts_combined_file_path, driver="GeoJSON")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
