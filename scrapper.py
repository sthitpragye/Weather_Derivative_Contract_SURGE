import requests
import base64
import time
import json
import os

URL = 'https://dash.upag.gov.in/_dash-update-component'
HEADERS = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.5 Safari/605.1.15',
    'Referer': 'https://dash.upag.gov.in/areaproductivitydash?t=&hide=true',
}

# 1. Initialize session to maintain session cookies
session = requests.Session()
session.headers.update(HEADERS)
session.get("https://dash.upag.gov.in/areaproductivitydash?t=&hide=true")

current_dir = os.getcwd()

def get_data_for_year(year, crop_name, metric_name):
    # This payload now dynamically accepts the crop and metric
    data_string = f'''{{
        "output": "..area-proc-dash-tab2-map-download.data..",
        "outputs": [{{"id": "area-proc-dash-tab2-map-download", "property": "data"}}],
        "inputs": [{{"id": "area-proc-dash-tab2-map-download-button", "property": "n_clicks", "value": 31}}],
        "changedPropIds": ["area-proc-dash-tab2-map-download-button.n_clicks"],
        "state": [
            {{"id": "area-proc-dash-year-slider", "property": "value", "value": {year}}},
            {{"id": "area-proc-dash-filters-store", "property": "data", "value": {{"crop":"{crop_name}","season":"Total","metric":"{metric_name}","uom":"('Lakh Ha', 'Lakh Tonnes', 'Kg/Ha')","filter":"By Value","display":["Trend"],"from":"","to":""}}}},
            {{"id": "area-proc-dash-tab2-map", "property": "style", "value": {{"display": "none"}}}},
            {{"id": "area-proc-dash-common-temp-store", "property": "data", "value": {{"statename": "Uttar Pradesh"}}}},
            {{"id": "area-proc-dash-metric-select", "property": "value", "value": "{metric_name}"}},
            {{"id": "area-proc-dash-season-select", "property": "value", "value": "Total"}},
            {{"id": "area-proc-dash-crop-select", "property": "value", "value": "{crop_name}"}},
            {{"id": "area-proc-dash-uom-select", "property": "value", "value": "('Lakh Ha', 'Lakh Tonnes', 'Kg/Ha')"}},
            {{"id": "area-proc-dash-filter-radio", "property": "value", "value": "By Value"}},
            {{"id": "url", "property": "search", "value": "?t=&hide=true"}}
        ]
    }}'''
    
    response = session.post(URL, data=data_string)
    if response.status_code == 200:
        data = response.json()
        raw_data = data['response']['area-proc-dash-tab2-map-download']['data']
        # Handle the dictionary format you just discovered
        if isinstance(raw_data, dict) and 'content' in raw_data:
            return raw_data['content'].encode('utf-8')
    return None

# To download Production
for year in range(1998, 2025):
    print(f"Harvesting Pulses Production: {year}...")
    content = get_data_for_year(year, "Total Pulses", "Production")
    if content:
        with open(f"pulses_prod_{year}.csv", "wb") as f: f.write(content)
    time.sleep(2)

# To download Area
for year in range(1998, 2025):
    print(f"Harvesting Pulses Area: {year}...")
    content = get_data_for_year(year, "Total Pulses", "Area")
    if content:
        with open(f"pulses_area_{year}.csv", "wb") as f: f.write(content)
    time.sleep(2)