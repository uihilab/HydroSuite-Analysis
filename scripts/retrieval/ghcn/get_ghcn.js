
import Hydrolang from '../../../hydrolang/hydro.js';
import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { parseStringPromise } from 'xml2js';

const hydro = new Hydrolang({ cache: false });

// Configuration
const BBOX = {
    west: -91.5, // Refined for Amite/Calcasieu
    south: 30.0,
    east: -90.0,
    north: 31.0
};
const START_DATE = '2000-01-01';
const END_DATE = '2020-12-31';

const DATA_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '../../data/raw/ghcn');
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });

async function main() {
    try {
        console.log("Fetching sites via HydroLang Logic (CUAHSI)...");

        // Access the datasource module directly from the instance
        // Assuming hydro.datasources.waterOneFlow exists based on standard usage
        // OR we can access the module if we know the structure.
        // data.js usually imports * from datasources.js

        const wof = hydro.datasources.waterOneFlow;
        if (!wof) throw new Error("Could not access waterOneFlow module from HydroLang instance.");

        // 1. Get Endpoint
        const endpoint = wof.sourceType('NOAA GHCN');
        console.log(`Endpoint: ${endpoint}`);

        // 2. Get Sites
        // Generate Body
        const sitesBody = wof.GetSitesByBoxObject.body({
            west: BBOX.west,
            south: BBOX.south,
            east: BBOX.east,
            north: BBOX.north
        });

        // Manually fetch to bypass specific proxy requirements if they are causing issues
        // But still using the library for all logic.
        const sitesResponse = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'text/xml; charset=utf-8' },
            body: `<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>${sitesBody}</soap:Body>
</soap:Envelope>`
        });

        const sitesXml = await sitesResponse.text();
        fs.writeFileSync(path.join(DATA_DIR, 'sites_response.xml'), sitesXml);
        console.log("Saved sites_response.xml");

        // Parse (Optional, but good to prove we can get a site)
        // Simple regex or xml2js to get a site code
        // For now, hardcode a known one or parse
        // const match = sitesXml.match(/<SiteCode>(.*?)<\/SiteCode>/);

        // 3. Get Values for a sample (Amite)
        const sampleSite = 'ghcn:US1LAEB0001';
        console.log(`Fetching values for ${sampleSite}...`);

        const valuesBody = wof.GetValuesObject.body({
            location: sampleSite,
            variable: 'ghcn:PRCP',
            startDate: START_DATE,
            endDate: END_DATE
        });

        const valuesResponse = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'text/xml; charset=utf-8' },
            body: `<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>${valuesBody}</soap:Body>
</soap:Envelope>`
        });

        const valuesXml = await valuesResponse.text();
        fs.writeFileSync(path.join(DATA_DIR, `${sampleSite.replace(':', '_')}.xml`), valuesXml);
        console.log(`Saved ${sampleSite} data`);

    } catch (e) {
        console.error("Error in GHCN retrieval:", e);
    }
}

main();
