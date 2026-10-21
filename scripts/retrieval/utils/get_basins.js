
import Hydrolang from '../../../hydrolang/hydro.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const hydro = new Hydrolang({ includeVisuals: false, cache: false });

const OUTPUT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '../../data/raw/basins');
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

const SITES = [
    { id: '07378500', name: 'Amite_DenhamSprings', source: 'nwissite' }
];

// Note: NLDI module supports 'getBasin'
// Params: featureSource, featureId

async function main() {
    for (const site of SITES) {
        console.log(`Getting basin for ${site.name}...`);
        try {
            const geojson = await hydro.data.retrieve({
                params: { source: 'nldi', datatype: 'getBasin' },
                args: {
                    featureSource: site.source,
                    featureId: site.id
                }
            });

            fs.writeFileSync(path.join(OUTPUT_DIR, `${site.name}.geojson`), JSON.stringify(geojson, null, 2));
            console.log(`Saved ${site.name}.geojson`);
        } catch (e) {
            console.error(e);
        }
    }
}

main();
