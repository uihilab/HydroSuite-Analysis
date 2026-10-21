
import Hydrolang from '../../../hydrolang/hydro.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Initialize HydroLang
const hydro = new Hydrolang({ cache: false });

const OUTPUT_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '../../data/raw/nwm');
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

async function downloadNWM() {
    const years = [2012, 2013]; // Example

    for (const year of years) {
        try {
            console.log(`Retrieving NWM Streamflow for ${year} sample...`);

            // Fetch a sample point for validation as per AORC logic
            // To do full retrieval, we'd need to iterate over time/space which is heavy for a script.
            // But this proves the usage of the library.

            const data = await hydro.data.retrieve({
                params: { source: 'nwm', datatype: 'point-data' },
                args: {
                    dataset: 'nwm-retrospective-2-1-zarr-pds',
                    variables: ['streamflow'],
                    latitude: 30.5,
                    longitude: -91.0,
                    startDate: `${year}-01-01T00:00:00Z`,
                    endDate: `${year}-01-01T01:00:00Z`
                }
            });

            fs.writeFileSync(path.join(OUTPUT_DIR, `sample_${year}_point.json`), JSON.stringify(data, null, 2));
            console.log(`Saved sample NWM data for ${year}`);

        } catch (e) {
            console.error(`Error with HydroLang NWM: ${e.message}`);
        }
    }
}

downloadNWM();
