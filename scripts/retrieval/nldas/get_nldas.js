
import Hydrolang from '../../../hydrolang/hydro.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const hydro = new Hydrolang({ cache: false });

const DOWNLOAD_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), '../../data/raw/nldas');
if (!fs.existsSync(DOWNLOAD_DIR)) {
    fs.mkdirSync(DOWNLOAD_DIR, { recursive: true });
}

async function main() {
    // 2000-2020 Daily Data
    const startYear = 2012;
    const endYear = 2020;

    // Helper to format date
    const formatDate = (date) => {
        const y = date.getUTCFullYear();
        const m = (date.getUTCMonth() + 1).toString().padStart(2, '0');
        const d = date.getUTCDate().toString().padStart(2, '0');
        return `${y}-${m}-${d}`;
    };

    const startDate = new Date('2012-06-01T00:00:00Z');
    const endDate = new Date('2012-07-31T00:00:00Z');

    const totalDays = (endDate - startDate) / (1000 * 60 * 60 * 24) + 1;
    console.log(`Starting NLDAS Daily Retrieval: ${formatDate(startDate)} to ${formatDate(endDate)} (${totalDays} days)`);

    let currentDayIndex = 0;
    for (let d = new Date(startDate); d <= endDate; d.setUTCDate(d.getUTCDate() + 1)) {
        currentDayIndex++;
        const dateStr = formatDate(d);
        const yyyy = dateStr.split('-')[0];
        const mm = dateStr.split('-')[1];
        const dd = dateStr.split('-')[2];

        const filename = `NLDAS_FOR0010_D.A${yyyy}${mm}${dd}.030.beta.nc`;
        const destPath = path.join(DOWNLOAD_DIR, filename);

        console.log(`[${currentDayIndex}/${totalDays}] Processing ${dateStr}...`);

        if (fs.existsSync(destPath)) {
            // console.log(`Skipping ${filename}`);
            continue;
        }

        console.log(`Downloading ${filename} (${dateStr})...`);

        try {
            const data = await hydro.data.retrieve({
                params: {
                    source: "nldas",
                    datatype: "raw-netcdf",
                },
                args: {
                    dataset: "nldas-3-daily",
                    startDate: dateStr
                }
            });

            if (data && data.byteLength > 0) {
                const buffer = Buffer.from(data);
                fs.writeFileSync(destPath, buffer);
                // console.log(`Saved ${filename}`);
            } else {
                console.error(`Empty response for ${dateStr}`);
            }

            // Simple rate limiting
            await new Promise(r => setTimeout(r, 100));

        } catch (e) {
            console.error(`Error downloading ${dateStr}: ${e.message}`);
        }
    }
}

main();
