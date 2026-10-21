
import Hydrolang from '../../../hydrolang/hydro.js';
import fs from 'fs';
import path from 'path';

const hydro = new Hydrolang({ cache: false });

const OUTPUT_DIR = path.join(process.env.HYDRO_DATA_DIR || process.cwd(), 'aorc');   // $HYDRO_DATA_DIR/aorc
if (!fs.existsSync(OUTPUT_DIR)) fs.mkdirSync(OUTPUT_DIR, { recursive: true });

// Louisiana BBOX
const BBOX = [-94.1, 28.8, -88.7, 33.1];

const VARIABLES = [
    'APCP_surface',
    'TMP_2maboveground',
    'SPFH_2maboveground',
    'DLWRF_surface',
    'DSWRF_surface',
    'PRES_surface',
    'UGRD_10maboveground',
    'VGRD_10maboveground'
];

// Helper to sum grids
function addGrid(acc, newGrid) {
    if (!acc) return newGrid; // First step
    for (let i = 0; i < acc.length; i++) {
        for (let j = 0; j < acc[i].length; j++) {
            acc[i][j] += newGrid[i][j];
        }
    }
    return acc;
}

async function processDay(year, month, day) {
    const yyyy = year;
    const mm = month.toString().padStart(2, '0');
    const dd = day.toString().padStart(2, '0');
    const dateStr = `${yyyy}-${mm}-${dd}`;

    const outFile = path.join(OUTPUT_DIR, `aorc_${yyyy}${mm}${dd}_daily.json`);
    if (fs.existsSync(outFile)) {
        process.stdout.write('.');
        return; // Skip existing
    }

    console.log(`\nProcessing ${dateStr}...`);

    let dailyData = {};
    // Initialize results structure
    VARIABLES.forEach(v => dailyData[v] = null);

    // Iterate Hours
    for (let h = 0; h < 24; h++) {
        const hh = h.toString().padStart(2, '0');
        const start = `${dateStr}T${hh}:00:00Z`;
        const end = `${dateStr}T${hh}:00:00Z`; // Hydrolang might expect duration or just point. 
        // Note: end param is tricky. Usually "start" is enough for 1 step?
        // Let's use start==end or 1 hour diff.
        // Based on test, we need start/end.
        // If end == start, does it get 1 frame? Yes likely.

        process.stdout.write(`${hh} `);

        try {
            // Parallel fetch all variables? Or sequential to save memory?
            // AORC might throttle. Sequential is safer.
            for (const v of VARIABLES) {
                const data = await hydro.data.retrieve({
                    params: { source: 'aorc', datatype: 'grid-data' },
                    args: {
                        dataset: 'aorc-v1.1',
                        variables: [v],
                        bbox: BBOX,
                        startDate: start,
                        endDate: start, // Single hour
                        format: 'json'
                    }
                });

                if (data && data.data) {
                    // Accumulate
                    dailyData[v] = addGrid(dailyData[v], data.data);
                } else {
                    // Handle missing?
                }
            }
        } catch (e) {
            console.error(`Error ${dateStr} ${hh}: ${e.message}`);
        }
    }

    // Average values (Except Precip which is Sum)
    // APCP is accumulation? Docs say "Hourly total precipitation". So Sum is correct.
    // Others (Temp, etc) should be Averaged / 24.

    VARIABLES.forEach(v => {
        if (v !== 'APCP_surface' && dailyData[v]) {
            const grid = dailyData[v];
            for (let i = 0; i < grid.length; i++) {
                for (let j = 0; j < grid[i].length; j++) {
                    grid[i][j] /= 24.0;
                }
            }
        }
    });

    // Save
    const output = {
        date: dateStr,
        bbox: BBOX,
        variables: dailyData
    };

    fs.writeFileSync(outFile, JSON.stringify(output));
    console.log(" Saved.");
}

async function main() {
    // 2012 Leap year
    const year = 2012;
    const daysInMonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    for (let m = 1; m <= 12; m++) {
        for (let d = 1; d <= daysInMonth[m - 1]; d++) {
            await processDay(year, m, d);
        }
    }
}

main();
