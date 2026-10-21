
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import https from 'https';

// --- CONFIGURATION ---
const DOWNLOAD_DIR = path.join(process.env.HYDRO_DATA_DIR || process.cwd(), 'prism');   // $HYDRO_DATA_DIR/prism
const START_YEAR = 2000;
const END_YEAR = 2020;
const VARS = ['ppt', 'tmean']; // Precip and Temp
const DELAY_MS = 1000; // 1 second delay to be nice

// Ensure directories exist
VARS.forEach(v => {
    const dir = path.join(DOWNLOAD_DIR, v);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
});

// Basic HTTP/HTTPS switch
import http from 'http';

async function downloadFile(url, dest) {
    return new Promise((resolve, reject) => {
        const protocol = url.startsWith('https') ? https : http;

        protocol.get(url, (response) => {
            // Check for Redirect
            if (response.statusCode === 301 || response.statusCode === 302 || response.statusCode === 307) {
                const newUrl = response.headers.location;
                // console.log(`Redirecting to ${newUrl}...`);
                downloadFile(newUrl, dest)
                    .then(resolve)
                    .catch(reject);
                return;
            }

            if (response.statusCode !== 200) {
                fs.unlink(dest, () => { });
                reject(new Error(`Status ${response.statusCode}`));
                return;
            }

            const file = fs.createWriteStream(dest);
            response.pipe(file);
            file.on('finish', () => {
                file.close(() => resolve());
            });
        }).on('error', (err) => {
            fs.unlink(dest, () => { });
            reject(err);
        });
    });
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
    console.log(`=== PRISM DOWNLOADER (${START_YEAR}-${END_YEAR}) ===`);
    console.log(`Target: ${DOWNLOAD_DIR}`);

    for (const variable of VARS) {
        console.log(`\n--- Processing ${variable.toUpperCase()} ---`);

        for (let year = START_YEAR; year <= END_YEAR; year++) {
            // Days check
            const isLeap = (year % 4 === 0 && year % 100 !== 0) || (year % 400 === 0);
            const daysInMonth = [31, isLeap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

            for (let month = 1; month <= 12; month++) {
                for (let day = 1; day <= daysInMonth[month - 1]; day++) {

                    const yyyy = year;
                    const mm = month.toString().padStart(2, '0');
                    const dd = day.toString().padStart(2, '0');
                    const dateStr = `${yyyy}${mm}${dd}`;

                    const filename = `PRISM_${variable}_stable_4kmD2_${dateStr}_bil.zip`;
                    const destPath = path.join(DOWNLOAD_DIR, variable, filename);

                    // Size validation: existing empty files must be re-downloaded
                    if (fs.existsSync(destPath)) {
                        const stats = fs.statSync(destPath);
                        if (stats.size > 1000) {
                            process.stdout.write('.');
                            continue;
                        }
                        // Delete if small
                        fs.unlinkSync(destPath);
                    }

                    // URL Construction 
                    // Use HTTP to permit easier redirection handling if needed, or stick to provided URL
                    const url = `http://services.nacse.org/prism/data/public/4km/${variable}/${dateStr}`;

                    try {
                        process.stdout.write('v');
                        await downloadFile(url, destPath);
                        await sleep(DELAY_MS);
                    } catch (e) {
                        console.error(`\nFailed ${dateStr} (${url}): ${e.message}`);
                    }
                }
            }
        }
    }
    console.log("\nDone!");
}

main();
