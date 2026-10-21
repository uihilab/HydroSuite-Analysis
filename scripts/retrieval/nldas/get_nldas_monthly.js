import Hydrolang from '../../../hydrolang/hydro.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const hydro = new Hydrolang({ cache: false });

// Output: $HYDRO_DATA_DIR/nldas (current directory if unset)
const DOWNLOAD_DIR = path.join(process.env.HYDRO_DATA_DIR || process.cwd(), 'nldas');

if (!fs.existsSync(DOWNLOAD_DIR)) {
    try {
        fs.mkdirSync(DOWNLOAD_DIR, { recursive: true });
    } catch (e) {
        console.error(`❌ Cannot create output directory: ${DOWNLOAD_DIR}`);
        console.error(e.message);
        process.exit(1);
    }
}

/**
 * Download NLDAS data for a specific month
 * 
 * Usage: node get_nldas_monthly.js 2012 8
 *        Downloads August 2012
 */
async function downloadMonth(year, month) {
    // Validate inputs
    if (year < 2000 || year > 2024) {
        console.error('❌ Invalid year. Must be 2000-2024');
        process.exit(1);
    }
    if (month < 1 || month > 12) {
        console.error('❌ Invalid month. Must be 1-12');
        process.exit(1);
    }

    // Get first and last day of month
    const startDate = new Date(Date.UTC(year, month - 1, 1));
    const endDate = new Date(Date.UTC(year, month, 0)); // Last day of month

    const formatDate = (date) => {
        const y = date.getUTCFullYear();
        const m = (date.getUTCMonth() + 1).toString().padStart(2, '0');
        const d = date.getUTCDate().toString().padStart(2, '0');
        return `${y}-${m}-${d}`;
    };

    const totalDays = endDate.getUTCDate();
    const monthName = startDate.toLocaleString('en-US', { month: 'long', timeZone: 'UTC' });

    console.log('='.repeat(70));
    console.log(`📥 NLDAS MONTHLY DOWNLOAD: ${monthName} ${year}`);
    console.log('='.repeat(70));
    console.log(`Period: ${formatDate(startDate)} to ${formatDate(endDate)} (${totalDays} days)`);
    console.log(`Destination: ${DOWNLOAD_DIR}`);
    console.log(`File size: ~640 MB per day = ~${(totalDays * 640 / 1024).toFixed(1)} GB total`);
    console.log('');

    let downloaded = 0;
    let skipped = 0;
    let failed = 0;

    for (let d = new Date(startDate); d <= endDate; d.setUTCDate(d.getUTCDate() + 1)) {
        const dateStr = formatDate(d);
        const yyyy = dateStr.split('-')[0];
        const mm = dateStr.split('-')[1];
        const dd = dateStr.split('-')[2];

        const filename = `NLDAS_FOR0010_D.A${yyyy}${mm}${dd}.030.beta.nc`;
        const destPath = path.join(DOWNLOAD_DIR, filename);
        const dayNum = d.getUTCDate();

        process.stdout.write(`[${dayNum}/${totalDays}] ${dateStr}... `);

        // Check if exists
        if (fs.existsSync(destPath)) {
            const stats = fs.statSync(destPath);
            const sizeMB = (stats.size / (1024 * 1024)).toFixed(1);
            console.log(`✓ Skip (${sizeMB} MB)`);
            skipped++;
            continue;
        }

        // Download
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
                // VALIDATION: Check minimum file size (10MB)
                // Real NLDAS files are ~640MB. Small files are likely XML/HTML errors.
                const MIN_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

                if (data.byteLength < MIN_SIZE_BYTES) {
                    console.log(`✗ Error: File too small (${(data.byteLength / 1024).toFixed(1)} KB). Likely error response.`);
                    try {
                        const preview = Buffer.from(data).slice(0, 100).toString();
                        console.log(`Content preview: ${preview.replace(/\n/g, ' ')}`);
                    } catch (e) { }
                    failed++;
                } else {
                    const buffer = Buffer.from(data);
                    fs.writeFileSync(destPath, buffer);
                    const sizeMB = (buffer.length / (1024 * 1024)).toFixed(1);
                    console.log(`✓ Downloaded (${sizeMB} MB)`);
                    downloaded++;
                }
            } else {
                console.log('✗ Empty response');
                failed++;
            }

            // Rate limiting
            await new Promise(r => setTimeout(r, 100));

        } catch (e) {
            console.log(`✗ Error: ${e.message}`);
            failed++;
        }
    }

    console.log('\\n' + '='.repeat(70));
    console.log('SUMMARY');
    console.log('='.repeat(70));
    console.log(`Downloaded: ${downloaded} files`);
    console.log(`Skipped:    ${skipped} files (already exist)`);
    console.log(`Failed:     ${failed} files`);
    console.log('='.repeat(70));
}

// Parse command line arguments
const args = process.argv.slice(2);

if (args.length < 2) {
    console.log('Usage: node get_nldas_monthly.js <year> <month>');
    console.log('');
    console.log('Examples:');
    console.log('  node get_nldas_monthly.js 2012 8      # Download August 2012');
    console.log('  node get_nldas_monthly.js 2016 1      # Download January 2016');
    console.log('');
    console.log('Note: Each day is ~640 MB. A full month = ~20 GB');
    process.exit(1);
}

const year = parseInt(args[0]);
const month = parseInt(args[1]);

downloadMonth(year, month);
