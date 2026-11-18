
const { chromium } = require('playwright');
const fs = require('fs');

(async () => {
    const args = process.argv.slice(2);
    if (args.length < 2) {
        console.error('Usage: node _pdf_generator.js <input.html> <output.pdf>');
        process.exit(1);
    }
    const [htmlPath, pdfPath] = args;

    const browser = await chromium.launch();
    const page = await browser.newPage();

    const html = fs.readFileSync(htmlPath, 'utf-8');
    await page.setContent(html, { waitUntil: 'networkidle' });

    await page.pdf({
        path: pdfPath,
        format: 'A4',
        margin: { top: '20mm', bottom: '20mm', left: '15mm', right: '15mm' },
        displayHeaderFooter: true,
        headerTemplate: '<div style="font-size: 8px; color: #9ca3af; padding: 0 15mm;">Ergonomic Assessment Report</div>',
        footerTemplate: '<div style="font-size: 8px; color: #9ca3af; padding: 0 15mm; text-align: center;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>',
    });

    await browser.close();
    console.log('PDF generated successfully');
})();
