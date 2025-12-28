// Export utilities for Dashboard data

interface AnalysisRecord {
    id: string;
    type: "histopathology" | "mammography";
    prediction: string;
    confidence: number;
    timestamp: string;
}

// Export to CSV
export const exportToCSV = (data: AnalysisRecord[], filename: string = 'analysis_report'): void => {
    if (data.length === 0) {
        throw new Error('No data to export');
    }

    // CSV Headers
    const headers = ['ID', 'Type', 'Prediction', 'Confidence (%)', 'Date', 'Time'];

    // Convert data to CSV rows
    const rows = data.map((record) => {
        const date = new Date(record.timestamp);
        return [
            record.id,
            record.type === 'histopathology' ? 'Histopathology' : 'Mammography',
            record.prediction,
            record.confidence.toFixed(2),
            date.toLocaleDateString('en-US'),
            date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
        ];
    });

    // Combine headers and rows
    const csvContent = [
        headers.join(','),
        ...rows.map(row => row.map(cell => `"${cell}"`).join(','))
    ].join('\n');

    // Create and download file
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', `${filename}_${formatDateForFilename()}.csv`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

// Export to Excel (XLSX format using plain JS)
export const exportToExcel = (data: AnalysisRecord[], filename: string = 'analysis_report'): void => {
    if (data.length === 0) {
        throw new Error('No data to export');
    }

    // Create XML-based Excel file (compatible with all Excel versions)
    const headers = ['ID', 'Type', 'Prediction', 'Confidence (%)', 'Date', 'Time'];

    let xmlContent = `<?xml version="1.0" encoding="UTF-8"?>
<?mso-application progid="Excel.Sheet"?>
<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"
    xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">
    <Styles>
        <Style ss:ID="Header">
            <Font ss:Bold="1" ss:Color="#FFFFFF"/>
            <Interior ss:Color="#8B5CF6" ss:Pattern="Solid"/>
            <Alignment ss:Horizontal="Center"/>
        </Style>
        <Style ss:ID="Success">
            <Font ss:Color="#22C55E"/>
        </Style>
        <Style ss:ID="Warning">
            <Font ss:Color="#F59E0B"/>
        </Style>
        <Style ss:ID="Danger">
            <Font ss:Color="#EF4444"/>
        </Style>
        <Style ss:ID="Default">
            <Alignment ss:Horizontal="Center"/>
        </Style>
    </Styles>
    <Worksheet ss:Name="Analysis Report">
        <Table>`;

    // Add header row
    xmlContent += '\n            <Row>';
    headers.forEach(header => {
        xmlContent += `<Cell ss:StyleID="Header"><Data ss:Type="String">${header}</Data></Cell>`;
    });
    xmlContent += '</Row>';

    // Add data rows
    data.forEach((record) => {
        const date = new Date(record.timestamp);
        const prediction = record.prediction.toLowerCase();
        let styleId = 'Default';

        if (prediction.includes('benign')) styleId = 'Success';
        else if (prediction.includes('suspicious')) styleId = 'Warning';
        else if (prediction.includes('malignant')) styleId = 'Danger';

        xmlContent += '\n            <Row>';
        xmlContent += `<Cell ss:StyleID="Default"><Data ss:Type="String">${record.id}</Data></Cell>`;
        xmlContent += `<Cell ss:StyleID="Default"><Data ss:Type="String">${record.type === 'histopathology' ? 'Histopathology' : 'Mammography'}</Data></Cell>`;
        xmlContent += `<Cell ss:StyleID="${styleId}"><Data ss:Type="String">${record.prediction}</Data></Cell>`;
        xmlContent += `<Cell ss:StyleID="Default"><Data ss:Type="Number">${record.confidence.toFixed(2)}</Data></Cell>`;
        xmlContent += `<Cell ss:StyleID="Default"><Data ss:Type="String">${date.toLocaleDateString('en-US')}</Data></Cell>`;
        xmlContent += `<Cell ss:StyleID="Default"><Data ss:Type="String">${date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</Data></Cell>`;
        xmlContent += '</Row>';
    });

    xmlContent += `
        </Table>
    </Worksheet>
</Workbook>`;

    // Create and download file
    const blob = new Blob([xmlContent], { type: 'application/vnd.ms-excel' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', `${filename}_${formatDateForFilename()}.xls`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

// Export summary statistics
export const exportSummaryReport = (data: AnalysisRecord[]): void => {
    if (data.length === 0) {
        throw new Error('No data to export');
    }

    // Calculate statistics
    const stats = {
        total: data.length,
        histopathology: data.filter(r => r.type === 'histopathology').length,
        mammography: data.filter(r => r.type === 'mammography').length,
        benign: data.filter(r => r.prediction.toLowerCase().includes('benign')).length,
        suspicious: data.filter(r => r.prediction.toLowerCase().includes('suspicious')).length,
        malignant: data.filter(r => r.prediction.toLowerCase().includes('malignant')).length,
        avgConfidence: data.reduce((sum, r) => sum + r.confidence, 0) / data.length,
    };

    // Create summary CSV
    const summaryContent = `DeepBreast AI - Analysis Summary Report
Generated: ${new Date().toLocaleString()}

=== OVERVIEW ===
Total Analyses,${stats.total}
Histopathology Analyses,${stats.histopathology}
Mammography Analyses,${stats.mammography}

=== RESULTS BREAKDOWN ===
Benign Cases,${stats.benign}
Suspicious Cases,${stats.suspicious}
Malignant Cases,${stats.malignant}

=== CONFIDENCE ===
Average Confidence,${stats.avgConfidence.toFixed(2)}%

=== DETAILED DATA ===
ID,Type,Prediction,Confidence,Timestamp
${data.map(r => `${r.id},${r.type},${r.prediction},${r.confidence.toFixed(2)}%,${new Date(r.timestamp).toLocaleString()}`).join('\n')}
`;

    const blob = new Blob([summaryContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', `summary_report_${formatDateForFilename()}.csv`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
};

// Helper function to format date for filename
const formatDateForFilename = (): string => {
    const now = new Date();
    return `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}`;
};

// Get analysis history from localStorage
export const getAnalysisHistory = (): AnalysisRecord[] => {
    try {
        const historyStr = localStorage.getItem('analysisHistory');
        if (!historyStr) return [];
        return JSON.parse(historyStr);
    } catch {
        return [];
    }
};
