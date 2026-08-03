"""Read-only parsing helpers for dataset previews."""

import zipfile

import numpy as np


PREVIEWABLE_DATA_EXTENSIONS = ('.csv', '.txt', '.tsv')


def _decode_line(line):
    try:
        return line.decode('utf-8').strip()
    except UnicodeDecodeError:
        return line.decode('latin-1').strip()


def _read_preview_lines(file_obj, max_preview_rows=10):
    lines = []
    total_rows = 0
    for line in file_obj:
        total_rows += 1
        if total_rows <= max_preview_rows:
            lines.append(_decode_line(line))
    return lines, total_rows


def _build_preview(lines, source_name, total_rows):
    if not lines:
        return None

    first_line = lines[0]
    delimiter = ',' if ',' in first_line else '\t' if '\t' in first_line else ' '
    rows = [line.split(delimiter) for line in lines]
    chart_data = []
    for index, value in enumerate(rows[0] if rows else []):
        try:
            numeric_value = float(value)
            is_missing = bool(np.isnan(numeric_value) or not value.strip() or value.strip().lower() == 'nan')
            chart_data.append({
                'x': index,
                'y': None if is_missing else numeric_value,
                'missing': is_missing,
            })
        except (TypeError, ValueError):
            chart_data.append({'x': index, 'y': None, 'missing': True})

    total_points = len(chart_data)
    missing_points = sum(point['missing'] for point in chart_data)
    return {
        'fileName': source_name,
        'totalRows': total_rows,
        'columns': len(rows[0]) if rows else 0,
        'headers': rows[0] if rows else [],
        'rows': rows[1:10],
        'seriesRows': rows[:10],
        'chartData': chart_data[:500],
        'totalPoints': total_points,
        'missingPoints': missing_points,
        'missingRate': round(missing_points / total_points * 100, 1) if total_points else 0,
    }


def create_preview_from_upload(uploaded_file):
    """Return the existing Chart.js preview schema without saving the upload."""
    source_name = uploaded_file.name
    lower_name = source_name.lower()
    uploaded_file.seek(0)

    if lower_name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as archive:
            for inner_name in archive.namelist():
                if inner_name.lower().endswith(PREVIEWABLE_DATA_EXTENSIONS):
                    with archive.open(inner_name) as inner_file:
                        lines, total_rows = _read_preview_lines(inner_file)
                    return _build_preview(lines, inner_name, total_rows)
        return None

    if lower_name.endswith(PREVIEWABLE_DATA_EXTENSIONS):
        lines, total_rows = _read_preview_lines(uploaded_file)
        return _build_preview(lines, source_name, total_rows)

    return None
