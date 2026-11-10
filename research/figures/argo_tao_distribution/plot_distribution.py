"""Observed sampling coverage, not reconstruction accuracy or full profile QC.

Contract: one geographic quantitative panel; Argo index records aggregated into
1-degree cells, TAO sites with accepted 5–500 m temperature records over the
same time window. Plate Carree coordinates; actual Natural Earth land geometry.
Exports: editable SVG/PDF, 600 dpi PNG, counts and provenance. No subsampling.
"""
from pathlib import Path
import csv
import gzip
import json
import math
import os
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', '/tmp/argo-tao-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.lines import Line2D

START, END = '20150401', '20250630'
WEST, EAST, SOUTH, NORTH = 120, 280, -20, 20


def read_land(path):
    """Read the supplied Polygon shapefile without modifying it.

    ponytail: Polygon (type 5) only; use a GIS reader for any other shape type.
    """
    polygons = []
    with path.open('rb') as stream:
        header = stream.read(100)
        assert len(header) == 100 and struct.unpack('>i', header[:4])[0] == 9994
        assert struct.unpack('<i', header[32:36])[0] == 5
        while record_header := stream.read(8):
            assert len(record_header) == 8
            size = struct.unpack('>2i', record_header)[1] * 2
            record = stream.read(size)
            assert len(record) == size
            shape_type = struct.unpack_from('<i', record)[0]
            if shape_type == 0:
                continue
            assert shape_type == 5
            n_parts, n_points = struct.unpack_from('<2i', record, 36)
            offsets = list(struct.unpack_from(f'<{n_parts}i', record, 44)) + [n_points]
            points = np.frombuffer(record, dtype='<f8', offset=44+4*n_parts,
                                   count=2*n_points).reshape(-1, 2).copy()
            rings = [points[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
            assert all(len(ring) >= 4 for ring in rings)
            polygons.append(rings)
    assert polygons
    return polygons


def read_argo():
    counts = np.zeros((40, 160), dtype=np.int64)
    stats = dict(total_index_records=0, invalid_date=0, outside_time=0,
                 invalid_coordinates=0, outside_region=0, outside_pacific=0, duplicate_file=0,
                 retained_records=0)
    seen, floats = set(), set()
    with gzip.open(ROOT/'data/raw/profiles/ar_index_global_prof.txt.gz', 'rt') as stream:
        rows = csv.DictReader(line for line in stream if not line.startswith('#'))
        for row in rows:
            stats['total_index_records'] += 1
            date = row['date'][:8]
            if len(date) != 8 or not date.isdigit():
                stats['invalid_date'] += 1
                continue
            if not START <= date <= END:
                stats['outside_time'] += 1
                continue
            try:
                lat, lon = float(row['latitude']), float(row['longitude'])
            except ValueError:
                stats['invalid_coordinates'] += 1
                continue
            if not (math.isfinite(lat) and math.isfinite(lon)
                    and -90 <= lat <= 90 and -180 <= lon <= 360):
                stats['invalid_coordinates'] += 1
                continue
            lon %= 360
            if not (SOUTH <= lat <= NORTH and WEST <= lon <= EAST):
                stats['outside_region'] += 1
                continue
            if row['ocean'] != 'P':
                stats['outside_pacific'] += 1
                continue
            key = row['file']
            if key in seen:
                stats['duplicate_file'] += 1
                continue
            seen.add(key)
            floats.add(key.split('/')[1])
            counts[min(int(lat-SOUTH), 39), min(int(lon-WEST), 159)] += 1
            stats['retained_records'] += 1
    assert int(counts.sum()) == stats['retained_records'] == len(seen)
    assert sum(v for k, v in stats.items() if k != 'total_index_records') == stats['total_index_records']
    stats['unique_float_ids'] = len(floats)
    return counts, stats


def read_tao():
    sites, all_sites = [], []
    for path in sorted((ROOT/'data/raw/profiles/data').glob('*.ascii')):
        lines = path.read_text().splitlines()
        match = re.search(r'Location:\s*([\d.]+)([NS])\s+([\d.]+)([EW])', lines[0])
        if not match:
            raise ValueError(f'Unrecognized station header: {path.name}')
        lat = float(match[1]) * (1 if match[2] == 'N' else -1)
        lon = (float(match[3]) * (1 if match[4] == 'E' else -1)) % 360
        depths = []
        dates, accepted, inspected = set(), 0, 0
        quality_counts = {}
        for line in lines:
            text = line.strip()
            if text.startswith('Depth(M):'):
                depths = [float(v) for v in text.split(':', 1)[1].split('QUALITY')[0].split()]
            elif re.match(r'^\d{8}\s+\d{4}\s', text):
                fields = text.split()
                assert depths and len(fields) == len(depths)+3
                date, quality = fields[0], fields[-1]
                assert len(quality) == len(depths)
                if not START <= date <= END:
                    continue
                valid_record = False
                for depth, value, qc in zip(depths, fields[2:-1], quality):
                    if not 5 <= depth <= 500:
                        continue
                    inspected += 1
                    quality_counts[qc] = quality_counts.get(qc, 0)+1
                    value = float(value)
                    if qc in '123' and math.isfinite(value) and value != -9.999:
                        accepted += 1
                        valid_record = True
                if valid_record:
                    dates.add(date)
        site = dict(station=match[0].split(':')[1].strip(), latitude=lat,
                    longitude_east=lon, accepted_5day_dates=len(dates),
                    accepted_temperature_values=accepted, inspected_values=inspected,
                    quality_counts=quality_counts,
                    first_valid_date=min(dates) if dates else None,
                    last_valid_date=max(dates) if dates else None)
        all_sites.append(site)
        if dates and SOUTH <= lat <= NORTH and WEST <= lon <= EAST:
            sites.append(site)
    assert len({(s['latitude'], s['longitude_east']) for s in sites}) == len(sites)
    return sites, all_sites


def draw_land(ax, polygons):
    for rings in polygons:
        for shift in (0, 360):
            vertices, codes = [], []
            for ring in rings:
                coordinates = ring.copy()
                coordinates[:, 0] += shift
                vertices.extend(coordinates)
                codes.extend([MplPath.MOVETO] + [MplPath.LINETO]*(len(ring)-2) + [MplPath.CLOSEPOLY])
            vertices = np.asarray(vertices)
            if (vertices[:, 0].max() < 110 or vertices[:, 0].min() > 290
                    or vertices[:, 1].max() < -25 or vertices[:, 1].min() > 25):
                continue
            ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor='#d9dcdf',
                                   edgecolor='#7b8288', linewidth=0.35, zorder=3))


def main():
    counts, argo = read_argo()
    sites, all_sites = read_tao()
    land = read_land(ROOT/'data/reference/ne_110m_land/ne_110m_land.shp')
    assert counts.max() > 0 and sites
    summary = dict(time_window=[START, END], region=[WEST, EAST, SOUTH, NORTH],
                   argo=argo, tao_files=len(all_sites), tao_retained_sites=len(sites),
                   tao_excluded_sites=[s for s in all_sites if s not in sites],
                   tao_sites=sites, cell_size_degrees=1, count_range=[1, int(counts.max())],
                   argo_qc='Pacific ocean flag P; index only; no temperature/salinity or depth QC available.',
                   tao_qc='At least one finite nonmissing 5–500 m temperature with QC 1, 2 or 3.',
                   projection='Plate Carree / geographic longitude-latitude',
                   land='Natural Earth 1:110m land v4.1.0, WGS84',
                   caveat='Time-aggregated presence/counts do not establish monthly continuity or independent samples.')
    (OUT/'coverage_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT/'argo_counts_1degree.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['latitude_south', 'longitude_west_0_360', 'profile_count'])
        writer.writerows((SOUTH+y, WEST+x, int(counts[y, x])) for y in range(40) for x in range(160))
    with (OUT/'tao_sites.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in sites[0] if k != 'quality_counts'])
        writer.writeheader()
        writer.writerows({k: v for k, v in s.items() if k != 'quality_counts'} for s in sites)

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7.5, 'axes.linewidth': 0.6, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': True, 'axes.spines.right': True})
    fig = plt.figure(figsize=(7.2047244, 3.6220472), facecolor='white')
    ax = fig.add_axes([0.075, 0.32, 0.90, 0.51])
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#f6f8fa')
    mesh = ax.pcolormesh(np.arange(WEST, EAST+1), np.arange(SOUTH, NORTH+1),
                         np.ma.masked_equal(counts, 0), cmap=cmap,
                         norm=LogNorm(1, int(counts.max())), shading='flat',
                         rasterized=True, zorder=1)
    draw_land(ax, land)
    ax.scatter([s['longitude_east'] for s in sites], [s['latitude'] for s in sites],
               s=17, marker='^', facecolor='#d65a32', edgecolor='white', linewidth=0.5, zorder=5)
    ax.add_patch(Rectangle((WEST, SOUTH), EAST-WEST, NORTH-SOUTH, fill=False,
                           edgecolor='#333a40', linestyle=(0, (4, 2)), linewidth=0.6, zorder=4))
    ax.set(xlim=(110, 290), ylim=(-25, 25), aspect='equal')
    ax.set_xticks([120, 140, 160, 180, 200, 220, 240, 260, 280])
    ax.set_xticklabels(['120°E', '140°E', '160°E', '180°', '160°W', '140°W', '120°W', '100°W', '80°W'])
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(['20°S', '10°S', '0°', '10°N', '20°N'])
    ax.tick_params(length=2.5, pad=3)
    ax.grid(color='#52606b', linewidth=0.25, alpha=0.22, zorder=2)
    fig.text(0.075, 0.945, 'Argo profiles and TAO/TRITON moorings in the tropical Pacific',
             fontsize=10.2, weight='bold', va='top')
    fig.text(0.075, 0.875, f'Apr 2015 – Jun 2025  |  {argo["retained_records"]:,} Argo profiles  |  '
             f'{argo["unique_float_ids"]:,} floats  |  {len(sites)} mooring sites', fontsize=7.8)
    cax = fig.add_axes([0.20, 0.20, 0.43, 0.032])
    cb = fig.colorbar(mesh, cax=cax, orientation='horizontal')
    ticks = [v for v in (1, 3, 10, 30, 100, 300, 1000) if v <= counts.max()]
    if counts.max() > ticks[-1] * 1.5:
        ticks.append(int(counts.max()))
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(v) for v in ticks])
    cb.ax.minorticks_off()
    cb.ax.tick_params(length=2, pad=2)
    cb.set_label('Argo profiles per 1° × 1° cell (log scale)', fontsize=7.5, labelpad=4)
    handles = [Line2D([], [], color='none', marker='^', markersize=5,
                       markerfacecolor='#d65a32', markeredgecolor='white', label='TAO/TRITON'),
               Line2D([], [], color='#333a40', linestyle='--', linewidth=0.6, label='Study region')]
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.68, 0.205),
                frameon=False, fontsize=7.2, labelspacing=0.6, handlelength=1.7)
    fig.text(0.075, 0.045, 'Argo: Pacific index locations, not profile-quality screened.  TAO: valid 5–500 m temperature, QC 1–3.\n'
             'Counts aggregate the full period; station presence does not imply continuous coverage.  Land: Natural Earth.',
             fontsize=6.1, linespacing=1.5, color='#4d555d')
    fig.canvas.draw()
    sys.path.insert(0, str(Path.home()/'.codex/skills/nature-figure/scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig, json_out=str(OUT/'alignment.json'),
                                      exclude_axes=[cax], strict=True)
    fig.savefig(OUT/'argo_tao_distribution.svg', dpi=600, facecolor='white')
    fig.savefig(OUT/'argo_tao_distribution.pdf', dpi=600, facecolor='white')
    fig.savefig(OUT/'argo_tao_distribution.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ['argo', 'tao_files', 'tao_retained_sites', 'count_range']}, indent=2))


if __name__ == '__main__':
    # Runnable parser/data-integrity checks accompany the actual analysis.
    assert (-140 % 360) == 220 and SOUTH < NORTH and WEST < EAST
    main()
