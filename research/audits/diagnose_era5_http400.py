from pathlib import Path
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from download_samples import classify_cds_error, secrets_for
import cdsapi

secrets = secrets_for('CDSAPI_URL', 'CDSAPI_KEY')
client = cdsapi.Client(url=secrets['CDSAPI_URL'], key=secrets['CDSAPI_KEY'],
                       timeout=60, retry_max=0, quiet=True)
request = {
    'product_type': ['monthly_averaged_reanalysis'],
    'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
    'year': ['2018'], 'month': ['08'], 'time': ['00:00'],
    'data_format': 'netcdf', 'download_format': 'unarchived',
    'area': [20.25, 120, -20.25, 180],
}
target = Path(__file__).with_suffix('.part.nc')
try:
    client.retrieve('reanalysis-era5-single-levels-monthly-means', request, str(target))
    result = {'status': 'unexpected_success'}
except Exception as error:
    response = getattr(error, 'response', None)
    try:
        body = response.json()
    except Exception:
        body = None
    traceback_text = body.get('traceback', '') if isinstance(body, dict) else ''
    lowered = traceback_text.lower() if isinstance(traceback_text, str) else ''
    traceback_category = (
        'licence_not_accepted' if any(word in lowered for word in ('licence', 'license', 'terms')) else
        'request_validation' if any(word in lowered for word in ('invalid', 'validation', 'area', 'format', 'variable')) else
        'request_too_large' if any(word in lowered for word in ('too large', 'cost limit', 'quota')) else
        'other'
    )
    title = body.get('title') if isinstance(body, dict) else None
    traceback_lines = [line.strip() for line in traceback_text.splitlines() if line.strip()]
    safe_tail = traceback_lines[-1] if traceback_lines else None
    if safe_tail:
        safe_tail = re.sub(r'https?://\S+', '<URL_REDACTED>', safe_tail)
        safe_tail = re.sub(r'[A-Za-z0-9_-]{40,}', '<TOKEN_REDACTED>', safe_tail)
        safe_tail = ''.join(character for character in safe_tail if character.isprintable())[:300]
    result = {
        'exception_type': type(error).__name__,
        'http_status': getattr(response, 'status_code', None),
        'error_category': classify_cds_error(response),
        'json_keys': sorted(body) if isinstance(body, dict) else None,
        'title': title if isinstance(title, str) and len(title) < 120 and 'http' not in title.lower() else None,
        'traceback_category': traceback_category,
        'safe_traceback_tail': safe_tail,
        'reason_code': body.get('reason') if isinstance(body, dict) and isinstance(body.get('reason'), str)
                       and len(body['reason']) < 80 and 'http' not in body['reason'].lower() else None,
    }
finally:
    target.unlink(missing_ok=True)
print(json.dumps(result))
