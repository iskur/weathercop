# VARWG Cache Fixtures

This directory contains serialized VARWG instances used for testing. These pre-fitted instances dramatically reduce test runtime by avoiding re-fitting VARWG models for each test.

## Generating the Cache

To generate or regenerate the VARWG cache:

```bash
# From the weathercop project root:
python src/weathercop/tests/generate_vg_cache.py

# Force regenerate (even if cache already exists):
python src/weathercop/tests/generate_vg_cache.py --regenerate
```

The script will:
1. Load the test dataset from `~/data/opendata_dwd/multisite_testdata.nc`
2. Fit VARWG instances for each station with seasonal distributions
3. Save cache files in the standard Multisite structure:
   ```
   fixtures/
   ├── station_a/
   │   └── station_a_{identifier}.pkl
   ├── station_b/
   │   └── station_b_{identifier}.pkl
   └── vg_cache_ready  (marker file)
   ```

**Note:** This process takes 10-30 minutes depending on system performance.

## How It Works

### Test Execution

1. **With cache** (default if `vg_cache_ready` marker exists):
   - `multisite_instance` fixture loads cached VARWG instances
   - `reinitialize_vgs=False` - skips re-fitting
   - Tests run in minutes instead of hours

2. **Without cache**:
   - `multisite_instance` fixture fits VARWG instances fresh
   - `reinitialize_vgs=True` - performs full fitting
   - Tests run slowly but still pass

### Files

- `vg_cache_ready` - Marker file indicating cache is ready
- `station_*/` - Directories containing cached VARWG instances for each station
- `README.md` - This file

## When to Regenerate

Regenerate the cache if:
- VARWG library is updated with breaking changes
- Test data is modified
- You need to ensure reproducibility
- Cache becomes corrupted

Use: `python src/weathercop/tests/generate_vg_cache.py --regenerate`

## Troubleshooting

**Cache not found:**
```
RuntimeError: Test data not found at ~/data/opendata_dwd/multisite_testdata.nc
```
Solution: Ensure test data exists at the expected location.

**Cache fails to generate:**
Check system memory - VARWG fitting is memory-intensive. Consider closing other applications.

**Tests still slow:**
Verify `vg_cache_ready` marker exists and cache files are present in subdirectories.
