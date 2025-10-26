# WeatherCop Examples

## quickstart.py - Minimal Example

A 15-line example that demonstrates:
- Loading test data
- Initializing Multisite
- Generating 5-member ensemble
- Creating meteogram visualization

**Run it:**
```bash
python quickstart.py
```

**Output:** PNG files with ensemble visualizations

**Runtime:** ~2-5 minutes (first run may take longer for vine fitting)

## notebook_tutorial.ipynb - Detailed Tutorial

A Jupyter notebook with step-by-step explanations:
- What Multisite does (concept overview)
- How to load and inspect data
- Key parameters and their effects
- Running ensemble generation
- Interpreting visualizations
- Customization examples

**Run it:**
```bash
jupyter notebook notebook_tutorial.ipynb
```

## Using Real DWD Data

For production use with real German Weather Service data, see:
- `dwd_opendata` package documentation
- Main README for integration with VARWG

## Tips

- First run is slow (vine fitting takes 5-10 minutes)
- Subsequent runs with same data are much faster
- Use test data for learning
- Scale up to real data once comfortable with API
