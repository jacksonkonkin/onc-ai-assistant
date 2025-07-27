"""
Oceans 3.0 API: Downloading Data Products Example Script

This script shows how to order/download data products using the ONC Python API.
Replace "YOUR_TOKEN" with Oceans 3.0 API token.
Docs: https://wiki.oceannetworks.ca/display/O2A/ONC+Python+API
"""

from onc import ONC

# 1. Initialize ONC API client
onc = ONC("45b4e105-43ed-411e-bd1b-1d2799eda3c4")

# ------------------------------------------------------------
# EXAMPLE 1: Download a PNG plot of time-series data from a CTD
# Located at Straight of Georgia East (locationCode: SEVIP)
# ------------------------------------------------------------

params_plot = {
    "locationCode": "SEVIP",          # Location: Strait of Georgia East
    "deviceCategoryCode": "CTD",      # Device type: CTD (Conductivity, Temp, Depth)
    "dataProductCode": "TSSP",        # Product: Time Series Scalar Plot
    "extension": "png",               # File format: PNG image
    "dateFrom": "2019-06-20T00:00:00.000Z",  # Start time
    "dateTo": "2019-06-21T00:00:00.000Z",    # End time
    "dpo_qualityControl": "1",        # QC filter: enabled
    "dpo_resample": "none",           # No resampling
}
# Order and download the plot (returns local filename or download info)
result_plot = onc.orderDataProduct(params_plot, includeMetadataFile=False)
print("PNG Plot Download Result:", result_plot)

# ------------------------------------------------------------
# EXAMPLE 2: Download CSV file of time series scalar data from an ADCP
# Located at Barkley Canyon Axis (locationCode: BACAX)
# ------------------------------------------------------------

params_csv = {
    "locationCode": "BACAX",              # Location: Barkley Canyon Axis
    "deviceCategoryCode": "ADCP2MHZ",     # Device type: ADCP 2 MHz
    "dataProductCode": "TSSD",            # Product: Time Series Scalar Data
    "extension": "csv",                   # File format: CSV
    "dateFrom": "2016-07-27T00:00:00.000Z",  # Start time
    "dateTo": "2016-07-28T00:00:00.000Z",    # End time
    "dpo_qualityControl": 1,              # QC filter: enabled
    "dpo_resample": "none",               # No resampling
    "dpo_dataGaps": 0,                    # Include all data (no gaps)
}
# Order and download the data (returns local filename or download info)
result_csv = onc.orderDataProduct(params_csv)
print("CSV Data Download Result:", result_csv)

"""
NOTES:
- You can change the date range, location, or dataProductCode as needed.
- The returned result is usually a dict with filenames and metadata.
- Set includeMetadataFile=True if you want extra .xml metadata file.
- See ONC Python API docs for full parameter options and advanced usage.
"""

"""
Oceans 3.0 Data Product API Parameter Reference

Parameter Descriptions & Allowed Values:

- locationCode: 
    The location/station code (e.g., "SEVIP", "BACAX") where the device is deployed.

- deviceCategoryCode:
    The general device type (e.g., "CTD" for Conductivity/Temp/Depth, "ADCP2MHZ" for Acoustic Doppler Current Profiler 2 MHz).

- dataProductCode:
    The code for the type of data product to download (e.g., "TSSP" for Time Series Scalar Plot, "TSSD" for Time Series Scalar Data).

- extension:
    The desired file type/format for the download (e.g., "png", "csv", "mat", "pdf", "txt", etc.).

- dateFrom:
    Start of time interval for requested data, in ISO 8601 format (e.g., "2019-06-20T00:00:00.000Z").

- dateTo:
    End of time interval for requested data, in ISO 8601 format (e.g., "2019-06-21T00:00:00.000Z").

- dpo_qualityControl:
    Whether to apply quality control to the data.
    Allowed values: "1" (apply QC), "0" (do not apply QC).

- dpo_resample:
    Controls how data is summarized over time intervals (resampling).
    Allowed values:
        "none"      - No resampling; original data points.
        "average"   - Average value over each interval.
        "minMax"    - Min and max values for each interval.
        "minMaxAvg" - Min, max, and average for each interval.

- dpo_dataGaps:
    Whether to include indicators for data gaps.
    Allowed values: 0 (do not mark gaps), 1 (mark data gaps).

Other dpo_* options:
    These are "data product options" specific to the data product type.
    Consult ONC documentation for all possibilities.

For a complete list of codes and advanced parameter options,
see: https://wiki.oceannetworks.ca/display/O2A/ONC+Python+API
"""