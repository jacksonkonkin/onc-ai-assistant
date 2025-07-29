"""
Oceans 3.0 API Example: Downloading and Listing Archived Files

This script shows how to:
- List available archived files for a device, location, and/or file extension
- Download archived files by filename or by search parameters
- Get direct download URLs for use with a download manager

Reference: https://wiki.oceannetworks.ca/display/O2A/ONC+Python+API
"""

from onc import ONC

# --------------------------------------------------------------
# 1. Initialize ONC API client with API token.
# --------------------------------------------------------------
onc = ONC("YOUR_TOKEN") 

# --------------------------------------------------------------
# 2. List archived files from a device by deviceCode and timerange
# --------------------------------------------------------------
params_device = {
    "deviceCode": "RDIADCP600WH25471",
    "dateFrom": "2019-06-07T00:00:00.000Z",
    "dateTo":   "2019-06-08T00:00:00.000Z",
}
device_files = onc.getArchivefile(params_device)
print("Archived files for device:", device_files)

# You can also use:
# device_files = onc.getArchivefileByDevice(params_device)
# device_files = onc.getListByDevice(params_device)

# --------------------------------------------------------------
# 3. List archived files from a device with a specific extension
# --------------------------------------------------------------
params_device_ext = {
    "deviceCode": "RDIADCP600WH25471",
    "extension": "rdi",
    "dateFrom": "2019-06-07T00:00:00.000Z",
    "dateTo":   "2019-06-08T00:00:00.000Z",
}
device_ext_files = onc.getArchivefile(params_device_ext)
print("Archived .rdi files for device:", device_ext_files)

# --------------------------------------------------------------
# 4. Download a single file by its filename
# --------------------------------------------------------------
filename = "ICLISTENHF1560_20181005T000403.000Z-spect.mat"
onc.downloadArchivefile(filename, overwrite=True)
# Also available as: onc.getFile(filename, overwrite=True)

# --------------------------------------------------------------
# 5. List archived files by location and device category for a timerange
# --------------------------------------------------------------
params_location = {
    "deviceCategoryCode": "HYDROPHONE",
    "locationCode": "SEVIP",
    "dateFrom": "2018-10-05T00:05:00.000Z",
    "dateTo":   "2018-10-05T00:06:00.000Z",
}
location_files = onc.getArchivefile(params_location)
print("Archived files for HYDROPHONE at SEVIP:", location_files)

# You can also use:
# location_files = onc.getArchivefileByLocation(params_location)
# location_files = onc.getListByLocation(params_location)

# --------------------------------------------------------------
# 6. List archived files by location, device category, and extension
# --------------------------------------------------------------
params_location_ext = {
    "deviceCategoryCode": "HYDROPHONE",
    "locationCode": "SEVIP",
    "extension": "mat",
    "dateFrom": "2018-10-05T00:05:00.000Z",
    "dateTo":   "2018-10-05T00:06:00.000Z",
}
location_ext_files = onc.getArchivefile(params_location_ext)
print("Archived .mat files for HYDROPHONE at SEVIP:", location_ext_files)

# --------------------------------------------------------------
# 7. Download all archived files that match parameters (direct download)
# --------------------------------------------------------------
# This will actually download all matching files (be careful with large time ranges!)
params_direct = {
    "deviceCategoryCode": "HYDROPHONE",
    "locationCode": "SEVIP",
    "extension": "mat",
    "dateFrom": "2018-10-05T00:05:00.000Z",
    "dateTo":   "2018-10-05T00:06:00.000Z",
}
onc.downloadDirectArchivefile(params_direct)
# Also: onc.getDirectFiles(params_direct)

# --------------------------------------------------------------
# 8. Get download URLs for all matching archived files (for download manager)
# --------------------------------------------------------------
params_urls = {
    "deviceCategoryCode": "HYDROPHONE",
    "locationCode": "SEVIP",
    "extension": "mat",
    "dateFrom": "2018-10-05T00:00:00.000Z",
    "dateTo":   "2018-10-05T00:10:00.000Z",
}
# Print URLs, one per line
print("\nDownload URLs:")
print(onc.getArchivefileUrls(params_urls, joinedWithNewline=True))

"""
NOTES:
- You can use different date ranges, locations, device codes, extensions, etc.
- Downloaded files will be saved to your working directory by default.
- For bulk downloads, use getArchivefileUrls to generate a list of direct links.
"""

"""
Oceans 3.0 Archive File API Parameters Reference

Parameter Descriptions & Allowed Values:

- deviceCode:
    The unique code of a deployed instrument/device (e.g., "RDIADCP600WH25471").

- deviceCategoryCode:
    The general category/type of device (e.g., "HYDROPHONE", "CTD", "ADCP2MHZ").

- locationCode:
    The location/station code where the device is/was deployed (e.g., "SEVIP", "BACAX").

- extension:
    File extension/type to filter archived files (e.g., "rdi", "mat", "flac", "wav", "txt", "log", etc.).

- dateFrom:
    Start of the time window for the query, in ISO 8601 format (e.g., "2018-10-05T00:05:00.000Z").

- dateTo:
    End of the time window for the query, in ISO 8601 format (e.g., "2018-10-05T00:06:00.000Z").

- filename:
    The exact filename of a specific archived file you want to download (e.g., "ICLISTENHF1560_20181005T000403.000Z-spect.mat").

General Notes:
- The parameters deviceCode, deviceCategoryCode, locationCode, extension, dateFrom, and dateTo can be combined in different ways for flexible querying.
- For device-based queries, use deviceCode.
- For location-based queries, use deviceCategoryCode and locationCode.
- To filter by file type, use the extension parameter.
- Filenames are required for direct download with downloadArchivefile or getFile.

Main Methods (from ONC Python package):
- getArchivefile(params): List matching archived files using your params.
- downloadArchivefile(filename): Download a specific file by name.
- downloadDirectArchivefile(params): Download all files matching the params directly.
- getArchivefileUrls(params): Get direct download URLs for all matching files.

For full API documentation and advanced filtering, see:
https://wiki.oceannetworks.ca/display/O2A/ONC+Python+API
"""