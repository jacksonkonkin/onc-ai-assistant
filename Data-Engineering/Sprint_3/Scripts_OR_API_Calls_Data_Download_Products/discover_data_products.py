"""
Oceans 3.0 API Data Product Discovery Examples

This script demonstrates how to use the ONC (Oceans 3.0) Python API to discover
data products and filter them by code, extension, name, location, device, or category.

Replace "YOUR_TOKEN" with your actual Oceans 3.0 API token.

References:
- https://wiki.oceannetworks.ca/display/O2A/ONC+Python+API
"""

from onc import ONC

# ----------------------------------------------------------
# 1. Initialize ONC API client with your personal API token.
# ----------------------------------------------------------
# Get token from Oceans 3.0 profile page!
onc = ONC("YOUR_TOKEN")

# ----------------------------------------------------------
# 2. Get all data products and their parameters.
# ----------------------------------------------------------
# This will return a list of all available data products and their options.
data_products = onc.getDataProducts()
# Example output: [{'dataProductCode': 'TSSD', ...}, {'dataProductCode': 'RPSD', ...}, ...]

# ----------------------------------------------------------
# 3. Filter by Data Product Code (e.g., "TSSD" = Time Series Scalar Data)
# ----------------------------------------------------------
params = {
    "dataProductCode": "TSSD"
}
tssd_products = onc.getDataProducts(params)
# Returns only the data product options for "TSSD"

# ----------------------------------------------------------
# 4. Filter by File Extension (e.g., "pdf")
# ----------------------------------------------------------
params = {
    "extension": "pdf"
}
pdf_products = onc.getDataProducts(params)
# Returns data products available as PDFs

# ----------------------------------------------------------
# 5. Filter by both Data Product Code and File Extension
#    (e.g., TSSD data products as CSV files)
# ----------------------------------------------------------
params = {
    "dataProductCode": "TSSD",
    "extension": "csv"
}
tssd_csv_products = onc.getDataProducts(params)
# Returns only "TSSD" products that are available as CSV

# ----------------------------------------------------------
# 6. Filter by Data Product Name (search for a word in the name, e.g., "scalar")
# ----------------------------------------------------------
params = {
    "dataProductName": "scalar"
}
scalar_products = onc.getDataProducts(params)
# Returns all data products with "scalar" in their name

# ----------------------------------------------------------
# 7. Filter by Location (e.g., Barkley Canyon Axis, locationCode "BACAX")
# ----------------------------------------------------------
params = {
    "locationCode": "BACAX"
}
bacax_products = onc.getDataProducts(params)
# Returns data products available at location "BACAX"

# ----------------------------------------------------------
# 8. Filter by File Extension at a Specific Location
#    (e.g., MATLAB files at BACAX)
# ----------------------------------------------------------
params = {
    "extension": "mat",
    "locationCode": "BACAX"
}
bacax_mat_products = onc.getDataProducts(params)
# Returns all "mat" data products at location "BACAX"

# ----------------------------------------------------------
# 9. Filter by Device Code (all products for a specific deployed device)
# ----------------------------------------------------------
params = {
    "deviceCode": "NORTEKAQDPRO8398"
}
device_products = onc.getDataProducts(params)
# Returns data products available for the device "NORTEKAQDPRO8398"

# ----------------------------------------------------------
# 10. Filter by Device Category (e.g., Acoustic Doppler Current Profiler 150 kHz)
# ----------------------------------------------------------
params = {
    "deviceCategoryCode": "ADCP150KHZ"
}
category_products = onc.getDataProducts(params)
# Returns all data products for device category "ADCP150KHZ"




# ----------------------------------------------------------
# All API call formats are above and below are just the print
# functions to display the results in a readable format.
# ----------------------------------------------------------








# ----------------------------------------------------------
# 11. Functions to print the results in a readable format
# ----------------------------------------------------------

def pprint_data_products_and_params(products):
    """
    Prints a friendly paragraph description for each data product in the list.
    """
    for idx, prod in enumerate(products, 1):
        paragraph = (
            f"{idx}. Data Product '{prod.get('dataProductName', '-')}' "
            f"(Code: {prod.get('dataProductCode', '-')}, File Type: {prod.get('extension', '-')}) "
            f"{'includes' if prod.get('hasDeviceData', False) else 'does not include'} device data, "
            f"and {'includes' if prod.get('hasPropertyData', False) else 'does not include'} property data. "
        )
        # Add options if present
        if prod.get('dataProductOptions'):
            options = ', '.join(str(opt) for opt in prod['dataProductOptions'])
            paragraph += f"Available options: {options}. "
        # Add help link if present
        if prod.get('helpDocument'):
            paragraph += f"More info: {prod['helpDocument']}"

        print(paragraph)
        print()

def print_filter_data_product_with_code(products):
    """
    Prints a friendly paragraph description for each data product, including its options.
    """
    for idx, prod in enumerate(products, 1):
        # Main product description
        print(f"{idx}. '{prod.get('dataProductName', '-')}' (Code: {prod.get('dataProductCode', '-')}, File Type: {prod.get('extension', '-')})")
        print(f"   - Device Data: {'Yes' if prod.get('hasDeviceData', False) else 'No'}")
        print(f"   - Property Data: {'Yes' if prod.get('hasPropertyData', False) else 'No'}")
        if prod.get('helpDocument'):
            print(f"   - Help: {prod['helpDocument']}")
        
        # Options (if any)
        options = prod.get('dataProductOptions', [])
        if options:
            print("   - Options available:")
            for opt in options:
                print(f"     * Option: {opt.get('option', '-')}")
                if opt.get('allowableValues'):
                    print(f"       Allowable Values: {', '.join(map(str, opt['allowableValues']))}")
                if opt.get('defaultValue'):
                    print(f"       Default Value: {opt['defaultValue']}")
                if opt.get('documentation'):
                    print("       Documentation Links:")
                    for doc in opt['documentation']:
                        print(f"         - {doc}")
                # Print suboptions (if present)
                if opt.get('suboptions'):
                    print("       Suboptions:")
                    for sub in opt['suboptions']:
                        print(f"         - Option: {sub.get('option', '-')}")
                        if sub.get('allowableValues'):
                            print(f"           Allowable Values: {', '.join(map(str, sub['allowableValues']))}")
                        if sub.get('defaultValue'):
                            print(f"           Default Value: {sub['defaultValue']}")
                        if sub.get('documentation'):
                            print("           Documentation Links:")
                            for doc in sub['documentation']:
                                print(f"             - {doc}")
        else:
            print("   - No options available.")
        print()


def print_data_products_by_file_extension(products):
    """
    Prints a well-formatted, friendly summary,
    including all its configuration options and documentation.
    """
    for idx, prod in enumerate(products, 1):
        print(f"{idx}. '{prod.get('dataProductName', '-')}' (Code: {prod.get('dataProductCode', '-')}, File Type: {prod.get('extension', '-')})")
        print(f"   Device Data: {'Yes' if prod.get('hasDeviceData', False) else 'No'}")
        print(f"   Property Data: {'Yes' if prod.get('hasPropertyData', False) else 'No'}")
        if prod.get('helpDocument'):
            print(f"   More Info: {prod['helpDocument']}")
        options = prod.get('dataProductOptions', [])
        if options:
            print("   Available Options:")
            for opt in options:
                line = f"     • '{opt.get('option', '-')}'"
                if opt.get('allowableValues'):
                    line += f" | Values: {', '.join(map(str, opt['allowableValues']))}"
                if opt.get('defaultValue'):
                    line += f" | Default: {opt['defaultValue']}"
                print(line)
                if opt.get('documentation'):
                    print("       Docs:")
                    for doc in opt['documentation']:
                        print(f"         - {doc}")
                if opt.get('suboptions'):
                    print("       Suboptions:")
                    for sub in opt['suboptions']:
                        sub_line = f"         - Option: {sub.get('option', '-')}"
                        if sub.get('allowableValues'):
                            sub_line += f" | Values: {', '.join(map(str, sub['allowableValues']))}"
                        if sub.get('defaultValue'):
                            sub_line += f" | Default: {sub['defaultValue']}"
                        print(sub_line)
                        if sub.get('documentation'):
                            print("           Docs:")
                            for doc in sub['documentation']:
                                print(f"             - {doc}")
        else:
            print("   No extra options available.")
        print()

def print_data_products_product_code_extension(products):
    """
    Prints a well-formatted, friendly summary,
    including all its configuration options and documentation.
    """
    for idx, prod in enumerate(products, 1):
        print(f"{idx}. '{prod.get('dataProductName', '-')}' (Code: {prod.get('dataProductCode', '-')}, File Type: {prod.get('extension', '-')})")
        print(f"   Device Data: {'Yes' if prod.get('hasDeviceData', False) else 'No'}")
        print(f"   Property Data: {'Yes' if prod.get('hasPropertyData', False) else 'No'}")
        if prod.get('helpDocument'):
            print(f"   More Info: {prod['helpDocument']}")
        options = prod.get('dataProductOptions', [])
        if options:
            print("   Available Options:")
            for opt in options:
                line = f"     • '{opt.get('option', '-')}'"
                if opt.get('allowableValues'):
                    line += f" | Values: {', '.join(map(str, opt['allowableValues']))}"
                if opt.get('defaultValue'):
                    line += f" | Default: {opt['defaultValue']}"
                print(line)
                if opt.get('documentation'):
                    print("       Docs:")
                    for doc in opt['documentation']:
                        print(f"         - {doc}")
                if opt.get('suboptions'):
                    print("       Suboptions:")
                    for sub in opt['suboptions']:
                        sub_line = f"         - Option: {sub.get('option', '-')}"
                        if sub.get('allowableValues'):
                            sub_line += f" | Values: {', '.join(map(str, sub['allowableValues']))}"
                        if sub.get('defaultValue'):
                            sub_line += f" | Default: {sub['defaultValue']}"
                        print(sub_line)
                        if sub.get('documentation'):
                            print("           Docs:")
                            for doc in sub['documentation']:
                                print(f"             - {doc}")
        else:
            print("   No extra options available.")
        print()
        
def print_named_data_products_summary(products):
    """
    Prints a readable summary for each data product filtered by product name,
    including all its configuration options, allowable values/ranges, defaults, and docs.
    """
    for idx, prod in enumerate(products, 1):
        print(f"{idx}. '{prod.get('dataProductName', '-')}' (Code: {prod.get('dataProductCode', '-')}, File Type: {prod.get('extension', '-')})")
        print(f"   Device Data: {'Yes' if prod.get('hasDeviceData', False) else 'No'}")
        print(f"   Property Data: {'Yes' if prod.get('hasPropertyData', False) else 'No'}")
        if prod.get('helpDocument'):
            print(f"   More Info: {prod['helpDocument']}")
        options = prod.get('dataProductOptions', [])
        if options:
            print("   Available Options:")
            for opt in options:
                line = f"     • '{opt.get('option', '-')}'"
                if opt.get('allowableValues'):
                    line += f" | Values: {', '.join(map(str, opt['allowableValues']))}"
                if opt.get('defaultValue'):
                    line += f" | Default: {opt['defaultValue']}"
                if opt.get('allowableRange'):
                    r = opt['allowableRange']
                    range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                    line += f" | Range: {range_desc}"
                print(line)
                if opt.get('documentation'):
                    print("       Docs:")
                    for doc in opt['documentation']:
                        print(f"         - {doc}")
                if opt.get('suboptions'):
                    print("       Suboptions:")
                    for sub in opt['suboptions']:
                        sub_line = f"         - Option: {sub.get('option', '-')}"
                        if sub.get('allowableValues'):
                            sub_line += f" | Values: {', '.join(map(str, sub['allowableValues']))}"
                        if sub.get('defaultValue'):
                            sub_line += f" | Default: {sub['defaultValue']}"
                        if sub.get('allowableRange'):
                            r = sub['allowableRange']
                            range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                            sub_line += f" | Range: {range_desc}"
                        print(sub_line)
                        if sub.get('documentation'):
                            print("           Docs:")
                            for doc in sub['documentation']:
                                print(f"             - {doc}")
        else:
            print("   No extra options available.")
        print()

def print_location_filtered_data_products(products):
    """
    Prints a friendly summary of each data product filtered by location,
    including all configuration options, ranges, default values, and documentation.
    """
    for idx, prod in enumerate(products, 1):
        print(f"{idx}. '{prod.get('dataProductName', '-')}' (Code: {prod.get('dataProductCode', '-')}, File Type: {prod.get('extension', '-')})")
        print(f"   Device Data: {'Yes' if prod.get('hasDeviceData', False) else 'No'}")
        print(f"   Property Data: {'Yes' if prod.get('hasPropertyData', False) else 'No'}")
        if prod.get('helpDocument'):
            print(f"   More Info: {prod['helpDocument']}")
        options = prod.get('dataProductOptions', [])
        if options:
            print("   Available Options:")
            for opt in options:
                line = f"     • '{opt.get('option', '-')}'"
                if opt.get('allowableValues'):
                    line += f" | Values: {', '.join(map(str, opt['allowableValues']))}"
                if opt.get('defaultValue'):
                    line += f" | Default: {opt['defaultValue']}"
                if opt.get('allowableRange'):
                    r = opt['allowableRange']
                    range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                    line += f" | Range: {range_desc}"
                print(line)
                if opt.get('documentation'):
                    print("       Docs:")
                    for doc in opt['documentation']:
                        print(f"         - {doc}")
                if opt.get('suboptions'):
                    print("       Suboptions:")
                    for sub in opt['suboptions']:
                        sub_line = f"         - Option: {sub.get('option', '-')}"
                        if sub.get('allowableValues'):
                            sub_line += f" | Values: {', '.join(map(str, sub['allowableValues']))}"
                        if sub.get('defaultValue'):
                            sub_line += f" | Default: {sub['defaultValue']}"
                        if sub.get('allowableRange'):
                            r = sub['allowableRange']
                            range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                            sub_line += f" | Range: {range_desc}"
                        print(sub_line)
                        if sub.get('documentation'):
                            print("           Docs:")
                            for doc in sub['documentation']:
                                print(f"             - {doc}")
        else:
            print("   No extra options available.")
        print()

def print_extension_location_filtered_products(products):
    """
    Prints a summary for data products filtered by file extension at a specific location,
    including all their options, default values, allowable values, and documentation.
    """
    for idx, prod in enumerate(products, 1):
        print(f"{idx}. '{prod.get('dataProductName', '-')}' (Code: {prod.get('dataProductCode', '-')}, File Type: {prod.get('extension', '-')})")
        print(f"   Device Data: {'Yes' if prod.get('hasDeviceData', False) else 'No'}")
        print(f"   Property Data: {'Yes' if prod.get('hasPropertyData', False) else 'No'}")
        if prod.get('helpDocument'):
            print(f"   More Info: {prod['helpDocument']}")
        options = prod.get('dataProductOptions', [])
        if options:
            print("   Available Options:")
            for opt in options:
                line = f"     • '{opt.get('option', '-')}'"
                if opt.get('allowableValues'):
                    line += f" | Values: {', '.join(map(str, opt['allowableValues']))}"
                if opt.get('defaultValue'):
                    line += f" | Default: {opt['defaultValue']}"
                if opt.get('allowableRange'):
                    r = opt['allowableRange']
                    range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                    line += f" | Range: {range_desc}"
                print(line)
                if opt.get('documentation'):
                    print("       Docs:")
                    for doc in opt['documentation']:
                        print(f"         - {doc}")
                if opt.get('suboptions'):
                    print("       Suboptions:")
                    for sub in opt['suboptions']:
                        sub_line = f"         - Option: {sub.get('option', '-')}"
                        if sub.get('allowableValues'):
                            sub_line += f" | Values: {', '.join(map(str, sub['allowableValues']))}"
                        if sub.get('defaultValue'):
                            sub_line += f" | Default: {sub['defaultValue']}"
                        if sub.get('allowableRange'):
                            r = sub['allowableRange']
                            range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                            sub_line += f" | Range: {range_desc}"
                        print(sub_line)
                        if sub.get('documentation'):
                            print("           Docs:")
                            for doc in sub['documentation']:
                                print(f"             - {doc}")
        else:
            print("   No extra options available.")
        print()

def print_device_code_filtered_products(products):
    """
    Prints a readable summary for data products filtered by device code,
    including data type, file extension, and all configuration options if present.
    """
    for idx, prod in enumerate(products, 1):
        print(f"{idx}. '{prod.get('dataProductName', '-')}' (Code: {prod.get('dataProductCode', '-')}, File Type: {prod.get('extension', '-')})")
        print(f"   Device Data: {'Yes' if prod.get('hasDeviceData', False) else 'No'}")
        print(f"   Property Data: {'Yes' if prod.get('hasPropertyData', False) else 'No'}")
        if prod.get('helpDocument'):
            print(f"   More Info: {prod['helpDocument']}")
        options = prod.get('dataProductOptions', [])
        if options:
            print("   Available Options:")
            for opt in options:
                line = f"     • '{opt.get('option', '-')}'"
                if opt.get('allowableValues'):
                    line += f" | Values: {', '.join(map(str, opt['allowableValues']))}"
                if opt.get('defaultValue'):
                    line += f" | Default: {opt['defaultValue']}"
                if opt.get('allowableRange'):
                    r = opt['allowableRange']
                    range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                    line += f" | Range: {range_desc}"
                print(line)
                if opt.get('documentation'):
                    print("       Docs:")
                    for doc in opt['documentation']:
                        print(f"         - {doc}")
                if opt.get('suboptions'):
                    print("       Suboptions:")
                    for sub in opt['suboptions']:
                        sub_line = f"         - Option: {sub.get('option', '-')}"
                        if sub.get('allowableValues'):
                            sub_line += f" | Values: {', '.join(map(str, sub['allowableValues']))}"
                        if sub.get('defaultValue'):
                            sub_line += f" | Default: {sub['defaultValue']}"
                        if sub.get('allowableRange'):
                            r = sub['allowableRange']
                            range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                            sub_line += f" | Range: {range_desc}"
                        print(sub_line)
                        if sub.get('documentation'):
                            print("           Docs:")
                            for doc in sub['documentation']:
                                print(f"             - {doc}")
        else:
            print("   No extra options available.")
        print()

def print_device_category_filtered_products(products):
    """
    Prints a readable summary for data products filtered by device category,
    including file type, data type, and all configuration options (if present).
    """
    for idx, prod in enumerate(products, 1):
        print(f"{idx}. '{prod.get('dataProductName', '-')}' (Code: {prod.get('dataProductCode', '-')}, File Type: {prod.get('extension', '-')})")
        print(f"   Device Data: {'Yes' if prod.get('hasDeviceData', False) else 'No'}")
        print(f"   Property Data: {'Yes' if prod.get('hasPropertyData', False) else 'No'}")
        if prod.get('helpDocument'):
            print(f"   More Info: {prod['helpDocument']}")
        options = prod.get('dataProductOptions', [])
        if options:
            print("   Available Options:")
            for opt in options:
                line = f"     • '{opt.get('option', '-')}'"
                if opt.get('allowableValues'):
                    line += f" | Values: {', '.join(map(str, opt['allowableValues']))}"
                if opt.get('defaultValue'):
                    line += f" | Default: {opt['defaultValue']}"
                if opt.get('allowableRange'):
                    r = opt['allowableRange']
                    range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                    line += f" | Range: {range_desc}"
                print(line)
                if opt.get('documentation'):
                    print("       Docs:")
                    for doc in opt['documentation']:
                        print(f"         - {doc}")
                if opt.get('suboptions'):
                    print("       Suboptions:")
                    for sub in opt['suboptions']:
                        sub_line = f"         - Option: {sub.get('option', '-')}"
                        if sub.get('allowableValues'):
                            sub_line += f" | Values: {', '.join(map(str, sub['allowableValues']))}"
                        if sub.get('defaultValue'):
                            sub_line += f" | Default: {sub['defaultValue']}"
                        if sub.get('allowableRange'):
                            r = sub['allowableRange']
                            range_desc = f" [{r.get('lowerBound', '-')}-{r.get('upperBound', '-')}{r.get('unitOfMeasure', '')}]"
                            sub_line += f" | Range: {range_desc}"
                        print(sub_line)
                        if sub.get('documentation'):
                            print("           Docs:")
                            for doc in sub['documentation']:
                                print(f"             - {doc}")
        else:
            print("   No extra options available.")
        print()

# print_data_products_and_params(data_products[:1])
# print_filter_data_product_with_code(tssd_products[:1])
# print_data_products_by_file_extension(pdf_products[:1])
# print_data_products_product_code_extension(tssd_csv_products[:1])
# print_named_data_products_summary(scalar_products[:1])
# print_location_filtered_data_products(bacax_products[:1])
# print_extension_location_filtered_products(bacax_mat_products[:1])
# print_device_code_filtered_products(device_products[:1])
# print_device_category_filtered_products(category_products[:1])
