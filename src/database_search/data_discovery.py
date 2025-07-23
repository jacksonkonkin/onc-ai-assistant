import requests
import json

API_TOKEN = "b77b663d-e93b-40a3-a653-dfccb4a1b0cb"
BASE_URL = "https://data.oceannetworks.ca/api"


def _call_onc_api(endpoint: str, **filters):
    """Helper to make ONC API calls"""
    try:
        params = {'token': API_TOKEN, **filters}
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        
        if response.status_code != 200:
            return None, f"Error: HTTP {response.status_code}"
        
        return response.json(), None
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def process_location_node(location):
    """Process a location node and its children into a flat list"""
    location_without_children = location.copy()
    location_without_children.pop('children', None)
    current_data = [location_without_children]
    
    if location['children'] is not None:
        for child in location['children']:
            child_locations = process_location_node(child)
            current_data.extend(child_locations)
    
    return current_data

def discover_locations_tree(**filters):
    """Find locations tree"""
    all_locations = []
    data, error = _call_onc_api('locations/tree', **filters)
    result = ""
    if data:
        for location in data:
            locations = process_location_node(location)
            all_locations.extend(locations)
        for location in all_locations:
            result += f"Location name: {location.get('locationName', 'Unknown')}\n"
            result += f"Location code: {location.get('locationCode', 'Unknown')}\n"
            result += f"Description: {location.get('description', 'No description')}\n"
            result += f"Has device data: {location.get('hasDeviceData', 'Unknown')}\n"
            result += f"Has property data: {location.get('hasPropertyData', 'Unknown')}\n"
            result += "-" * 50 + "\n\n"
        return result
    
    return None

def get_location_and_children(location_code: str):
    """
    Get a specific location and all its children from the location tree.
    """
    def find_location_in_tree(locations, target_code):
        """Recursively search for a location in the tree"""
        for location in locations:
            if location.get('locationCode') == target_code:
                return location
            if location.get('children'):
                found = find_location_in_tree(location['children'], target_code)
                if found:
                    return found
        return None
    
    tree_data, error = _call_onc_api('locations/tree')
    if error:
        return []
    
    if not tree_data:
        return []
    
    target_location = find_location_in_tree(tree_data, location_code.upper())
    
    if not target_location:
        return []
    
    return process_location_node(target_location)


def discover_devices(**filters):
    """Find devices/instruments"""
    print(f"filters: {filters}")
    
    # if location code is provided, get all child locations too
    location_codes = []
    if 'locationCode' in filters:
        location_tree = get_location_and_children(filters['locationCode'])
        location_codes = [loc.get('locationCode') for loc in location_tree if loc.get('locationCode')]
        print(f"Searching in locations: {location_codes}")
    
    all_devices = []
    
    if location_codes:
        # search devices in the main location and all children
        for loc_code in location_codes:
            search_filters = filters.copy()
            search_filters['locationCode'] = loc_code
            
            data, error = _call_onc_api('devices', **search_filters)
            if error:
                continue
            
            if data:
                # add location info to each device for clarity
                for device in data:
                    device['searchLocationCode'] = loc_code
                all_devices.extend(data)
    else:
        # no location specified, search normally
        data, error = _call_onc_api('devices', **filters)
        if error:
            return error
        all_devices = data or []
    
    if not all_devices:
        return "No devices found"
    
    # remove duplicates 
    unique_devices = []
    seen_device_codes = set()
    for device in all_devices:
        device_code = device.get('deviceCode')
        if device_code not in seen_device_codes:
            unique_devices.append(device)
            seen_device_codes.add(device_code)
    
    result = f"Found {len(unique_devices)} unique devices:\n"
    
    for device in unique_devices:
        category = device.get('deviceCategoryCode', 'Unknown')
        location = device.get('searchLocationCode', device.get('locationCode', 'Unknown'))
        result += f"- {device.get('deviceName')} ({category}) at {location}\n"
    
    return result.strip()


def discover_properties(**filters):
    """Find measurable properties"""
    print(f"filters: {filters}")
    
    # if location code is provided, get all child locations too
    location_codes = []
    if 'locationCode' in filters:
        location_tree = get_location_and_children(filters['locationCode'])
        location_codes = [loc.get('locationCode') for loc in location_tree if loc.get('locationCode')]
        print(f"Searching in locations: {location_codes}")
    
    all_properties = []
    
    if location_codes:
        # search properties in the main location and all children
        for loc_code in location_codes:
            search_filters = filters.copy()
            search_filters['locationCode'] = loc_code
            
            data, error = _call_onc_api('properties', **search_filters)
            if error:
                continue
            
            if data:
                all_properties.extend(data)
    else:
        # no location specified, search normally
        data, error = _call_onc_api('properties', **filters)
        if error:
            return error
        all_properties = data or []
    
    if not all_properties:
        return "No properties found"
    
    # remove duplicates
    unique_properties = []
    seen_property_codes = set()
    for prop in all_properties:
        property_code = prop.get('propertyCode')
        if property_code not in seen_property_codes:
            unique_properties.append(prop)
            seen_property_codes.add(property_code)
    
    result = f"Found {len(unique_properties)} unique properties:\n"
    for prop in unique_properties:
        unit = prop.get('unitOfMeasure', '')
        unit_str = f" ({unit})" if unit else ""
        result += f"- {prop.get('propertyName')}{unit_str}\n"
    
    return result.strip()


def discover_device_categories(**filters):
    """Find device categories"""
    print(f"filters: {filters}")
    
    # if location code is provided, get all child locations too
    location_codes = []
    if 'locationCode' in filters:
        location_tree = get_location_and_children(filters['locationCode'])
        location_codes = [loc.get('locationCode') for loc in location_tree if loc.get('locationCode')]
        print(f"Searching in locations: {location_codes}")
    
    all_categories = []
    
    if location_codes:
        # search device categories in the main location and all children
        for loc_code in location_codes:
            search_filters = filters.copy()
            search_filters['locationCode'] = loc_code
            
            data, error = _call_onc_api('deviceCategories', **search_filters)
            if error:
                continue
            
            if data:
                all_categories.extend(data)
    else:
        # no location specified, search normally
        data, error = _call_onc_api('deviceCategories', **filters)
        if error:
            return error
        all_categories = data or []
    
    if not all_categories:
        return "No device categories found"
    
    # remove duplicates
    unique_categories = []
    seen_category_codes = set()
    for cat in all_categories:
        category_code = cat.get('deviceCategoryCode')
        # print(category_code)
        if category_code not in seen_category_codes:
            unique_categories.append(cat)
            seen_category_codes.add(category_code)
    
    result = f"Found {len(unique_categories)} unique device categories:\n"
    for cat in unique_categories:
        result += f"{cat.get('deviceCategoryCode')} - {cat.get('deviceCategoryName')}\n"
    
    return result.strip()


def discover_deployments(**filters):
    """Find deployment periods"""
    print(f"filters: {filters}")
    
    # if location code is provided, get all child locations too
    location_codes = []
    if 'locationCode' in filters:
        location_tree = get_location_and_children(filters['locationCode'])
        location_codes = [loc.get('locationCode') for loc in location_tree if loc.get('locationCode')]
        print(f"Searching in locations: {location_codes}")
    
    all_deployments = []
    
    if location_codes:
        # search deployments in the main location and all children
        for loc_code in location_codes:
            search_filters = filters.copy()
            search_filters['locationCode'] = loc_code
            
            data, error = _call_onc_api('deployments', **search_filters)
            if error:
                continue
            
            if data:
                all_deployments.extend(data)
    else:
        # no location specified, search normally
        data, error = _call_onc_api('deployments', **filters)
        if error:
            return error
        all_deployments = data or []
    
    if not all_deployments:
        return "No deployments found"
    
    result = f"Found {len(all_deployments)} deployments:\n"
    for dep in all_deployments:
        start = dep.get('begin', '')[:10] if dep.get('begin') else 'Unknown'
        end = dep.get('end', '')[:10] if dep.get('end') else 'Ongoing'
        result += f"- {dep.get('deviceCode')} at {dep.get('locationCode')}: {start} to {end}\n"
    
    return result.strip()


def discover_data_products(**filters):
    """Find available data products"""
    print(f"filters: {filters}")
    
    # if location code is provided, get all child locations too
    location_codes = []
    if 'locationCode' in filters:
        location_tree = get_location_and_children(filters['locationCode'])
        location_codes = [loc.get('locationCode') for loc in location_tree if loc.get('locationCode')]
        print(f"Searching in locations: {location_codes}")
    
    all_data_products = []
    
    if location_codes:
        # search data products in the main location and all children
        for loc_code in location_codes:
            search_filters = filters.copy()
            search_filters['locationCode'] = loc_code
            
            data, error = _call_onc_api('dataProducts', **search_filters)
            if error:
                continue
            
            if data:
                all_data_products.extend(data)
    else:
        # no location specified, search normally
        data, error = _call_onc_api('dataProducts', **filters)
        if error:
            return error
        all_data_products = data or []
    
    if not all_data_products:
        return "No data products found"
    
    # remove duplicates based on dataProductCode and extension
    unique_products = []
    seen_products = set()
    for dp in all_data_products:
        product_key = (dp.get('dataProductCode'), dp.get('extension'))
        if product_key not in seen_products:
            unique_products.append(dp)
            seen_products.add(product_key)
    
    result = f"Found {len(unique_products)} unique data products:\n"
    for dp in unique_products:
        result += f"- {dp.get('dataProductName')} (.{dp.get('extension')})\n"
    
    return result.strip()


def discover_locations(**filters):
    """Find locations"""
    data, error = _call_onc_api('locations', **filters)
    if error:
        return error
    
    if not data:
        return "No locations found"
    
    result = f"Found {len(data)} locations:\n"
    for loc in data:
        result += f"- {loc.get('locationName')} ({loc.get('locationCode')})\n"
    
    return result.strip()


def discover_onc_data(query_type: str, **filters):
    """
    Main function to discover ONC data. Returns simple plaintext.
    
    Args:
        query_type: 'devices', 'properties', 'device_categories', 'deployments', 'data_products', 'locations'
        **filters: locationCode, deviceCategoryCode, propertyCode, deviceCode
    
    Returns:
        Simple plaintext string
    """

    functions = {
        'devices': discover_devices,
        'properties': discover_properties,
        'device_categories': discover_device_categories,
        'deployments': discover_deployments,
        'data_products': discover_data_products,
        'locations': discover_locations_tree,
    }
    
    if query_type not in functions:
        return f"Invalid query type. Use: {', '.join(functions.keys())}"
    
    return functions[query_type](**filters)


if __name__ == "__main__":
    
    print(discover_onc_data('locations', locationCode='CBY'))

    # print("\n" + "="*50 + "\n")
    # print(discover_onc_data('device_categories', locationCode="CBY"))