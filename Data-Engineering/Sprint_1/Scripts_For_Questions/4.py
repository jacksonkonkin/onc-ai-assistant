from onc import ONC

onc = ONC("TOKEN")

# Step 1: Get CBY and all its child locations
location_info = onc.getLocations({
    "locationCode": "CBY",
    "includeChildren": True
})

# Step 2: Extract all location codes
location_codes = [loc['locationCode'] for loc in location_info]

# Step 3: Retrieve all unique property codes (case-insensitive)
all_properties = set()
for code in location_codes:
    try:
        props = onc.getProperties({"locationCode": code})
        for prop in props:
            all_properties.add(prop.get('propertyCode', '').lower())
    except Exception:
        continue

# Step 4: Define exact properties of interest for ice thickness and related conditions
direct_ice = {
    'icethickness',
    'icedraft'
}

indirect_ice = {
    'seawatertemperature',
    'airtemperature',
    'salinity',
    'pressure',
    'depth',
    'internaltemperature',
    'absolutehumidity',
    'dewpoint',
    'wetbulbtemperature',
    'solarradiation'
}

# Step 5: Filter available properties by category
available_direct_ice = sorted(p for p in all_properties if p in direct_ice)
available_indirect_ice = sorted(p for p in all_properties if p in indirect_ice)

# Step 6: Print the results
print("Sensors that directly measure or indicate ice thickness:")
for p in available_direct_ice:
    print(f" - {p}")

print("\nSensors indirectly helpful for understanding or validating ice conditions:")
for p in available_indirect_ice:
    print(f" - {p}")
