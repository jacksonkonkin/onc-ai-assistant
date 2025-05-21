from onc import ONC

onc = ONC("45b4e105-43ed-411e-bd1b-1d2799eda3c4")

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

# Step 4: Define exact properties of interest
direct_properties = {
    'ph',
    'co2partialpressure',
    'co2concentrationlinearized'
}

indirect_properties = {
    'seawatertemperature',
    'salinity',
    'pressure',
    'depth',
    'conductivity',
    'oxygen',
    'nitrateconcentration',
    'chlorophyll',
    'cdom'
}

# Step 5: Filter from the available properties
available_direct = sorted(p for p in all_properties if p in direct_properties)
available_indirect = sorted(p for p in all_properties if p in indirect_properties)

# Step 6: Print the results
print("Sensors that can directly help detect ocean acidification:")
for p in available_direct:
    print(f" - {p}")

print("\nSensors that can indirectly help (e.g., CO₂ solubility, biological feedback):")
for p in available_indirect:
    print(f" - {p}")
