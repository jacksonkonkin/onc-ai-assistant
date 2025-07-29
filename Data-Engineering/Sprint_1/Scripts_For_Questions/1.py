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
        # Ignore errors and continue
        continue

# Step 4: Define temperature related properties (based on your list)
temperature_properties = {
    'airtemperature',
    'internaltemperature',
    'seawatertemperature',
    'wetbulbtemperature',
    'srslicebalancebuoytemperature',
    'differentialtemperature'
}

# Step 5: Filter available temperature properties found in all_properties
available_temperatures = sorted(p for p in all_properties if p in temperature_properties)

# Step 6: Separate into air/internal vs water temperatures
air_temps = {
    'airtemperature',
    'internaltemperature',
    'wetbulbtemperature',
    'differentialtemperature',
    'srslicebalancebuoytemperature'
}

water_temps = {
    'seawatertemperature'
}

available_air_temps = [p for p in available_temperatures if p in air_temps]
available_water_temps = [p for p in available_temperatures if p in water_temps]

# Step 7: Print results
print("Air/Internal Temperature Sensors:")
for t in available_air_temps:
    print(f" - {t}")

print("\nWater Temperature Sensors:")
for t in available_water_temps:
    print(f" - {t}")
