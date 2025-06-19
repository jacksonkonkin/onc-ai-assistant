from onc import ONC
import re

onc = ONC("45b4e105-43ed-411e-bd1b-1d2799eda3c4")

# Step 1: Get CBY and all its child locations
location_info = onc.getLocations({
    "locationCode": "CBY",
    "includeChildren": True
})

# Step 2: Extract all location codes
location_codes = [loc['locationCode'] for loc in location_info]

# Step 3: Get instruments from all locations
all_devices = []
seen_categories = set()

for code in location_codes:
    params = {"locationCode": code}
    devices = onc.getDeviceCategories(params)
    for device in devices:
        category = device.get('deviceCategoryName')
        if category not in seen_categories:
            seen_categories.add(category)
            all_devices.append(category)

# Step 4: Print results
print("The instruments in the Cambridge Bay Observatory are:\n")
for i, device_name in enumerate(all_devices, 1):
    print(f"{i}. {device_name}")

print(f"\nTotal number of unique instrument categories: {len(all_devices)}")
