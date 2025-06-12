import requests
from datetime import datetime, timedelta, timezone
import time
import json

API_TOKEN = "b77b663d-e93b-40a3-a653-dfccb4a1b0cb"
BASE_URL = "https://data.oceannetworks.ca/api"

def get_cambridge_bay_sensor_data(
        location_code="CBYIP", 
        relevant_device_catagory_code="CTD", 
        relevant_property_code="seawatertemperature",
        start_time=None,
        end_time=datetime.now(timezone.utc)
        ):
    
    start_total = time.time()
    if start_time == None:
        start_time = end_time - timedelta(days=1)

    date_from = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    date_to = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')

    try:
        # get devices
        step1_start = time.time()
        devices_response = requests.get(
            f"{BASE_URL}/devices",
            params={
                'token': API_TOKEN,
                'locationCode': location_code
            },
            timeout=30
        )
        
        if devices_response.status_code != 200:
            return f"Error getting devices: HTTP {devices_response.status_code}"
        
        devices = devices_response.json()
        # print(json.dumps(devices, indent=2))
        # print('number of devices')
        # print(len(devices))
        step1_time = time.time() - step1_start
        # print(f"Step 1 (get devices): {step1_time:.2f}s")
    
        relevant_devices = [d for d in devices if d.get('deviceCategoryCode') == relevant_device_catagory_code]
        
        if not relevant_devices:
            total_time = time.time() - start_total
            print(f"Total time: {total_time:.2f}s")
            return f"No {relevant_device_catagory_code} devices found"
        
        # print(f"Found {len(relevant_devices)} {relevant_device_catagory_code} devices")
        
        # try each device until we find one with data
        for relevant_device in relevant_devices:
            device_code = relevant_device['deviceCode']
            device_name = relevant_device['deviceName']
            # print(f"Getting data from: {device_name}")
            # print(f"Device code: {device_code}")
            
            # get most recent data from device (last 10 minutes)
            step2_start = time.time()
            
            # print(f"Time range: {start_time} to {end_time}")
            
            data_response = requests.get(
                f"{BASE_URL}/scalardata/device",
                params={
                    'token': API_TOKEN,
                    'deviceCode': device_code,
                    'dateFrom': date_from,
                    'dateTo': date_to,
                    'rowLimit': 1  # only get the most recent reading
                },
                timeout=30
            )
            
            if data_response.status_code != 200:
                print(f"Error getting data from {device_name}: HTTP {data_response.status_code}")
                continue  # try next device
            
            data = data_response.json()

            # view raw data
            # print(json.dumps(data, indent=2))
            
            step2_time = time.time() - step2_start
            # print(f"Step 2 (get data fro device): {step2_time:.2f}s")
            
            # check if this device has sensor data
            sensor_data_list = data.get('sensorData', [])
            if not sensor_data_list:
                print(f"No sensor data found from {device_name}, trying next device...")
                continue
            
            # print(f"Found {len(sensor_data_list)} sensors")
            # print(json.dumps(sensor_data_list, indent=2))
            temp_sensor_data = None
            for sensor_info in sensor_data_list:
                if isinstance(sensor_info, dict):
                    property_code = sensor_info.get('propertyCode', '')
                    sensor_name = sensor_info.get('sensorName', 'Unknown')
                    
                    # print(f"Checking sensor: {sensor_name} - {property_code}")
                    
                    if property_code == relevant_property_code:
                        temp_sensor_data = sensor_info
                        # print(f"Found {relevant_property_code} sensor: {sensor_name}")
                        break
            
            if temp_sensor_data:
                temp_data = temp_sensor_data.get('data', {})
                values = temp_data.get('values', [])
                sample_times = temp_data.get('sampleTimes', [])
                unit = temp_sensor_data.get('unitOfMeasure', 'C')
                
                if values:
                    # get the most recent reading (last value)
                    latest_reading = values[-1]
                    latest_time = sample_times[-1] if sample_times else 'unknown time'
                    
                    # print(f"Got {len(values)} sensor reading(s)")
                    step_3_start = time.time()
                    # actual result
                    formatted_time = end_time.strftime('%B %d, %Y at %I:%M %p UTC')
                    result = f"{sensor_name} in Cambridge Bay: \n{latest_reading}{unit} \nFrom {device_name} at {latest_time}"
                    total_time = time.time() - start_total
                    # print(f"Step 3 (data processing): {time.time() - step_3_start:.3f}s")
                    print(f"Total execution time: {total_time:.2f}s")
                    # print()
                    return result
                else:
                    print(f"No sensor values found from {device_name}, trying next device...")
                    continue
        
        # if we get here, no devices had temperature data
        total_time = time.time() - start_total
        print(f"Total time: {total_time:.2f}s")
        return f"No {relevant_property_code} data found from any {relevant_devices} devices in Cambridge Bay"
        
    except Exception as e:
        total_time = time.time() - start_total
        print(f"Total time (with error): {total_time:.2f}s")
        print(f"Exception details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"

if __name__ == "__main__":
    end_time = datetime(2024, 1, 11, tzinfo=timezone.utc)
    # print("Getting temperature from Cambridge Bay CTD sensors...")
    result = get_cambridge_bay_sensor_data(
        location_code='CBYSS.M2', 
        relevant_device_catagory_code='METSTN', 
        relevant_property_code='airdensity',
        # end_time=end_time
        )
    print(f"{result}\n")
    result = get_cambridge_bay_sensor_data(
        location_code='CBYIP', 
        relevant_device_catagory_code='VIDEOCAM', 
        relevant_property_code='focus',
        # end_time=end_time
        )
    print(f"{result}\n")