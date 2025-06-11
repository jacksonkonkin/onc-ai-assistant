from onc import ONC
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from typing import Optional
import time 
import traceback

# Initialize ONC API
onc = ONC("9473c10d-5a6f-49bc-864b-80cfb1b0b932")

# --- Models ---
class DeviceQueryParams(BaseModel):
    locationCode: str

class ScalarDataParams(BaseModel):
    deviceCode: str
    sensorCategoryCodes: Optional[str] = None
    dateFrom: Optional[str] = None
    dateTo: Optional[str] = None
    rowLimit: int = 5

class RawDataParams(BaseModel):
    deviceCode: str
    dateFrom: Optional[str] = None
    dateTo: Optional[str] = None
    rowLimit: int = 10
    outputFormat: str 

# --- API call Functions ---

'''finds all divices codes in given device category and location'''
def get_devices(location_code: str, Device_category: str):
    params = DeviceQueryParams(locationCode=location_code)
    devices = onc.getDevices(params.model_dump())
    requested_devices = [d for d in devices if d.get('deviceCategoryCode') == Device_category]
    print(f"Found {len(requested_devices)} {Device_category} devices")
    return requested_devices

'''fetch sensor data for one device code'''
def fetch_sensor_data(device_code: str, sensor: Optional[str], DateFrom: str, DateTo: str, row_limit: int = 5):
    scalar_params = ScalarDataParams(
        deviceCode=device_code,
        sensorCategoryCodes=sensor,
        dateFrom=DateFrom,
        dateTo=DateTo,
        rowLimit=row_limit
    )
    data = onc.getScalardata(scalar_params.model_dump())
    return data.get('sensorData', [])

'''fetch raw data for a device '''
def fetch_raw_device_data(device_code: str, DateFrom: str, DateTo: str, row_limit: int, outputFormat: str):
    raw_params = RawDataParams(
        deviceCode=device_code,
        dateFrom=DateFrom,
        dateTo=DateTo,
        rowLimit=row_limit,
        outputFormat= outputFormat
    )
    data= onc.getRawdata(raw_params.model_dump())["data"]
    return data

'''fucntion to parse the all sensor data'''
def describe_sensor_data(sensor_data_list, sensor: str, device_name: str):
    results = []
    for sensor in sensor_data_list:
        name = sensor.get("sensorName", "Unknown Sensor")
        category = sensor.get("sensorCategoryCode", "Unknown Category")
        property_code = sensor.get("propertyCode", "Unknown Property")
        unit = sensor.get("unitOfMeasure", "")
        data = sensor.get("data", {})
        values = data.get("values", [])
        times = data.get("sampleTimes", [])
        flags = data.get("qaqcFlags", [])

        if not values or not times:
            continue

        value = values[-1]
        timestamp = times[-1]
        qaqc = flags[-1] if flags else None
        status = "✓ Passed QA/QC" if qaqc == 0 else "⚠ Check QA/QC"

        sentence = (
            f"{name} sensor [{category}] measuring '{property_code}': "
            f"{value:.2f} {unit} at {timestamp} from {device_name} ({status} QA/QC: Quality Assurance/Quality Control)"
        )
        results.append(sentence)

    return results

'''fucntion to get data '''
def get_cambridge_bay_scalar_sensor_data(location_code: str, Device_category: str, DateFrom: Optional[str] = None, DateTo: Optional[str] = None, sensor_type: Optional[str] = None):
    try:
        if not DateFrom or not DateTo:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=10)
            DateFrom = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            DateTo = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')

        devices = get_devices(location_code, Device_category)
        if not devices:
            return "No {Device_category} devices found"

        for device in devices:
            device_code = device['deviceCode']
            device_name = device['deviceName']
            print(f"\nTrying {device_name} ({device_code})")

            sensor_data_list = fetch_sensor_data(device_code, sensor_type, DateFrom, DateTo)
            if not sensor_data_list:
                print(f"No {sensor_type} data from {device_name}, trying next device...")
                continue

            descriptions = describe_sensor_data(sensor_data_list, sensor_type, device_name)
            return "\n".join(descriptions)

        return f"No {sensor_type} data found from any {Device_category} devices in Cambridge Bay"

    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"
    
'''fucntion to get raw data from a device '''
def get_cambridge_bay_raw_device_data(location_code: str, Device_category: str, DateFrom: Optional[str] = None, DateTo: Optional[str] = None):
    try:
        if not DateFrom or not DateTo:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=10)
            DateFrom = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            DateTo = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')

        devices = get_devices(location_code, Device_category)
        if not devices:
            return "No {Device_category} devices found"

        for device in devices:
            device_code = device['deviceCode']
            device_name = device['deviceName']
            print(f"\nTrying {device_name} ({device_code})")

            raw_data_readings = fetch_raw_device_data(device_code, DateFrom, DateTo, row_limit=10, outputFormat="object")
            if not raw_data_readings:
                print(f"No raw data from {device_name}, trying next device...")
                continue
            
            return raw_data_readings

        return f"No Raw data found from any {Device_category} devices in Cambridge Bay"

    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"

# --- Main ---
if __name__ == "__main__":

    #testing:---
    location_code = "CBYIP"
    device_category = "CTD" 
    sensor_type = "pressure" 
    print("Fetching latest {sensor_type} data from Cambridge Bay...\n")

    result1 = get_cambridge_bay_scalar_sensor_data(location_code,device_category,"2024-12-31T00:00:00.000Z","2024-12-31T23:59:59.000Z")
    #result2= get_cambridge_bay_raw_device_data(location_code,device_category,"2024-12-31T00:00:00.000Z","2024-12-31T23:59:59.000Z")
    print(result1)


