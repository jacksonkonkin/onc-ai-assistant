from onc import ONC
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
from typing import Optional
import time 
import traceback
import json

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
    rowLimit: int =1
    resamplePeriod: int =1

class RawDataParams(BaseModel):
    deviceCode: str
    dateFrom: Optional[str] = None
    dateTo: Optional[str] = None
    rowLimit: int = 1
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
def fetch_sensor_data(device_code: str, sensor: Optional[str], DateFrom: str, DateTo: str, row_limit: int, resample_Period: int):
    scalar_params = ScalarDataParams(
        deviceCode=device_code,
        sensorCategoryCodes=sensor,
        dateFrom=DateFrom,
        dateTo=DateTo,
        rowLimit=row_limit,
        resamplePeriod=resample_Period
    )
    data = onc.getScalardata(scalar_params.model_dump())

    '''print("===first Full API Response ===")
    print(json.dumps(data, indent=2))
    print("=========================\n")'''

    return data.get('sensorData', [])

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

        for value, timestamp, flag in zip(values, times, flags or [None]*len(values)):
            if flag == 1:
                status = "✓ Passed all QA/QC tests"
            elif flag == 2 or flag == 3:
                status = "Data probably good/failed minor QA/QC test"
            elif flag == 0:
                status = "No QA/QC tests"
            elif flag == 7:
                status = "Averaged valued"
            elif flag == 4:
                status =  "⚠ Data failed major QA/QC tests"
            else:
                status = "Interpolated value or Missing data"
            
            sentence = (
                f"{name} sensor [{category}] measuring '{property_code}': "
                f"{value:.2f} {unit} at {timestamp} from {device_name} ({status})"
            )
            results.append(sentence)

    return results


'''fucntion to get data '''
def get_cambridge_bay_scalar_sensor_data(location_code: str, Device_category: str,sensor_type: Optional[str] = None, DateFrom: Optional[str] = None, DateTo: Optional[str] = None, Row_limit: Optional[int] =1 , Resample_period: Optional[int]=1):
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

            sensor_data_list = fetch_sensor_data(device_code, sensor_type, DateFrom, DateTo, Row_limit, Resample_period)
            if not sensor_data_list:
                print(f"No {sensor_type} data from {device_name}, trying next device...")
                continue

            descriptions = describe_sensor_data(sensor_data_list, sensor_type, device_name)
            return "\n".join(descriptions)

        return f"No {sensor_type} data found from any {Device_category} devices in Cambridge Bay"

    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}"
    
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

    location_code = "CBYIP"
    device_category = "CTD" 
    sensor_type = "temperature"  #sensorCategoryCodes

    
    # default values: row_limit = 1 and resample_period= 1sec
    # if we want hourly data for 24 hours: row_limit = 24 and resample_period= 3600sec
    # if we want daily data for a month: row_limit = 31 and resample_period= 86400sec
    Row_limit = 31
    resample_Period = 86400 
    date_From="2025-01-01T00:00:00.000Z"
    date_To= "2025-02-01T00:00:00.000Z"
    

    print("Fetching latest {sensor_type} data from Cambridge Bay...\n")

    result = get_cambridge_bay_scalar_sensor_data(location_code,device_category,sensor_type,date_From,date_To,Row_limit,resample_Period)
    #result2= get_cambridge_bay_raw_device_data(location_code,device_category,"2025-02-31T00:00:00.000Z","2025-02-31T23:59:59.000Z")
    print(result)


