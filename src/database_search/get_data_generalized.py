import requests
from datetime import datetime, timedelta, timezone
import time
import json
import os
import math
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("ONC_API_KEY")
# print(API_TOKEN)
BASE_URL = "https://data.oceannetworks.ca/api"

#Fucntion to download files:
def get_cambridge_bay_data_products(
        location_code, 
        relevant_device_catagory_code, 
        relevant_property_code,
        relevant_dataproduct_name,
        extension,
        start_time,
        end_time
    ):
    
    start_total = time.time()
    if start_time == None:
        start_time = end_time - timedelta(days=1)

    try:
        # get data product code 
        step1_start = time.time()
        devices_product_discovery = requests.get(         #dataProducts discovery API call to get dataProduct code and extension 
            f"{BASE_URL}/dataProducts",
            params={  
                'token': API_TOKEN,
                'locationCode': location_code,
                'deviceCategoryCode':relevant_device_catagory_code,
                'propertyCode':relevant_property_code,
            },
            timeout=30
        )
        if devices_product_discovery.status_code != 200:
            return f"Error getting data Products: HTTP {devices_product_discovery.status_code}"
        
        relevant_data_products = devices_product_discovery.json()
        step1_time = time.time() - step1_start
    
        data_products = [d for d in relevant_data_products if d.get('dataProductName') == relevant_dataproduct_name and d.get('extension') == extension]

        if not data_products:
            total_time = time.time() - start_total
            print(f"Total time: {total_time:.2f}s")
            return f"No {relevant_dataproduct_name} found in {relevant_device_catagory_code}"
        
        #looping through each data product code 
        for data_product in data_products:
            data_product_code = data_product['dataProductCode']
            extension= data_product['extension']

             # get most recent data from device (last 10 minutes)
            step2_start = time.time()
            
            # print(f"Time range: {start_time} to {end_time}")
            print(f"data product code: {data_product_code} & extension: {extension}")

            #API call to request dataProduct delivery and retrive request ID from the response 
            request_dataproduct = requests.get(   
            f"{BASE_URL}/dataProductDelivery",
            params={
                'token': API_TOKEN,
                'method': 'request',  
                'locationCode': location_code,
                'deviceCategoryCode': relevant_device_catagory_code,
                'propertyCode': relevant_property_code,
                'dataProductCode': data_product_code,
                'extension': extension,
                'dateFrom': start_time,
                'dateTo': end_time,
            },
            timeout=30
            )
            dataproduct_request_details = request_dataproduct.json()
            requestID = dataproduct_request_details.get("dpRequestId")
         
            if requestID:
               print(f"requestID: {requestID}")
            else:
               print(f"no requestID found{dataproduct_request_details}")
               continue

            status = 'none'
            runID = None
            file_count = None
            while status != 'complete':
                #API call to run data Product devilery and retrive runID, status, and filecount
                run_dataproduct = requests.get(   
                f"{BASE_URL}/dataProductDelivery",
                params={
                'token': API_TOKEN,
                'method': 'run',   
                'dpRequestId': requestID
                },
                timeout=30
                )
                dataproduct_run_details = run_dataproduct.json()

                first_run = dataproduct_run_details[0]
                status = first_run.get("status")
                runID = first_run.get("dpRunId")
                file_count = first_run.get("fileCount")

                print(f"Status: {status}, runID: {runID}, fileCount: {file_count}")

                if status == 'complete' and runID:
                  print(f"runID: {runID}")                   #print runID when then file is ready to download
                elif status in ['error', 'cancelled']:    
                  return f"Data product request failed or was cancelled: {dataproduct_run_details}"
                else:
                  time.sleep(10)  # wait before next poll

            
            
            base_name = f"{location_code}_{data_product_code}_{runID}"
            for file_index in range(1, file_count+1):
                while True:
                    print(f"Downloading file {file_index} of {file_count}...")
                    #API call to download the file 
                    download_dataproduct = requests.get(
                         f"{BASE_URL}/dataProductDelivery",
                         params={
                         'token': API_TOKEN,
                         'method': 'download',
                         'dpRunId': runID,
                         'index': file_index
                         },
                        timeout=60  
                    )
                    if  download_dataproduct.status_code == 202:
                        print(f"File {file_index} not ready yet. Retrying...")
                        time.sleep(10)                     
                    elif download_dataproduct.status_code == 200:
                        filename = f"{base_name}_file{file_index}.{extension}"
                        with open(filename, 'wb') as f:
                            f.write(download_dataproduct.content)
                        print(f"File saved: {os.path.abspath(filename)}")
                        break  # move to next file
                    else:            
                        print(f"Error downloading file index {file_index}: HTTP {download_dataproduct.status_code}")
                        print(f"Response: {download_dataproduct.text}")
                        break ## move to next file
                  
        
        total_time = time.time() - start_total
        print(f"Total time: {total_time:.2f}s")
              
    except Exception as e:
        total_time = time.time() - start_total
        print(f"Total time (with error): {total_time:.2f}s")
        print(f"Exception details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"

def get_devices_at_location(location_code: str, device_category_code: str = None):
    """get all devices at a location, optionally filtered by category"""
    try:
        devices_response = requests.get(
            f"{BASE_URL}/devices",
            params={
                'token': API_TOKEN,
                'locationCode': location_code
            },
            timeout=30
        )
        
        if devices_response.status_code != 200:
            return None, f"Error getting devices: HTTP {devices_response.status_code}"
        
        devices = devices_response.json()
        
        if device_category_code:
            devices = [d for d in devices if d.get('deviceCategoryCode') == device_category_code]
        
        return devices, None
        
    except Exception as e:
        return None, f"Exception getting devices: {e}"

def get_sensor_data_from_device(
       device_code: str, 
       start_time: datetime, 
       end_time: datetime, 
       resample_period: int, row_limit: int
   ):
   """get sensor data from a specific device for a time range"""
   try:
       date_from = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
       date_to = end_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
       print(f"Date from: {date_from}")
       print(f"Date to: {date_to}")
       
       data_response = requests.get(
           f"{BASE_URL}/scalardata/device",
           params={
               'token': API_TOKEN,
               'deviceCode': device_code,
               'dateFrom': date_from,
               'dateTo': date_to,
               'rowLimit': row_limit,
               'resamplePeriod': resample_period,
               'getLatest': True
           },
           timeout=60
       )
       
       # save response to file
       filename = f"sensor_data_{device_code}_{int(time.time())}.json"
       with open(filename, 'w') as f:
           json.dump(data_response.json(), f, indent=2)
       print(f"Saved response to {filename}")
       
       if data_response.status_code != 200:
           return None, f"HTTP {data_response.status_code}"
       
       return data_response.json(), None
       
   except Exception as e:
       return None, f"Exception: {e}"

def find_property_sensor(sensor_data_list: list, property_code: str):
    """find the sensor that measures the specified property"""
    for sensor_info in sensor_data_list:
        if isinstance(sensor_info, dict):
            if sensor_info.get('propertyCode') == property_code:
                return sensor_info
    return None

def extract_sensor_values(sensor_data: dict):
    """extract values, times, and metadata from sensor data"""
    temp_data = sensor_data.get('data', {})
    values = temp_data.get('values', [])
    sample_times = temp_data.get('sampleTimes', [])
    unit = sensor_data.get('unitOfMeasure', '')
    sensor_name = sensor_data.get('sensorName', 'Unknown')
    
    # filter out nan values
    valid_data = []
    valid_times = []
    for i, value in enumerate(values):
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            valid_data.append(value)
            if i < len(sample_times):
                valid_times.append(sample_times[i])
    
    return {
        'values': valid_data,
        'times': valid_times,
        'unit': unit,
        'sensor_name': sensor_name,
        'total_readings': len(valid_data)
    }

def calculate_statistics(values: list, analysis_type: str):
    """calculate statistics based on analysis type"""
    if not values:
        return None, "No valid data"
    
    if analysis_type == "instant":
        return values[-1], None
    elif analysis_type == "average":
        return sum(values) / len(values), None
    elif analysis_type == "min":
        return min(values), None
    elif analysis_type == "max":
        return max(values), None
    elif analysis_type == "all":
        return {
            'latest': values[0],
            'average': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }, None
    else:
        return None, f"Unknown analysis type: {analysis_type}"

def format_result(location_code: str, device_category_code: str, property_code: str,
                 sensor_name: str, device_name: str, stat_result, unit: str, 
                 analysis_type: str, start_time: datetime, end_time: datetime, 
                 latest_time: str = None):
    """format the final result string"""
    header = f"\n[Location: {location_code}, Device Category: {device_category_code}, Property: {property_code}]"
    
    if analysis_type == "instant":
        return f"{header}\n{sensor_name} in Cambridge Bay (Latest): \n{stat_result:.2f}{unit} \nFrom {device_name} at {latest_time}"
    
    elif analysis_type in ["average", "min", "max"]:
        start_str = start_time.strftime('%B %d, %Y')
        end_str = end_time.strftime('%B %d, %Y')
        return f"{header}\n{sensor_name} in Cambridge Bay ({analysis_type.title()}): \n{stat_result:.2f}{unit} \nFrom {start_str} to {end_str}\nDevice: {device_name}"
    
    elif analysis_type == "all":
        start_str = start_time.strftime('%B %d, %Y')
        end_str = end_time.strftime('%B %d, %Y')
        result = f"{header}\n{sensor_name} in Cambridge Bay ({start_str} to {end_str}):\n"
        result += f"Latest: {stat_result['latest']:.2f}{unit}\n"
        result += f"Average: {stat_result['average']:.2f}{unit}\n"
        result += f"Minimum: {stat_result['min']:.2f}{unit}\n"
        result += f"Maximum: {stat_result['max']:.2f}{unit}\n"
        result += f"Based on {stat_result['count']} readings from {device_name}"
        return result

def calculate_optimal_resampling(start_time: datetime, end_time: datetime, analysis_type=None):
    """calculate resampling period and row limit based on time span"""

    if analysis_type == 'instant':
        return 1, 1

    time_span = end_time - start_time
    total_days = time_span.total_seconds() / 86400

    # hourly, max 24 hours
    if total_days <= 1:
        resample_period = 3600
        row_limit = 24
        return resample_period, row_limit
    
    # daily, max 31 days
    elif total_days <= 31:
        resample_period = 86400
        row_limit = 31
        return resample_period, row_limit

    elif total_days <= 365:
        resample_period = 604800
        row_limit = 52
        return resample_period, row_limit
    
    # weekly, max 2 years worth
    else:
        resample_period = 604800
        row_limit = 104
        return resample_period, row_limit

def get_cambridge_bay_sensor_data(
        location_code="CBYIP", 
        device_category_code="CTD", 
        property_code="seawatertemperature",
        start_time=None,
        end_time=None,
        analysis_type="instant"  # "average", "min", "max", "instant", "all"
    ):
    """main function to get sensor data with statistical analysis"""
    start_total = time.time()
    
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is None:
        start_time = end_time - timedelta(days=1)
    
    try:     
        resample_period, row_limit = calculate_optimal_resampling(start_time, end_time, analysis_type)
        print(f"Using resample_period: {resample_period}s, row_limit: {row_limit}")
        
        step1_start = time.time()
        devices, error = get_devices_at_location(location_code, device_category_code)
        if error:
            return error
        
        if not devices:
            return f"No {device_category_code} devices found at {location_code}"
        
        step1_time = time.time() - step1_start
        print(f"Step 1 (get devices): {step1_time:.2f}s - Found {len(devices)} devices")
        
        # try each device until we find one with data
        for device in devices:
            device_code = device['deviceCode']
            device_name = device['deviceName']
            print(f"Trying device: {device_name} ({device_code})")
            
            step2_start = time.time()
            
            data, error = get_sensor_data_from_device(
                device_code, start_time, end_time, resample_period, row_limit
            )
            
            if error:
                print(f"Error getting data from {device_name}: {error}")
                continue
            
            step2_time = time.time() - step2_start
            print(f"Step 2 (get data): {step2_time:.2f}s")
            
            sensor_data_list = data.get('sensorData', [])
            print(json.dumps(sensor_data_list, indent=2))
            if not sensor_data_list:
                print(f"No sensor data from {device_name}, trying next device...")
                continue
            
            target_sensor = find_property_sensor(sensor_data_list, property_code)
            if not target_sensor:
                print(f"No {property_code} sensor in {device_name}, trying next device...")
                continue
            
            sensor_info = extract_sensor_values(target_sensor)
            
            if not sensor_info['values']:
                print(f"No valid values from {device_name}, trying next device...")
                continue
            
            print(f"Found {sensor_info['total_readings']} valid readings from {sensor_info['sensor_name']}")
            
            stat_result, stat_error = calculate_statistics(sensor_info['values'], analysis_type)
            if stat_error:
                return f"Error calculating statistics: {stat_error}"
            
            latest_time = sensor_info['times'][-1] if sensor_info['times'] else 'unknown time'
            
            result = format_result(
                location_code, device_category_code, property_code,
                sensor_info['sensor_name'], device_name, stat_result, 
                sensor_info['unit'], analysis_type, start_time, end_time, latest_time
            )
            
            total_time = time.time() - start_total
            print(f"Total execution time: {total_time:.2f}s")
            return result
        
        # if no devices had usable data
        total_time = time.time() - start_total
        print(f"Total time: {total_time:.2f}s")
        return f"\nNo {property_code} data available from {device_category_code} devices at {location_code}"
        
    except Exception as e:
        total_time = time.time() - start_total
        print(f"Total time (with error): {total_time:.2f}s")
        print(f"Exception details: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"

if __name__ == "__main__":
    end_time = datetime.now()
    start_time = end_time - timedelta(days=31)
    result = get_cambridge_bay_sensor_data(
        location_code="CBYIP", 
        device_category_code="CTD", 
        property_code="seawatertemperature",
        start_time=start_time, 
        end_time=end_time, 
        analysis_type="all"
        )
    print(result)
    # result = get_cambridge_bay_sensor_data()
    # print(result)
    # default values: row_limit = 1 and resample_period= 1sec
    # if we want hourly data for 24 hours: row_limit = 24 and resample_period= 3600sec
    # if we want daily data for a month: row_limit = 31 and resample_period= 86400sec
    #Row_limit = 31
    #resample_Period = 86400 

    # print("Getting temperature from Cambridge Bay CTD sensors...")

    # result = get_cambridge_bay_sensor_data(
    #     location_code='CBYIP', 
    #     relevant_device_catagory_code='CTD',
    #     relevant_property_code='seawatertemperature',
    #     # end_time=end_time
    # )
    # print("DEVICE BASED RESULT")
    # print(f"{result}\n")
    # result = get_cambridge_bay_sensor_data(
    #     location_code='CBYIP', 
    #     relevant_device_catagory_code='VIDEOCAM', 
    #     relevant_property_code='focus',
    #     # end_time=end_time
    # )
    
    # print(f"{result}\n") 
    
    # result = get_cambridge_bay_data_products(
    #     location_code='CBYIP', 
    #     relevant_device_catagory_code='CTD', 
    #     relevant_property_code='seawatertemperature',
    #     relevant_dataproduct_name= 'Time Series Scalar Data',
    #     extension= 'txt',
    #     start_time='2025-01-31T00:00:00.000Z',
    #     end_time= '2025-01-31T23:59:59.000Z'
    # )
    # print(f"{result}\n")
