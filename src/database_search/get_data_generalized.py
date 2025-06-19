import requests
from datetime import datetime, timedelta, timezone
import time
import json
import os


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


#Fucntion to download files:
def get_cambridge_bay_data_products( location_code, 
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
    


if __name__ == "__main__":
    end_time = datetime(2024, 1, 11, tzinfo=timezone.utc)

    # default values: row_limit = 1 and resample_period= 1sec
    # if we want hourly data for 24 hours: row_limit = 24 and resample_period= 3600sec
    # if we want daily data for a month: row_limit = 31 and resample_period= 86400sec
    #Row_limit = 31
    #resample_Period = 86400 

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
    
    result = get_cambridge_bay_data_products(
        location_code='CBYIP', 
        relevant_device_catagory_code='CTD', 
        relevant_property_code='depth',
        relevant_dataproduct_name= 'Time Series Scalar Data',
        extension= 'txt',
        start_time='2025-01-31T00:00:00.000Z',
        end_time= '2025-01-31T23:59:59.000Z'
        )
    print(f"{result}\n")