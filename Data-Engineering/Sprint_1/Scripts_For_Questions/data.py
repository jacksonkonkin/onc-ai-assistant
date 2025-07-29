from onc import ONC

# Initialize ONC
onc = ONC("0d9c6bcb-292a-4276-b26c-1a074a5e3b05")

# Parameters for the request
params = {
    "locationCode": "CBYIP",
}

# Get data
deployments = onc.getDeployments(params)
properties = onc.getProperties()

# Define a function to format a deployment dictionary into a string
def format_deployment(deployment):
    return (f"This device was deployed on {deployment['begin']}, "
            f"is located at latitude {deployment['lat']} and longitude {deployment['lon']}, "
            f"is at depth {deployment['depth']}, "
            f"and ended on {deployment['end']}. "
            f"Device category is {deployment['deviceCategoryCode']}, "
            f"device code is {deployment['deviceCode']}. "
            f"Has device data: {deployment['hasDeviceData']}.\n")

def format_property(property):
    return (f"This property is {property['propertyName']}, "
            f"and this property's code is {property['propertyCode']}.\n")


# Open a text file for writing
with open("deployments.txt", "w") as file:
    # Write each formatted deployment to the file
    for deployment in deployments:
        formatted_text = format_deployment(deployment)
        file.write(formatted_text)

with open("ocean_properties.txt", "w") as file:
    for property in properties:
        formatted_text = format_property(property)
        file.write(formatted_text)

active_instruments = 0

with open("devices.txt", "w") as file:
    for deployment in deployments:
        if deployment['end'] is None:
            active_instruments += 1
    file.write(f"The number of instruments currently collecting data on the Cambridge Bay coastal community observatory is {active_instruments}")




    



