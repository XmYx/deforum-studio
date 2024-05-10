import os
import json

# Specify the folder path
folder_path = "/home/charmander/projects/deforum/presets"

# Specify the keys to be placed at the top of the dictionary
top_keys = [
    "batch_name",
    "width",
    "height",
    "sampler_name",
    "scheduler"
]

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file ends with ".txt"
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        # Read the contents of the file
        with open(file_path, "r") as file:
            file_contents = file.read()
        
        # Parse the contents as a dictionary
        data_dict = json.loads(file_contents)
        
        # Update the "batch_name" value with the filename (without ".txt")
        data_dict["batch_name"] = filename[:-4]
        
        # Replace the key "W" with "width" if it exists
        if "W" in data_dict:
            data_dict["width"] = data_dict.pop("W")
        
        # Replace the key "H" with "height" if it exists
        if "H" in data_dict:
            data_dict["height"] = data_dict.pop("H")
        
        # Set default values for "sampler_name" and "scheduler" if they don't exist
        data_dict.setdefault("sampler_name", "dpmpp_2m_sde_gpu")
        data_dict.setdefault("scheduler", "karras")
        
        # Create a new dictionary with the specified keys at the top in the exact order
        updated_dict = {key: data_dict[key] for key in top_keys if key in data_dict}
        
        # Add the remaining key-value pairs from the original dictionary
        for key, value in data_dict.items():
            if key not in top_keys:
                updated_dict[key] = value
        
        # Convert the updated dictionary back to JSON
        updated_contents = json.dumps(updated_dict, indent=4)
        
        # Write the updated contents back to the file
        with open(file_path, "w") as file:
            file.write(updated_contents)
        
        print(f"Updated {filename}")

print("Script execution completed.")