import requests

from ..pipeline_utils import DeforumDataObject


def get_civitai_link_from_modelid(modelId: str):
    # Fetch model details
    response = requests.get(f"https://civitai.com/api/v1/models/{modelId}")
    response.raise_for_status()
    model_data = response.json()
    download_url = model_data['modelVersions'][0]['downloadUrl']
    filename = model_data['modelVersions'][0]['files'][0]['name']
    return DeforumDataObject(filename=filename, url=download_url)
