import requests
import urllib3
import json
from tqdm import tqdm
def token_request():
    maximo_response = maximo_session.post('https://192.168.91.179/visual-inspection/api/tokens',
                                         json = {'grant_type': "password", 
                                                 'username': "poc",
                                                 'password': "pcvision10!"},
                                         verify = False)
    return maximo_response.status_code, json.loads(maximo_response.text)
def import_dataset(filename):
    with open(filename, 'rb') as f:
        maximo_response = maximo_session.post("https://192.168.91.179/visual-inspection/api/datasets/import",
                                                headers = {'X-Auth-Token' : token},
                                                files = {
                                                    'name' : "API_UPLOAD_TEST",
                                                    'files' : (filename, f)
                                                    },
                                                verify = False
                                                )
        print(f"dataset upload code: {maximo_response.status_code}")
        return json.loads(maximo_response.text)
def get_export (inference_url, token):
    maximo_response = maximo_session.get(inference_url+'/export',
                                         headers = {'X-Auth-Token': token,'Content-type': 'application/json'},
                                         verify = False)
    return maximo_response

if __name__ == "__main__":
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    requests.packages.urllib3.disable_warnings()
    foldername = "videos"
    insulators_url = "https://192.168.91.179/visual-inspection/api/dlapis/b574d02f-1fb8-49f7-9ecd-1f23718d0a08"
    maximo_session = requests.Session()
    maximo_response_code, token_json = token_request()
    print ("maximo token response code =% d"% maximo_response_code)
    token = token_json['token']
    #ultimate_insulator_inference_function(token)
    dataset_filename = "dataset_aug.zip"
    import_dataset(dataset_filename)