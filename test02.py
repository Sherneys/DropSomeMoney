from google.cloud import automl
import os, cv2

# Setup
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="client_secrets.json"
file_path = 'Temp.jpg'
project_id = ''
model_id = ''

def capture_pic():
    cam = cv2.VideoCapture(1)
  
    # reading the input using the camera
    result, image = cam.read()
    
    # If image will detected without any error, 
    # show result
    if result:
    
        # showing result, it take frame name and image 
        # output
        cv2.imshow("Test", image)
    
        # saving image in local storage
        cv2.imwrite("Temp.jpg", image)
    
        # If keyboard interrupt occurs, destroy image 
        # window
        cv2.destroyWindow("Test")
    
    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")

def get_prediction(file_path, project_id, model_id):
    prediction_client = automl.PredictionServiceClient()

    model_full_id = automl.AutoMlClient.model_path(project_id, "us-central1", model_id)

    with open(file_path, "rb") as content_file:
        content = content_file.read()

    image = automl.Image(image_bytes=content)
    payload = automl.ExamplePayload(image=image)
    params = {"score_threshold": "0.0"}

    request = automl.PredictRequest(name=model_full_id, payload=payload, params=params)
    response = prediction_client.predict(request=request)

    return response


capture_pic()
print("Prediction results:")
max = 0
for result in get_prediction(file_path, project_id, model_id).payload:
    print("Predicted class name: {}".format(result.display_name))
    print("Predicted class score: {}".format(result.classification.score))
    if result.classification.score > max:
        name_max = result.display_name
        max = result.classification.score
print(name_max , max)

if name_max == "1000bank" and max > 0.9999 :
    print (True)
else:
    print(False)
