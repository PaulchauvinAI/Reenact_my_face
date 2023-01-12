# Reenact your face in a video



## To launch the app in local

Build the docker image
```
docker build -t myimage .  
```

Run uvicorn and the docker image

```
uvicorn app.main:app --host 0.0.0.0 --port 80
  
docker run -p 80:80 myimage
```

Access the simple UI provided by uvicorn at  http://0.0.0.0/docs#/



## Push and deploy image on google cloud
I removed the Google Cloud secrets so this part won't work

```
gcloud builds submit --tag gcr.io/wombo-project/myimage 

gcloud run deploy --image gcr.io/wombo-project/myimage --platform managed --port 80 --memory 1G
```


## Make API calls

curl --request GET \
  --url {url_of_your_image}

curl -X 'POST' \
  {url_of_your_image} \
  -H 'accept: application/json' \
  -d ''

curl -X 'POST' {url_of_your_image} -H 'accept: application/json'-d ''