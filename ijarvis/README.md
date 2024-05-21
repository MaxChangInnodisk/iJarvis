# ijarvis

## Start ijarvis in background
```bash
cd "</path/to/ijarvis>"

sudo docker run \
-dt \
--privileged \
--device /dev/video0 \
--device /dev/video1 \
--device /dev/video2 \
--device /dev/video3 \
--name ijarvis \
--runtime=nvidia \
--network=host \
-v $(pwd)/src:/opt/inno/ijarvis \
-v $(pwd)/logs:/opt/inno/ijarvis/logs \
-v $(pwd)/nltk_data:/usr/local/share/nltk_data \
innodiskorg/ijarvis:v0.0.1 \
bash
```

```bsah
sudo docker start ijarvis
sudo docker exec -it ijarvis python3 app.py
```

## Stop ijarvis
```bash
sudo docker stop ijarvis
```

## Remove ijarvis
```bash
sudo docker rm ijarvis
```

## Build docker image
```bash
sudo docker build \
-t innodiskorg/ijarvis:v0.0.1 \
-f ./docker/dockerfile \
.
```