# sheetlyy

A personal web app for converting photos of sheet music into MusicXML files. This project uses image segmentation techniques and a transformer model based heavily on [homr](https://github.com/liebharc/homr), adapted to support multi-page outputs, a web-based interface, and some bug fixes.

## How to run (production)

Because I'm not going to spend hundreds of dollars running inference workers on the cloud for a personal project, when I can run them locally for free:

### RQ workers

```
source .env
uv run -- rq worker-pool -n 2 --url $REDIS_URL
```

## How to run (dev)

### Web server

```
uv run -- litestar --app=web.app:app run --reload
```

### Redis

```
sudo docker run -d --name redis -p 6379:6379 redis:8.2
```

### RQ workers

```
uv run -- rq worker-pool -n 2
```

## How to test

```
uv run pytest
```
