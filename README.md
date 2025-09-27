# sheetlyy

A web app for converting photos of sheet music into MusicXML files

## How to run (production)

### RQ workers

```
source .env
uv run -- rq worker-pool -n 2
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
