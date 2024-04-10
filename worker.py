
from uvicorn.workers import UvicornWorker


class LimitedConcurrencyUvicornWorker(UvicornWorker):
    """Passing in args for the Uvicorn worker isn't possible from the """
    """gunicorn command line, so setting up a worker with the values needed """
    """for doing limited concurrency."""
    CONFIG_KWARGS = {
        "limit_concurrency": 4,
        "workers": 1,
        # "backlog": 5,
    }
