import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openenv.core.env_server.http_server import create_app

from models import ETLAction, ETLObservation

from .constants import MAX_CONCURRENT_ENVS
from .etl_pipeline_doctor_environment import EtlPipelineDoctorEnvironment

app = create_app(
    EtlPipelineDoctorEnvironment,
    ETLAction,
    ETLObservation,
    env_name="etl_pipeline_doctor",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
