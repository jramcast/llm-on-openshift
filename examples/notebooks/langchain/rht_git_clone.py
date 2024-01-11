import os
from pathlib import Path
from git import Repo


def clone_course(sku, branch="main"):
    username = os.getenv("GIT_USER")
    password = os.getenv("GIT_PASS")

    Repo.clone_from(
        f"https://{username}:{password}@github.com/RedHatTraining/{sku}",
        to_path=Path.home() / f"courses/{sku}",
        branch=branch,
        depth=1
    )


