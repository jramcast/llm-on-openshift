import os
from pathlib import Path
from git import Repo



def clone_course(sku):
    username = os.getenv("GIT_USER")
    password = os.getenv("GIT_PASS")
    
    Repo.clone_from(
        f"https://{username}:{password}@github.com/RedHatTraining/{sku}",
        to_path=Path.home() / f"courses/{sku}",
        branch="main",
        depth=1
    )


