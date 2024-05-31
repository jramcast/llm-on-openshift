
import os
import logging
from repositories import clone_or_update_rht_repos_with_content

PROJECTS_DIR = os.getenv("PROJECTS_DIR", ".projects")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
)

if __name__ == "__main__":
    clone_or_update_rht_repos_with_content(PROJECTS_DIR)
