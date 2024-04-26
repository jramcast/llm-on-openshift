import os
import re
import dotenv
import logging
import subprocess
from pathlib import Path
from github import Github
from github import Auth
from github.GithubException import UnknownObjectException, GithubException

dotenv.load_dotenv()

log = logging.getLogger(__name__)

DEFAULT_SKU_REGEX = "^(AD|AI|BFX|JB|RH|CL|DO|RES|TL|WS|PT).+$"
ALLOWED_SKU_REGEX = os.getenv("ALLOWED_SKU_REGEX", DEFAULT_SKU_REGEX)
GITHUB_USER = os.getenv("GITHUB_USER")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_ORGANIZATION = os.getenv("GITHUB_ORGANIZATION", "RedHatTraining")


def clone_or_update_rht_repos_with_content(projects_dir: Path | str):
    """
    Fetch the repos that have the "content" dir
    """
    projects_path = Path(projects_dir)
    projects_path.mkdir(parents=True, exist_ok=True)

    for repo in get_rht_repos_with_content():
        repo_path = projects_path / repo.name
        if not repo_path.exists():
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",  # Reduce clone size by fetching blobs on demand
                    f"https://{GITHUB_USER}:{GITHUB_TOKEN}@github.com/{GITHUB_ORGANIZATION}/{repo.name}",
                ],
                cwd=projects_path,
                check=True,
            )
        subprocess.run(["git", "pull"], cwd=repo_path, check=True)
        log.info(f"{repo.name} repo clone and updated in {repo_path}")


def get_rht_repos():
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "No GitHub token found. "
            "Use the GITHUB_TOKEN env variable to provide a token"
        )

    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)

    org = g.get_organization(GITHUB_ORGANIZATION)
    repos = org.get_repos()

    sku_regex = re.compile(ALLOWED_SKU_REGEX)
    for repo in repos:
        if sku_regex.match(repo.name):
            yield repo


def get_rht_repos_with_content():
    repos = get_rht_repos()
    for repo in repos:
        try:
            repo.get_contents("content")
            yield repo
        except UnknownObjectException:  # content dir does not exist
            log.debug(f"{repo.name} does not have a 'content' directory. Skipping...")
        except GithubException as err:  # content dir does not exist
            log.warning(
                f"Error while trying to get {repo.name} contents. {err.message}. Skipping..."
            )
