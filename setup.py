import setuptools

with open("README.md", "r", encoding="utf-8") as f:
  long_description = f.read()
  
  
__version__ = "0.0.0"

REPO_NAME = "News_Article_Sort"
AUTHOR_USER_NAME = "tproma"
SRC_REPO = "textSort"
AUTHOR_EMAIL = "tanjinaproma@gmail.com"


setuptools.setup(
    name = SRC_REPO,
    version = __version__,
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    deescription = "Pyhton package for NLP app",
    long_description = long_description,
    long_description_content = "text/markdown",
    url=f"https://github.com/tproma/End-To-End-NLP-Project-News-Article-Sorting.git",
    project_urls = {
      "Bug Tracker" : f"https://github.com/tproma/End-To-End-NLP-Project-News-Article-Sorting.git/issues",
    },
  
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where = "src")
  
)