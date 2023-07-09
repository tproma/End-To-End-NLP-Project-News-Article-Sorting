import os
import urllib.request as request
import zipfile
from pathlib import Path
from ArticleSorting.logging import logger
from ArticleSorting.utils.common import get_size
from ArticleSorting.entity import DataIngestionConfig