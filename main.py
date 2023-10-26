from ArticleSorting.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from ArticleSorting.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from ArticleSorting.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from ArticleSorting.logging import logger



STAGE_NAME = "Data Ingestion stage"
try: 
    logger.info(f"........stage {STAGE_NAME} started.......")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"........stage {STAGE_NAME} completed.......")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME = "Data Validation stage"
try: 
    logger.info(f"........stage {STAGE_NAME} started.......")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f"........stage {STAGE_NAME} completed.......")

except Exception as e:
    logger.exception(e)
    raise e
