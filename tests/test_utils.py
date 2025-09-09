"""
Tests for miscellaneous utilities like configuration and logging.
"""
import yaml
from pathlib import Path
from core.config import get_config
from utils.logging import YOLOLogger, LogLevel, get_logger, set_log_level
from tests.config import TEST_LOG_DIR

def test_get_config_priority(tmp_path):
    """Tests that get_config prioritizes hyp dict > hyp file > default cfg."""
    cfg = {"hyp": {"box": 5.0}}
    hyp_path = tmp_path / "h.yaml"
    hyp_path.write_text(yaml.safe_dump({"box": 6.0}))
    hyp = {"box": 7.0}
    c = get_config(cfg=cfg, hyp_path=hyp_path, hyp=hyp)
    assert c.get("box") == 7.0

def test_loss_weights_and_postprocess_defaults():
    """Tests that default config contains necessary loss and postprocess keys."""
    c = get_config()
    assert "box" in c.loss_weights and "cls" in c.loss_weights
    assert "conf_thresh" in c.postprocess_config and "iou_thresh" in c.postprocess_config

def test_logger_initialization(tmp_path):
    """Test that the YOLOLogger can be initialized."""
    logger = YOLOLogger(tmp_path, level=LogLevel.INFO)
    assert logger.writer is not None, "TensorBoard writer should be initialized"
    logger.close()

def test_log_scalars_and_metrics(tmp_path):
    """Test logging of scalar values and metric dictionaries."""
    logger = YOLOLogger(tmp_path)
    logger.info('scalar/test', 1.23, step=0)
    logger.log_metrics({'mAP50': 0.75}, step=1)
    logger.close()
    assert len(list(tmp_path.glob("events.out.tfevents.*"))) > 0

def test_log_level_filtering(tmp_path):
    """Test that log messages are filtered by log level."""
    set_log_level(LogLevel.INFO)
    logger = YOLOLogger(tmp_path, level=LogLevel.INFO)
    logger.debug("test/debug_msg", "This should not be logged.")
    logger.info("test/info_msg", "This should be logged.")
    logger.close()
    log_content = (tmp_path / "log.txt").read_text()
    assert "This should not be logged" not in log_content
    assert "This should be logged" in log_content

def test_global_logger_instance():
    """Test that get_logger() returns a singleton instance."""
    log_dir = Path(TEST_LOG_DIR) / 'global_test'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger1 = get_logger(str(log_dir))
    logger2 = get_logger()
    assert logger1 is logger2
    logger1.close()
