import sys
import logging

pipeline_logger = logging.getLogger("pipeline_logger")
out_shell_handler = logging.StreamHandler(sys.stdout)
out_shell_formatter = logging.Formatter("%(asctime)s %(name)s: %(message)s")
out_shell_handler.setFormatter(out_shell_formatter)
pipeline_logger.addHandler(out_shell_handler)
pipeline_logger.setLevel(logging.INFO)
