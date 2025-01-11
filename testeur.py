import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from config import settings


class TestRunner:
    def __init__(self, config_file, model_name, profile, context_size, eval_batch_size, n_predict, log_dir=None, verbose=False):
        if log_dir is None:
            self.log_dir = Path(settings.LOG_DIR)
            self.debug_dir = Path(settings.LOG_DEBUG_DIR)
        else:
            self.log_dir = Path(log_dir)
            self.debug_dir = self.log_dir / "debug"
        self.log_dir.mkdir(exist_ok=True)
        self.debug_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.model_name = model_name
        profile_name = profile if profile else model_name
        self.model_profile = settings.get_llm_profile(profile_name)
        self.context_size = context_size or self.model_profile.context_size
        self.eval_batch_size = eval_batch_size or self.model_profile.eval_batch_size
        self.n_predict = n_predict or self.model_profile.n_predict
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = self._load_config(config_file)
        self.setup_logging()
    
    
    def setup_logging(self):
        safe_model_name = self.model_name.replace('/', '_').replace('\\', '_').replace('@', '_')
        log_name = settings.LOG_NAME_FORMAT.format(
            model_name=safe_model_name,
            context_size=self.context_size,
            eval_batch_size=self.eval_batch_size,
            timestamp=datetime.now().strftime(settings.LOG_DATE_FORMAT)
        )
        
        # Main log handler - always created
        main_handler = logging.FileHandler(self.log_dir / f"{log_name}.log", encoding='utf-8')
        main_handler.setFormatter(logging.Formatter('%(message)s'))
        main_handler.setLevel(logging.INFO)
        
        handlers = [main_handler]
        
        # Debug log handler - only created in verbose mode
        if self.verbose:
            debug_handler = logging.FileHandler(self.debug_dir / f"{log_name}_debug.log", encoding='utf-8')
            debug_handler.setFormatter(logging.Formatter('%(message)s'))
            debug_handler.setLevel(logging.DEBUG)
            handlers.append(debug_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        console_handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        handlers.append(console_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            handlers=handlers
        )

    def _load_config(self, config_file):
            with open(config_file, 'r') as f:
                return json.load(f)

    def run_tests(self):
        successful_tests = 0
        total_tests = len(self.config["configurations"])
        
        for idx, test in enumerate(self.config["configurations"], 1):
            # Add model profile to command
            if "--model_profile" not in test['command']:
                test['command'] += f" --model_profile {self.model_name}"
            logging.info(f"\nTest {idx}/{total_tests}: {test['name']}\n  -command : {test['command']}")
        
            start_time = datetime.now()
            
            try:
                result = subprocess.run(
                    test['command'],
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=800
                )
                
                duration = (datetime.now() - start_time).total_seconds()
                success = result.returncode == 0
                
                if success:
                    successful_tests += 1
                
                self._log_result(test, result, duration)
                
            except subprocess.TimeoutExpired:
                logging.error("Test timed out after 720 seconds")
            except KeyboardInterrupt:
                logging.info("\nTest suite interrupted by user")
                break
            except Exception as e:
                logging.error(f"Test failed: {str(e)}")
                
        self._write_summary(successful_tests, total_tests)
        return successful_tests, total_tests

    def _log_result(self, test, result, duration):
        # Main log always gets the LLM response
        logging.info(f"\nTest: {test['name']} with command: {test['command']}")
        logging.info(f"Duration: {duration:.2f}s")
        logging.info("LLM Response:")
        logging.info(result.stdout)
        
        # Debug log gets additional technical details
        debug_logger = logging.getLogger('debug')
        debug_logger.debug(f"Status: {'Success' if result.returncode == 0 else 'Failed'}")
        if result.stderr:
            debug_logger.debug("\nErrors:")
            debug_logger.debug(result.stderr)

    def _write_summary(self, successful_tests, total_tests):
        # Write summary directly to main log file
        logging.info("\nTEST SUITE SUMMARY")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Test Suite: {self.config['test_suite']}")
        logging.info(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Success Rate: {successful_tests}/{total_tests}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Model Testing Suite')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to test configuration JSON file')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the model being tested')
    parser.add_argument('--context_size', type=int,
                       help='Context window size of the model')
    parser.add_argument('--eval_batch_size', type=int,
                       help='Evaluation batch size used in LM Studio')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory for log files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--n_predict', type=int,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--model_profile', type=str, 
                       default='default',
                       help='Name of the LLM profile to use')
    args = parser.parse_args()
    runner = TestRunner(
        config_file=args.config,
        model_name=args.model_name,
        profile=args.model_profile,
        context_size=args.context_size,
        eval_batch_size=args.eval_batch_size,
        n_predict=args.n_predict,
        log_dir=args.log_dir,
        verbose=args.verbose

    )
    successful, total = runner.run_tests()
    logging.info(f"\nTest suite completed: {successful}/{total} tests successful")

