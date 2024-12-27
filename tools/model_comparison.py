from testeur import TestRunner
from log_analyzer import LogAnalyzer

def run_comparison(config_file):
    analyzer = LogAnalyzer()
    results = {}
    
    for model in config_file['models']:
        runner = TestRunner(
            config_file=model['configurations'],
            model_name=model['name'],
            context_size=4096,
            eval_batch_size=16384
        )
        successful, total = runner.run_tests()
        
        # Get latest log file for this model
        latest_log = max(analyzer.log_dir.glob(f"{model['name']}*.log"))
        results[model['name']] = analyzer.parse_log(latest_log)
    
    # Generate comparison plots
    analyzer.plot_comparisons(results)
    plt.savefig('model_comparison.png')
