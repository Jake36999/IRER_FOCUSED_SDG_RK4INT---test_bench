# IRER Test Bench Scanner

## Overview
The IRER Test Bench Scanner is a standalone project designed to complement the IRER physics stack. It provides a robust framework for monitoring and assisting with development across various physics engines without requiring any modifications to the existing IRER architecture. This scanner focuses on governance, compliance, and advanced analysis of codebases, ensuring that best practices are followed during development.

## Features
- **File Traversal**: Efficiently scans directories to identify relevant files for analysis.
- **AST Parsing**: Analyzes the structure of code files using Abstract Syntax Tree (AST) techniques to validate against governance rules.
- **AI Integration**: Leverages AI functionalities to provide insights and evaluations based on the scanned data.
- **Governance Rules**: Implements a set of predefined governance rules to ensure compliance and identify potential issues in the codebase.
- **Reporting**: Generates detailed reports in both JSON and Markdown formats, summarizing scan results and governance violations.

## Project Structure
```
/irer-test-bench-scanner
├── scanner_main.py         # Entry point for the test bench scanner
├── core                    # Core functionalities
│   ├── file_walker.py      # File traversal logic
│   ├── ast_parser.py       # AST parsing logic
│   └── ai_orchestrator.py  # AI integration for analysis
├── rules                   # Governance rules
│   └── governance.yaml      # YAML file defining compliance policies
├── reports                 # Scan reports
│   ├── scan_results.json    # JSON file for scan results
│   └── scan_summary.md      # Markdown summary of scan results
├── tests                   # Unit tests
│   └── test_bundler.py      # Tests for core functionalities
└── requirements.txt        # Project dependencies
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd irer-test-bench-scanner
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the scanner, execute the following command:
```
python scanner_main.py --path <directory-path> --rules rules/governance.yaml
```
Replace `<directory-path>` with the path of the directory you want to scan.

## Contribution
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.