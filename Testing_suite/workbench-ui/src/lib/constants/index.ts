// Frontend mirror of backend bundler_constants.py
export const DANGEROUS_PATTERNS = [
  'eval', 'exec', 'compile', 'subprocess', 'os.system', 'pickle',
];

export const DEFAULT_IGNORE_DIRS = [
  '.venv', 'venv', 'env', 'virtualenv', '.virtualenv', '.envs', '__pycache__', '.pytest_cache', '.mypy_cache', 'site-packages', 'dist-packages',
  '.git', '.git/objects', '.git/refs', '.git/hooks', 'dist', 'build', 'target', 'vendor', 'wheelhouse',
  'lib', 'lib64', 'bin', 'share', '.local', 'conda', 'opt',
];
