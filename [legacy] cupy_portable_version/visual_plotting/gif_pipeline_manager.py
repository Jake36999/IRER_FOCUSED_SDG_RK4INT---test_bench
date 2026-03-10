import os
import shutil
import subprocess
import json
import logging
from datetime import datetime

GIFS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'GIFS')
ARCHIVE_DIR = os.path.join(GIFS_DIR, 'archive')
CONTROL_GIF = os.path.join(GIFS_DIR, 'control_run.gif')
PREV_BEST_GIF = os.path.join(GIFS_DIR, 'previous_best.gif')
NEW_BEST_GIF = os.path.join(GIFS_DIR, 'new_best.gif')
RENDER_META = os.path.join(GIFS_DIR, 'render_meta.json')
TRUE_GOLDEN_GIF = os.path.join(os.path.dirname(__file__), 'render_true_golden.gif')

# Remove global logging.basicConfig and use a local logger
logger = logging.getLogger('GIF_PIPELINE')

def init_gif_directory():
    os.makedirs(GIFS_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    # Copy true golden gif or create placeholder
    if os.path.exists(TRUE_GOLDEN_GIF):
        shutil.copy2(TRUE_GOLDEN_GIF, CONTROL_GIF)
        logger.info(f'Copied true golden GIF to {CONTROL_GIF}')
    else:
        # Create a 1x1 px placeholder GIF
        with open(CONTROL_GIF, 'wb') as f:
            f.write(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xFF\xFF\xFF!\xF9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;')
        logger.warning(f'No true golden GIF found, created placeholder at {CONTROL_GIF}')

def process_new_high_score(h5_filepath: str, new_sse: float):
    # Archive previous_best.gif if exists
    if os.path.exists(PREV_BEST_GIF):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive_name = f'previous_best_{timestamp}.gif'
        shutil.move(PREV_BEST_GIF, os.path.join(ARCHIVE_DIR, archive_name))
        logger.info(f'Archived previous_best.gif to {archive_name}')
    # Move new_best.gif to previous_best.gif if exists
    prev_sse = None
    if os.path.exists(NEW_BEST_GIF):
        shutil.move(NEW_BEST_GIF, PREV_BEST_GIF)
        logger.info('Cycled new_best.gif to previous_best.gif')
        # Try to get previous SSE from meta
        if os.path.exists(RENDER_META):
            try:
                with open(RENDER_META, 'r') as f:
                    meta = json.load(f)
                    prev_sse = meta.get('new_sse')
            except Exception as e:
                logger.warning(f'Could not read previous SSE from meta: {e}')
    # Render new GIF to a temp file, then atomically swap
    temp_gif = os.path.join(GIFS_DIR, "temp_new.gif")
    try:
        subprocess.run([
            'python',
            os.path.join(os.path.dirname(__file__), 'visual_analysis_suite.py'),
            '--input', h5_filepath,
            '--output', temp_gif
        ], check=True)
        os.replace(temp_gif, NEW_BEST_GIF)  # Atomic rename
        logger.info(f'Rendered new_best.gif from {h5_filepath}')
    except Exception as e:
        logger.error(f'Failed to render new_best.gif: {e}')
        if os.path.exists(temp_gif):
            os.remove(temp_gif)
        return
    # Get control SSE
    control_sse = 0.129
    # Write meta
    meta = {
        'control_sse': control_sse,
        'prev_sse': prev_sse,
        'new_sse': new_sse
    }
    with open(RENDER_META, 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f'Updated render_meta.json: {meta}')
