const http = require('http');
const path = require('path');
const fs = require('fs');

// Python detection server URL
const PYTHON_SERVER = 'http://127.0.0.1:5001';

// Supported file types
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'];
const VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'];

/**
 * Make a single HTTP request to the Python Flask server.
 */
function _sendRequest(endpoint, filePath) {
  return new Promise((resolve, reject) => {
    const postData = JSON.stringify({ file_path: filePath });

    const options = {
      hostname: '127.0.0.1',
      port: 5001,
      path: endpoint,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData),
      },
      timeout: 600000, // 10 min timeout for videos
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error(`Failed to parse response: ${data}`));
        }
      });
    });

    req.on('error', (err) => reject(err));

    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Detection request timed out'));
    });

    req.write(postData);
    req.end();
  });
}

/**
 * Send a detection request to the Python Flask server.
 * Automatically retries on connection errors while the server is starting up.
 * Retries up to 20 times, 3 seconds apart (~60 sec total wait).
 */
async function callPythonServer(endpoint, filePath) {
  const MAX_RETRIES = 20;
  const RETRY_DELAY_MS = 3000;
  const RETRYABLE_CODES = ['ECONNREFUSED', 'ECONNRESET', 'ENOTFOUND', 'EPIPE', 'ETIMEDOUT'];

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      return await _sendRequest(endpoint, filePath);
    } catch (err) {
      const isRetryable = RETRYABLE_CODES.includes(err.code);
      if (isRetryable && attempt < MAX_RETRIES) {
        console.log(`[DEEPSHIELD] Python server not ready (${err.code}), retrying in 3s... (${attempt}/${MAX_RETRIES})`);
        await new Promise(r => setTimeout(r, RETRY_DELAY_MS));
      } else if (isRetryable) {
        throw new Error('Python detection server is not running. Start it with: python python/detect.py --server');
      } else {
        throw err;
      }
    }
  }
}

const analyzeFile = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    const filePath = req.file.path;
    const ext = path.extname(req.file.originalname).toLowerCase();

    // Determine file type
    const isImage = IMAGE_EXTENSIONS.includes(ext);
    const isVideo = VIDEO_EXTENSIONS.includes(ext);

    if (!isImage && !isVideo) {
      fs.unlink(filePath, () => {});
      return res.status(400).json({
        message: 'Unsupported file type. Please upload an image or video.'
      });
    }

    console.log(`[DEEPSHIELD] Running detection: ${isVideo ? 'VIDEO' : 'IMAGE'}`);
    console.log(`[DEEPSHIELD] File: ${filePath}`);

    try {
      // Call the persistent Python server instead of spawning a new process
      const endpoint = isVideo ? '/detect-video' : '/detect';
      const result = await callPythonServer(endpoint, filePath);

      // Add file metadata
      result.fileName = req.file.originalname;
      result.fileType = isVideo ? 'video' : 'image';
      result.fileSize = req.file.size;
      result.analyzedAt = new Date().toISOString();

      // Clean up uploaded file after analysis
      setTimeout(() => {
        fs.unlink(filePath, () => {});
      }, 5000);

      console.log(`[DEEPSHIELD] Result: ${result.prediction} (${result.confidence}%)`);
      return res.json(result);
    } catch (err) {
      console.error('[DEEPSHIELD] Detection error:', err.message);

      // Clean up uploaded file on error
      fs.unlink(filePath, () => {});

      return res.status(500).json({
        message: err.message.includes('not running')
          ? 'AI detection server is starting up. Please wait 30 seconds and try again.'
          : 'AI detection failed. Please try again.',
        error: err.message
      });
    }
  } catch (error) {
    console.error('[DEEPSHIELD] Analysis error:', error);
    res.status(500).json({ message: 'Analysis failed. Please try again.' });
  }
};

module.exports = { analyzeFile };
