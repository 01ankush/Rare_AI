const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');
const { spawn } = require('child_process');

// Multer setup
const storage = multer.diskStorage({
  destination: 'uploads/',
  filename: (req, file, cb) => {
    cb(null, `voice_${Date.now()}${path.extname(file.originalname)}`);
  }
});
const upload = multer({ storage });

// POST /api/voice-upload
router.post('/', upload.single('audio'), (req, res) => {
  const inputPath = req.file.path;
  const outputPath = inputPath.replace(path.extname(inputPath), '.wav');

  // Convert to WAV if not already
  const convertToWav = () => {
    return new Promise((resolve, reject) => {
      ffmpeg(inputPath)
        .toFormat('wav')
        .on('end', () => resolve(outputPath))
        .on('error', reject)
        .save(outputPath);
    });
  };

  const runPredictionScript = (wavPath) => {
    return new Promise((resolve, reject) => {
      const py = spawn('python', ['models/voice_from_wav.py', wavPath]);
      let output = '';

      py.stdout.on('data', (data) => output += data.toString());
      py.stderr.on('data', (err) => console.error(`Python Error: ${err}`));
      py.on('close', (code) => {
        if (code === 0) resolve(parseInt(output.trim()));
        else reject('Python script failed.');
      });
    });
  };

  (async () => {
    try {
      const wavPath = path.extname(inputPath) === '.wav' ? inputPath : await convertToWav();
      const prediction = await runPredictionScript(wavPath);
      res.json({ prediction });
      // Clean up files
      fs.unlinkSync(inputPath);
      if (wavPath !== inputPath) fs.unlinkSync(wavPath);
    } catch (err) {
      console.error(err);
      res.status(500).json({ error: 'Processing failed' });
    }
  })();
});

module.exports = router;
