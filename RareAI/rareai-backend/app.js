const express = require('express');
const cors = require('cors');
const symptomsRoute = require('./routes/symptoms');
const photoRoute = require('./routes/photo');
const voiceRoute = require('./routes/voice');



const app = express();

app.use(cors());
app.use(express.json());
app.use('/api/symptoms', symptomsRoute);
app.use('/api/photo', photoRoute);
app.use('/api/voice', voiceRoute);

app.get('/api/status', (req, res) => {
  res.send({ message: 'Backend is live 🚀' });
});

module.exports = app;
