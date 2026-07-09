require('dotenv').config();

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

const authRoutes = require('./routes/auth');
const analyzeRoutes = require('./routes/analyze');

const app = express();
const PORT = process.env.PORT || 5000;

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Middleware
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:3000', 'http://127.0.0.1:5173'],
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Static files
app.use('/uploads', express.static(uploadsDir));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/analyze', analyzeRoutes);

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'DEEPSHIELD API is running' });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(400).json({ message: 'File too large. Maximum size is 100MB.' });
  }
  
  if (err.message === 'Unsupported file type') {
    return res.status(400).json({ message: 'Unsupported file type.' });
  }
  
  res.status(500).json({ message: 'Internal server error' });
});

app.listen(PORT, () => {
  console.log(`\n  🛡️  DEEPSHIELD API Server`);
  console.log(`  ========================`);
  console.log(`  Status:  Running`);
  console.log(`  Port:    ${PORT}`);
  console.log(`  Mode:    ${process.env.NODE_ENV || 'development'}`);
  console.log(`  URL:     http://localhost:${PORT}\n`);
});
