import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Sparkles, Terminal, Cpu } from 'lucide-react';

const API_URL = 'http://localhost:8000';

function App() {
  const [text, setText] = useState('');
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const fetchPredictions = useCallback(async (currentText) => {
    if (!currentText.trim()) {
      setPredictions([]);
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post(`${API_URL}/complete`, {
        text: currentText,
        top_n: 5
      });
      setPredictions(response.data.predictions);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      fetchPredictions(text);
    }, 500);
    return () => clearTimeout(timeoutId);
  }, [text, fetchPredictions]);

  const handleApplyPrediction = (word) => {
    setText((prev) => prev.trim() + ' ' + word + ' ');
  };

  return (
    <div className="container">
      <div className="glow"></div>
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
          <div style={{ background: 'rgba(255,255,255,0.05)', padding: '0.5rem', borderRadius: '12px' }}>
            <Cpu size={32} color="#00f2fe" />
          </div>
        </div>
        
        <h1>AI AutoComplete</h1>
        <p className="subtitle">Long Short-Term Memory Neural Network</p>

        <div className="input-wrapper">
          <textarea
            placeholder="Start typing your sentence..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>

        {isLoading && (
          <div className="loading-dots">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        )}

        <div className="predictions">
          {predictions.map((pred, i) => (
            <div
              key={i}
              className="prediction-chip"
              onClick={() => handleApplyPrediction(pred.word)}
              style={{ animationDelay: `${i * 0.1}s` }}
            >
              <Sparkles size={14} />
              <span>{pred.word}</span>
              <span className="prob">{Math.round(pred.probability * 100)}%</span>
            </div>
          ))}
        </div>

        <div style={{ marginTop: '3rem', display: 'flex', alignItems: 'center', gap: '8px', color: 'rgba(255,255,255,0.2)', fontSize: '0.8rem', justifyContent: 'center' }}>
          <Terminal size={14} />
          <span>Powered by TensorFlow & LSTM</span>
        </div>
      </div>
    </div>
  );
}

export default App;
