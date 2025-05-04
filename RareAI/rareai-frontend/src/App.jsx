import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import VideoUpload from "./pages/VideoUpload";
import VoiceUpload from "./pages/VoiceUpload";
import SymptomsPage from "./pages/SymptomsPage";
import ResultsPage from "./pages/ResultsPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/videoupload" element={<VideoUpload />} />
        <Route path="/voiceupload" element={<VoiceUpload />} />
        <Route path="/symptoms" element={<SymptomsPage />} />
        <Route path="/results" element={<ResultsPage />} />
      </Routes>
    </Router>
  );
}

export default App;
