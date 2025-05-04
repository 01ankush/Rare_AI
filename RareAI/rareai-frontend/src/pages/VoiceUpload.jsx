import React from "react";
import { useNavigate } from "react-router-dom";

function VoiceUpload() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 w-full">
      {/* Navbar with back button */}
      <nav className="w-full bg-orange-500 text-white p-4 shadow-md">
        <div className="w-full max-w-4xl mx-auto flex items-center justify-between">
          <button 
            onClick={() => navigate(-1)}
            className="text-white hover:text-gray-200 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <h1 className="text-xl md:text-2xl font-bold">Voice Upload</h1>
          <div className="w-6"></div> {/* Spacer for balance */}
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-grow w-full flex flex-col items-center justify-center p-4 sm:p-6">
        <div className="w-full max-w-md px-4">
          {/* Upload Card */}
          <div className="bg-white p-6 rounded-xl shadow-lg text-center w-full mb-6">
            <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <h2 className="text-lg md:text-xl font-semibold text-gray-800 mb-2">Upload Voice Recording</h2>
            <p className="text-sm md:text-base text-gray-600 mb-4">Select an audio file from your device</p>
            <button 
              className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-3 px-4 rounded-lg shadow transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
              onClick={() => navigate("/symptoms")}
            >
              Choose Audio File
            </button>
          </div>

          {/* Record Card */}
          <div className="bg-white p-6 rounded-xl shadow-lg text-center w-full">
            <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100 mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            <h2 className="text-lg md:text-xl font-semibold text-gray-800 mb-2">Record Voice</h2>
            <p className="text-sm md:text-base text-gray-600 mb-4">Record a new audio using your microphone</p>
            <button 
              className="w-full bg-green-500 hover:bg-green-600 text-white font-medium py-3 px-4 rounded-lg shadow transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50"
              onClick={() => {/* Handle voice recording */}}
            >
              Start Recording
            </button>
          </div>
        </div>
      </main>

      {/* Optional footer */}
      <footer className="w-full p-4 text-center text-xs text-gray-500">
        <div className="max-w-4xl mx-auto">
          Speak clearly in a quiet environment for best results
        </div>
      </footer>
    </div>
  );
}

export default VoiceUpload;