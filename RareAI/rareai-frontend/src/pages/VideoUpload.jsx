import React from "react";
import { useNavigate } from "react-router-dom";

function VideoUpload() {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen flex flex-col bg-gray-50">
            {/* Navbar with back button */}
            <nav className="bg-orange-500 text-white px-4 py-4 shadow-md">
                <div className="max-w-4xl mx-auto flex items-center justify-between">
                    <button
                        onClick={() => navigate(-1)}
                        className="text-white hover:text-gray-200 transition-colors"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                        </svg>
                    </button>
                    <h1 className="text-xl md:text-2xl font-bold">Video Upload</h1>
                    <div className="w-6"></div> {/* Spacer for balance */}
                </div>
            </nav>

            {/* Main content */}
            <main className="flex-grow flex flex-col items-center justify-center p-6">
                <div className="w-full max-w-md space-y-6">
                    {/* Upload Card */}
                    <div className="bg-white p-6 rounded-xl shadow-lg text-center">
                        <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 mb-4">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                        </div>
                        <h2 className="text-lg md:text-xl font-semibold text-gray-800 mb-2">Upload Video</h2>
                        <p className="text-sm md:text-base text-gray-600 mb-4">Select a video file from your device</p>
                        <button
                            className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-3 px-4 rounded-lg shadow transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
                            onClick={() => navigate('/voiceupload')}
                        >
                            Choose File
                        </button>

                    </div>

                    {/* Record Card */}
                    <div className="bg-white p-6 rounded-xl shadow-lg text-center">
                        <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-green-100 mb-4">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                        </div>
                        <h2 className="text-lg md:text-xl font-semibold text-gray-800 mb-2">Record Video</h2>
                        <p className="text-sm md:text-base text-gray-600 mb-4">Record a new video using your camera</p>
                        <button
                            className="w-full bg-green-500 hover:bg-green-600 text-white font-medium py-3 px-4 rounded-lg shadow transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50"
                            onClick={() => {/* Handle recording */ }}
                        >
                            Start Recording
                        </button>
                    </div>
                </div>
            </main>

            {/* Optional footer */}
            <footer className="p-4 text-center text-xs text-gray-500">
                Ensure good lighting and clear audio for best results
            </footer>
        </div>
    );
}

export default VideoUpload;