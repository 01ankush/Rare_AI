import React from "react";
import { useNavigate } from "react-router-dom";

function ResultsPage() {
  const navigate = useNavigate();

  // Mock data - replace with actual analysis results from your backend
  const analysisResults = {
    possibleConditions: [
      {
        name: "Generalized Anxiety Disorder",
        confidence: "78% match",
        description: "A mental health disorder characterized by persistent and excessive worry that interferes with daily activities. Our analysis detected elevated stress markers in your voice patterns and facial expressions consistent with anxiety symptoms.",
        symptomsMatch: [
          "Restlessness",
          "Fatigue",
          "Difficulty concentrating",
          "Muscle tension"
        ],
        recommendations: [
          "Consult with a mental health professional for formal diagnosis",
          "Practice mindfulness meditation (10-15 minutes daily)",
          "Regular physical exercise",
          "Consider cognitive behavioral therapy"
        ]
      },
      {
        name: "Moderate Depression",
        confidence: "65% match",
        description: "A mood disorder that causes persistent feelings of sadness and loss of interest. Your vocal patterns showed reduced pitch variability and your facial expressions displayed limited positive affect.",
        symptomsMatch: [
          "Persistent sadness",
          "Loss of interest in activities",
          "Changes in sleep patterns",
          "Low energy"
        ],
        recommendations: [
          "Schedule an appointment with a psychiatrist",
          "Maintain a regular sleep schedule",
          "Increase social interactions",
          "Consider light therapy if seasonal patterns detected"
        ]
      }
    ],
    keyObservations: [
      "Elevated stress markers in vocal analysis",
      "Micro-expressions indicating anxiety",
      "Self-reported symptoms align with 3/5 DSM-5 criteria for GAD",
      "Moderate reduction in speech prosody"
    ],
    disclaimer: "This analysis is not a medical diagnosis. Our AI system has identified potential matches based on behavioral patterns and reported symptoms. Please consult a qualified healthcare professional for accurate diagnosis and treatment."
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Navbar */}
      <nav className="bg-orange-500 text-white px-4 py-4 shadow-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <button 
            onClick={() => navigate(-1)}
            className="text-white hover:text-gray-200 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <h1 className="text-xl md:text-2xl font-bold">Your Analysis Results</h1>
          <div className="w-6"></div> {/* Spacer for balance */}
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-grow w-full p-4 sm:p-6 max-w-7xl mx-auto">
        {/* Results Summary */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h2 className="text-2xl md:text-3xl font-bold text-gray-800 mb-4">
            Based on our analysis...
          </h2>
          <p className="text-lg text-gray-600 mb-6">
            We've detected behavioral and vocal patterns that may indicate:
          </p>
          
          {/* Conditions Cards */}
          <div className="space-y-6">
            {analysisResults.possibleConditions.map((condition, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-xl font-semibold text-gray-800">{condition.name}</h3>
                  <span className="bg-orange-100 text-orange-800 text-sm font-medium px-3 py-1 rounded-full">
                    {condition.confidence}
                  </span>
                </div>
                
                <p className="text-gray-600 mb-4">{condition.description}</p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-gray-700 mb-2">Matching Symptoms:</h4>
                    <ul className="list-disc pl-5 space-y-1 text-gray-600">
                      {condition.symptomsMatch.map((symptom, i) => (
                        <li key={i}>{symptom}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-gray-700 mb-2">Recommendations:</h4>
                    <ul className="list-disc pl-5 space-y-1 text-gray-600">
                      {condition.recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Key Observations */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Key Observations From Your Analysis</h3>
          <ul className="space-y-3">
            {analysisResults.keyObservations.map((obs, index) => (
              <li key={index} className="flex items-start">
                <svg className="h-5 w-5 text-orange-500 mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-gray-600">{obs}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Disclaimer and Next Steps */}
        <div className="bg-blue-50 border border-blue-100 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-blue-800 mb-3">Important Disclaimer</h3>
          <p className="text-blue-700 mb-4">{analysisResults.disclaimer}</p>
          <button
            onClick={() => navigate('/')}
            className="bg-orange-500 hover:bg-orange-600 text-white font-medium py-2 px-6 rounded-lg shadow transition-colors duration-300"
          >
            Return to Home
          </button>
        </div>
      </main>

      {/* Footer */}
      <footer className="p-4 text-center text-xs text-gray-500 border-t border-gray-200">
        <p>RareAI Analysis Report • {new Date().toLocaleDateString()}</p>
      </footer>
    </div>
  );
}

export default ResultsPage;